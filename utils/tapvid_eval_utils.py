import os
import pickle
from collections import defaultdict
from pprint import pprint
from queue import Queue
from threading import Thread

import numpy as np

from data.tapvid import dataset as tapvid_dataset


def _format_model_preds_for_tapvid(vid_meta: tapvid_dataset.VideoSpec, vid_data: dict, evaluation_mode: str):
    assert evaluation_mode in ["first", "cfg"]

    query_points = np.zeros((vid_meta.num_query_pts, 3))
    gt_occluded = np.zeros((vid_meta.num_query_pts, vid_meta.num_target_frames))
    gt_tracks = np.zeros((vid_meta.num_query_pts, vid_meta.num_target_frames, 2))

    pred_occluded = np.zeros_like(gt_occluded)
    pred_tracks = np.zeros_like(gt_tracks)

    # fix evaluation scale to [256, 256]
    scale = (256 / np.array([*vid_meta.dataset_res]))[None, None]  # 1,1,2

    for j, ((pi, st, et), (pred, occ)) in enumerate(vid_data.items()):
        if evaluation_mode == "cfg":
            # CFG evaluation has two target frame pairs for evaluation: (f0, f0) and (f0, f0 + delta)
            # To reduce noise, we fix (f0, f0) prediction to GT for all models.
            pred_tracks[j, 0] = vid_meta.gt_pts[pi, st] * np.flip(np.array([*vid_meta.dataset_res]), -1)  # x,y
            pred_tracks[j, 1] = pred

            # Fix the first frame to be index 0
            query_points[j] = np.concatenate([np.array([0]), np.flip(vid_meta.gt_pts[pi, st], -1)], -1)

            gt_tracks[j, 0] = vid_meta.gt_pts[pi, st]
            gt_tracks[j, 1] = vid_meta.gt_pts[pi, et]

            gt_occluded[j, 0] = vid_meta.gt_occ[pi, st]
            gt_occluded[j, 1] = vid_meta.gt_occ[pi, et]

            # Fix (f0, f0) pred occ to GT
            pred_occluded[j, 0] = vid_meta.gt_occ[pi, st]
            pred_occluded[j, 1] = occ
        else:
            gt_pt0_xy = vid_meta.gt_pts[pi, st]
            gt_pt1_xy = vid_meta.gt_pts[pi, et]

            # First evaluation has T target frame pairs: (f0, f0)...(f0, f0 + T - 1)
            pred_tracks[pi, et] = pred
            gt_tracks[pi, et] = gt_pt1_xy
            query_points[pi] = np.array([st, gt_pt0_xy[1], gt_pt0_xy[0]])
            gt_occluded[pi, et] = vid_meta.gt_occ[pi, et]

            pred_occluded[pi, et] = occ

    # pred_tracks are in full_res space
    pred_tracks *= np.flip(scale, -1)

    # other things reading directly from the dataset are raster [0, 1]
    query_points[..., 1:] *= 256
    gt_tracks *= 256

    gt_occluded = gt_occluded.astype(bool)
    pred_occluded = pred_occluded.astype(bool)

    # add batch dimension
    query_points = np.expand_dims(query_points, 0)
    gt_occluded = np.expand_dims(gt_occluded, 0)
    gt_tracks = np.expand_dims(gt_tracks, 0)
    pred_occluded = np.expand_dims(pred_occluded, 0)
    pred_tracks = np.expand_dims(pred_tracks, 0)

    return query_points, gt_occluded, gt_tracks, pred_occluded, pred_tracks


def compute_tapvid_metrics(
    query_points,
    gt_occluded,
    gt_tracks,
    pred_occluded,
    pred_tracks,
    query_mode,
):
    """Computes TAP-Vid metrics (Jaccard, Pts. Within Thresh, Occ. Acc.)

    See the TAP-Vid paper for details on the metric computation.  All inputs are
    given in raster coordinates.  The first three arguments should be the direct
    outputs of the reader: the 'query_points', 'occluded', and 'target_points'.
    The paper metrics assume these are scaled relative to 256x256 images.
    pred_occluded and pred_tracks are your algorithm's predictions.

    This function takes a batch of inputs, and computes metrics separately for
    each video.  The metrics for the full benchmark are a simple mean of the
    metrics across the full set of videos.  These numbers are between 0 and 1,
    but the paper multiplies them by 100 to ease reading.

    Args:
       query_points: The query points, an in the format [t, y, x].  Its size is
         [b, n, 3], where b is the batch size and n is the number of queries
       gt_occluded: A boolean array of shape [b, n, t], where t is the number
         of frames.  True indicates that the point is occluded.
       gt_tracks: The target points, of shape [b, n, t, 2].  Each point is
         in the format [x, y]
       pred_occluded: A boolean array of predicted occlusions, in the same
         format as gt_occluded.
       pred_tracks: An array of track predictions from your algorithm, in the
         same format as gt_tracks.
       query_mode: Either 'first' or 'strided', depending on how queries are
         sampled.  If 'first', we assume the prior knowledge that all points
         before the query point are occluded, and these are removed from the
         evaluation.

    Returns:
        A dict with the following keys:

        occlusion_accuracy: Accuracy at predicting occlusion.
        pts_within_{x} for x in [1, 2, 4, 8, 16]: Fraction of points
          predicted to be within the given pixel threshold, ignoring occlusion
          prediction.
        jaccard_{x} for x in [1, 2, 4, 8, 16]: Jaccard metric for the given
          threshold
        average_pts_within_thresh: average across pts_within_{x}
        average_jaccard: average across jaccard_{x}
    """
    assert gt_occluded.shape == pred_occluded.shape

    metrics = {}

    # Don't evaluate the query point.  Numpy doesn't have one_hot, so we
    # replicate it by indexing into an identity matrix.
    one_hot_eye = np.eye(gt_tracks.shape[2])
    query_frame = query_points[..., 0]
    query_frame = np.round(query_frame).astype(np.int32)
    evaluation_points = one_hot_eye[query_frame] == 0

    # If we're using the first point on the track as a query, don't evaluate the
    # other points.
    if query_mode == "first":
        assert gt_occluded.shape[0] == 1, "Expected batch size 1 gt_occluded"
        for i in range(gt_occluded.shape[1]):
            index = np.where(gt_occluded[0, i] == 0)[0][0]
            evaluation_points[0, i, :index] = False
    elif query_mode != "strided":
        raise ValueError("Unknown query mode " + query_mode)

    # breakpoint()
    # Occlusion accuracy is simply how often the predicted occlusion equals the
    # ground truth.
    occ_acc = np.sum(
        np.equal(pred_occluded, gt_occluded) & evaluation_points,
        axis=(1, 2),
    ) / np.sum(evaluation_points)
    metrics["occlusion_accuracy"] = occ_acc

    # Added by DW based on code by SK
    metrics["occ_tp"] = np.sum(np.equal(pred_occluded, gt_occluded) & gt_occluded & evaluation_points, axis=(1, 2))
    metrics["occ_fp"] = np.sum(
        np.logical_not(np.equal(pred_occluded, gt_occluded)) & pred_occluded & evaluation_points, axis=(1, 2)
    )
    metrics["occ_fn"] = np.sum(
        np.logical_not(np.equal(pred_occluded, gt_occluded)) & np.logical_not(pred_occluded) & evaluation_points,
        axis=(1, 2),
    )

    # Next, convert the predictions and ground truth positions into pixel
    # coordinates.
    visible = np.logical_not(gt_occluded)
    pred_visible = np.logical_not(pred_occluded)
    all_frac_within = []
    all_jaccard = []
    # breakpoint()
    L2_error = np.sqrt(
        np.sum(
            np.square(pred_tracks - gt_tracks),
            axis=-1,
        )
    )
    masked_L2_error = L2_error * (1 - gt_occluded)
    avg_distance = np.sum(masked_L2_error) / np.sum(1 - gt_occluded)
    metrics["avg_distance"] = np.array([avg_distance])
    nonzero_masked_error = L2_error[(1 - gt_occluded).astype(bool)]
    assert np.allclose(masked_L2_error.sum(), nonzero_masked_error.sum()), (
        masked_L2_error.sum(),
        nonzero_masked_error.sum(),
        set((1 - gt_occluded).flatten().tolist()),
    )
    assert nonzero_masked_error.size == np.sum(1 - gt_occluded)
    assert np.allclose(avg_distance, nonzero_masked_error.mean()), (
        avg_distance,
        nonzero_masked_error.mean(),
        avg_distance - nonzero_masked_error.mean(),
        set((1 - gt_occluded).flatten().tolist()),
    )
    metrics["median_distance"] = np.array([np.median(nonzero_masked_error)])

    for thresh in [1, 2, 4, 8, 16]:
        # True positives are points that are within the threshold and where both
        # the prediction and the ground truth are listed as visible.
        within_dist = np.sum(
            np.square(pred_tracks - gt_tracks),
            axis=-1,
        ) < np.square(thresh)
        is_correct = np.logical_and(within_dist, visible)

        # Compute the frac_within_threshold, which is the fraction of points
        # within the threshold among points that are visible in the ground truth,
        # ignoring whether they're predicted to be visible.
        count_correct = np.sum(
            is_correct & evaluation_points,
            axis=(1, 2),
        )
        count_visible_points = np.sum(visible & evaluation_points, axis=(1, 2))
        frac_correct = count_correct / count_visible_points
        metrics["pts_within_" + str(thresh)] = frac_correct

        metrics["num_visible"] = count_visible_points
        metrics["num_pts_within_" + str(thresh)] = count_correct

        all_frac_within.append(frac_correct)

        true_positives = np.sum(is_correct & pred_visible & evaluation_points, axis=(1, 2))

        # The denominator of the jaccard metric is the true positives plus
        # false positives plus false negatives.  However, note that true positives
        # plus false negatives is simply the number of points in the ground truth
        # which is easier to compute than trying to compute all three quantities.
        # Thus we just add the number of points in the ground truth to the number
        # of false positives.
        #
        # False positives are simply points that are predicted to be visible,
        # but the ground truth is not visible or too far from the prediction.
        gt_positives = np.sum(visible & evaluation_points, axis=(1, 2))
        false_positives = (~visible) & pred_visible
        false_positives = false_positives | ((~within_dist) & pred_visible)
        false_positives = np.sum(false_positives & evaluation_points, axis=(1, 2))
        jaccard = true_positives / (gt_positives + false_positives)
        metrics["jaccard_" + str(thresh)] = jaccard
        all_jaccard.append(jaccard)
    metrics["average_jaccard"] = np.mean(
        np.stack(all_jaccard, axis=1),
        axis=1,
    )
    metrics["average_pts_within_thresh"] = np.mean(
        np.stack(all_frac_within, axis=1),
        axis=1,
    )
    return metrics


class AsyncTAPVidEvaluator:
    def __init__(self, dataset: tapvid_dataset.FlatTAPVidDataset, debug=False, cache_dir=None):
        """
        Runs TAP-Vid Evaluations on separate thread.

        Arguments:
          dataset: The TAP-Vid Dataset being used.
          debug: If set, runs on the main thread instead.
          cache_dir: If set, saves cache file during evaluation.
        """

        self.task_queue = Queue()
        self.worker_status = None
        self.dataset = dataset
        self.debug = debug
        self.cache_dir = cache_dir

        self.videos = {i: e for i, e in enumerate(dataset.vid_specs)}
        self.collection = dict()
        self.final_tapvid_metrics = defaultdict(list)

        if not self.debug:
            self.worker = Thread(target=self.collect)
            self.worker.daemon = True
            self.worker.start()

    def collect(self):
        """Collect metrics frame pair by frame pair, and evaluate when a video is complete."""

        try:
            while True:
                results_dict = self.task_queue.get()
                if results_dict is None:
                    break

                uids = results_dict.pop("uids")
                pred_occ = results_dict.pop("occlusion")

                assert results_dict.keys() == {"expec", "argmax", "multi_scale"}

                for pred_key, pred_val in results_dict.items():
                    collection = self.collection.setdefault(pred_key, {})

                    for i, uid in enumerate(uids):
                        vi, pi, st, et = uid

                        vid_meta = self.videos[vi]
                        vidname = vid_meta.vidname

                        collection.setdefault(vidname, {})
                        if (pi, st, et) in collection[vidname]:
                            continue

                        collection[vidname][(pi, st, et)] = (pred_val[i], pred_occ[i])

                        if len(collection[vidname]) == vid_meta.num_expected_evals:
                            query_points, gt_occluded, gt_tracks, pred_occluded, pred_tracks = (
                                _format_model_preds_for_tapvid(
                                    vid_meta, collection[vidname], "first" if self.dataset.frame_delta < 0 else "cfg"
                                )
                            )

                            if self.cache_dir:
                                individual_preds_dir = os.path.join(self.cache_dir, "preds_per_video", vid_meta.vidname)
                                os.makedirs(individual_preds_dir, exist_ok=True)

                                np.savez_compressed(
                                    os.path.join(individual_preds_dir, f"{pred_key}.npz"),
                                    query_points=query_points,
                                    gt_occluded=gt_occluded,
                                    gt_tracks=gt_tracks,
                                    pred_occluded=pred_occluded,
                                    pred_tracks=pred_tracks,
                                )

                            metrics = compute_tapvid_metrics(
                                query_points, gt_occluded, gt_tracks, pred_occluded, pred_tracks, "first"
                            )

                            for m in metrics:
                                self.final_tapvid_metrics[f"{pred_key}_{m}"].extend(metrics[m])

                            self.final_tapvid_metrics[f"{pred_key}_video_name"].append(vid_meta.vidname)

                if self.cache_dir:
                    with open(os.path.join(self.cache_dir, f"tapvid_eval_cache.pkl"), "wb") as f:
                        pickle.dump(self.collection, f)

                if self.debug:
                    break

        except Exception as e:
            self.worker_status = e
            return

    def submit_eval_job(self, results_dict):
        """Lauch async eval job. If debug is set, main thread will run it and this will be blocking."""
        if self.worker_status is not None:
            raise self.worker_status

        self.task_queue.put(results_dict, block=False)
        if self.debug:
            self.collect()

    def final_result(self):
        """Get final result. This will terminate the running thread."""
        if not self.debug:
            self.task_queue.put(None, block=False)
            self.worker.join()

        reporting = [
            "average_jaccard",
            "avg_distance",
            "median_distance",
            "average_pts_within_thresh",
            "occlusion_accuracy",
        ]

        report = defaultdict(dict)

        for key in sorted(self.final_tapvid_metrics):
            for r in reporting:
                if r in key:
                    typ = key[: key.find(r) - 1]
                    report[typ][r] = np.array(self.final_tapvid_metrics[key]).mean().item()

        # compute occ_f1 separately
        for typ in report:
            occ_tp = np.sum(self.final_tapvid_metrics[f"{typ}_occ_tp"])
            occ_fp = np.sum(self.final_tapvid_metrics[f"{typ}_occ_fp"])
            occ_fn = np.sum(self.final_tapvid_metrics[f"{typ}_occ_fn"])
            precision = occ_tp / (occ_tp + occ_fp)
            recall = occ_tp / (occ_tp + occ_fn)
            occ_f1 = 2 * ((precision * recall) / (precision + recall))
            self.final_tapvid_metrics[f"{typ}_occlusion_f1"] = occ_f1
            report[typ]["occlusion_f1"] = occ_f1.item()

        print("\n\n\nReport:")
        pprint(report, width=1)
        return self.final_tapvid_metrics
