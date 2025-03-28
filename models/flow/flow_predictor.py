import os

import requests
import torch
import tqdm
from torch import nn

from huggingface_hub import hf_hub_download

from models.base_cwm import masking_generator, modeling_pretrain
from models.flow import occ_predictor, perturber, soft_argmax
from utils import constants, dist_logging, utils

logger = dist_logging.get_logger(__name__)


class FlowPredictor(nn.Module):
    def __init__(
        self,
        cwm_model: modeling_pretrain.PretrainVisionTransformer,
        perturber: perturber.GaussPerturber,
        softargmax_module: soft_argmax.SoftArgmax,
        occ_module: occ_predictor.OccPredictor,
        masking_iters: int,
        masking_ratio: float,
        zoom_iters: int,
    ):
        """
        Predicts flow and occlusion of points across two frames.

        Arguments:
          cwm_model: The base_cwm next frame predictor with signature RGB -> RGB.
          perturber: The gaussian perturber that generates and apply perturbations.
          softargmax_module: The soft argmax module that computes differentiable argmax.
          occ_module: The occlusion predictor that predicts occlusion from delta threshold.
          msaking_iters: Number of multi-mask instances.
          zoom_iters: Number of multi-scale instances.
        """
        super().__init__()

        self.perturber = perturber
        self.cwm_model = cwm_model
        self.softargmax_module = softargmax_module
        self.occ_module = occ_module

        assert (
            self.cwm_model.patch_size[-1] == self.cwm_model.patch_size[-2]
        ), f"base_cwm patch size is not square: {self.cwm_model.patch_size}"

        assert self.cwm_model.patch_size[0] == 1, f"base_cwm has > 1 tubelet size: {self.cwm_model.patch_size[0]}"

        assert masking_iters > 0, f"masking_iters must be > 0"

        # self.input_size = self.cwm_model.input_size
        # self.patch_size = self.cwm_model.patch_size[-1]
        # self.n_patches = self.cwm_model.n_patches

        self.masking_ratio = masking_ratio
        self.masking_iters = masking_iters
        self.masking_generator = masking_generator.MultiMaskingGenerator(
            (2, *self.n_patches), masking_ratio, masking_iters
        )
        self.zoom_iters = zoom_iters

    @property
    def input_size(self):
        return self.cwm_model.input_size

    @property
    def patch_size(self):
        return self.cwm_model.patch_size[-1]

    @property
    def n_patches(self):
        return self.cwm_model.n_patches

    def _preproc_video(self, video, pixel_loc):
        # Moved from flow_cwm
        B, C, T, H, W = video.shape

        if (H, W) == self.input_size:
            new_video = video
            new_pixel_loc = pixel_loc
            return new_video, new_pixel_loc

        new_video = utils.batch_resize_video(video, self.input_size)
        utils.size_guard(new_video, (B, C, T, *self.input_size))
        new_pixel_loc = utils.rescale_points(pixel_loc, (H, W), self.input_size)
        return new_video, new_pixel_loc

    def _postproc_video(self, video, og_size):
        # remove padding
        B, C, T, H, W = video.size()

        if (H, W) == og_size:
            return video

        resized = utils.batch_resize_video(video, og_size)
        return resized

    def _get_heuristic_mask(self, mask, pixel_loc1):
        pixel_loc1_patch = (pixel_loc1 // self.patch_size).long()

        new_mask = mask.clone()
        num_patches_per_frame = mask.size(-1) // 2
        for b in range(mask.size(0)):
            patch_y, patch_x = pixel_loc1_patch[b]
            unmasked = []
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    if (
                        patch_y + dy < 0
                        or patch_y + dy >= self.n_patches[0]
                        or patch_x + dx < 0
                        or patch_x + dx >= self.n_patches[1]
                    ):
                        continue
                    masked = mask[b, num_patches_per_frame + (patch_y + dy) * self.n_patches[1] + (patch_x + dx)]
                    if not masked:
                        unmasked.append((patch_y + dy, patch_x + dx))

            new_mask[b][torch.nonzero(new_mask[b])[: len(unmasked)]] = 0
            for um in unmasked:
                new_mask[b, num_patches_per_frame + um[0] * self.n_patches[1] + um[1]] = 1

        return new_mask

    def _single_mask_forward(self, video, mask, pixel_loc0, heuristic_masking_loc=None):
        B, C, T, H, W = video.size()

        pixel_loc = pixel_loc0

        # If heuristic_masking_loc is provided, convert to model input space
        # then create a patch mask that includes (covers) that location.
        # We can use previous prediction to construct a more effective mask.
        if heuristic_masking_loc is not None:
            pixel_loc = torch.stack([pixel_loc0, heuristic_masking_loc], dim=1)

        preproc_video, pixel_loc = self._preproc_video(video, pixel_loc)

        pixel_loc0 = pixel_loc
        if heuristic_masking_loc is not None:
            pixel_loc0, heuristic_masking_loc = torch.unbind(pixel_loc, dim=1)
            mask = self._get_heuristic_mask(mask, heuristic_masking_loc)

        video_pert = preproc_video.clone()

        pixel_row = pixel_loc0[:, 0]
        pixel_col = pixel_loc0[:, 1]
        pred_clean, encoder_out = self.cwm_model.get_counterfactual(preproc_video, mask, get_encoder_out=True)

        # reconstructing frame1 without any perturbation
        encoder_out = encoder_out.detach()
        pred_clean = pred_clean.detach()
        recon_clean = self.cwm_model.unpatchify(pred_clean, mask)

        recon_clean = self._postproc_video(recon_clean, (H, W))

        # getting patch embeddings
        f0_emb = encoder_out[:, : self.n_patches[0] * self.n_patches[1]]
        _, N, C_e = f0_emb.size()
        f0_emb = torch.reshape(f0_emb, (B, self.n_patches[0], self.n_patches[1], C_e))
        patch_embedding = torch.stack(
            [
                f0_emb[
                    b,
                    int(pixel_row[b] / self.patch_size),
                    int(pixel_col[b] / self.patch_size),
                ]
                for b in range(B)
            ]
        )

        # applying perturbation
        video_pert, pert = self.perturber(video_pert, pixel_loc0, patch_embedding, return_pert=True)
        utils.size_guard(video_pert, preproc_video.size())

        pred_pert = self.cwm_model.get_counterfactual(video_pert, mask)
        recon_pert = self.cwm_model.unpatchify(pred_pert, mask)
        recon_pert = self._postproc_video(recon_pert, (H, W))
        utils.size_guard(recon_pert, (B, C, 1, H, W))

        recon_delta = (recon_pert - recon_clean)[:, :, 0]
        utils.size_guard(recon_delta, (B, C, H, W))  # C, H, W

        pred_loc, std_dev, prob_heatmap, reduced_recon_delta = self.softargmax_module(recon_delta)

        return {"expec": pred_loc, "argmax": utils.compute_2d_argmax(prob_heatmap), "recon_delta": recon_delta}

    def _multi_mask_forward(self, video, pixel_loc0, heuristic_masking_loc=None):
        B, C, T, H, W = video.size()

        masks = self.masking_generator(B).bool().cuda()
        recon_delta = torch.zeros((B, C, H, W)).to(video.device)

        all_recon_deltas = []

        for i in range(masks.size(1)):
            mask = masks[:, i]
            out = self._single_mask_forward(video, mask, pixel_loc0, heuristic_masking_loc=heuristic_masking_loc)

            recon_delta += out["recon_delta"]
            all_recon_deltas.append(out["recon_delta"])

        recon_delta /= masks.size(1)
        pred_loc, std_dev, prob_heatmap, reduced_recon_delta = self.softargmax_module(recon_delta)

        return {
            "expec": pred_loc,
            "argmax": utils.compute_2d_argmax(prob_heatmap),
            "occ": self.occ_module(all_recon_deltas),
        }

    def _multi_scale_forward(self, video, pixel_loc0, pred_pixel_loc1):

        crop_size = min(video.shape[-2:])

        pt0 = pixel_loc0.clone()
        pt1 = pred_pixel_loc1.clone()
        ofs = torch.zeros_like(pt0).unsqueeze(-1)

        for z in range(self.zoom_iters):
            B, C, T, H, W = video.size()
            hw = torch.Tensor([[[H, W]]]).to(video.device)
            pts = torch.stack([pt0, pt1], 1)  # (B, 2), 2

            # first, shift upper left corner and redefine bottom right
            # if no shift is necessary, bottom right is the same
            mins = (torch.maximum(pts - crop_size // 2, torch.zeros_like(pts).to(pts.device))).long()
            maxs = mins + crop_size

            # next, shift bottom right corner and redefine upper left
            # if no shift is necessary, upper left is the same
            maxs = (torch.minimum(maxs, hw)).long()
            mins = maxs - crop_size

            cropped = []
            for b in range(video.size(0)):
                f0 = video[b, :, 0, mins[b, 0, 0] : maxs[b, 0, 0], mins[b, 0, 1] : maxs[b, 0, 1]]
                f1 = video[b, :, 1, mins[b, 1, 0] : maxs[b, 1, 0], mins[b, 1, 1] : maxs[b, 1, 1]]
                cropped.append(torch.stack([f0, f1], 1))

            video = torch.stack(cropped, 0)
            utils.size_guard(video, (B, C, T, crop_size, crop_size))

            pt0 = pt0 - mins[:, 0]
            pt1 = pt1 - mins[:, 1]  # B, 2
            ofs = ofs + mins

            # for multi-scale, we perform heuristic masking
            out = self._multi_mask_forward(video, pt0, heuristic_masking_loc=pt1)

            pt1 = out["argmax"]

            crop_size = int(crop_size * 0.75)

        final_pt0 = pt0 + ofs[:, 0]
        assert torch.allclose(final_pt0, pixel_loc0), f"Tracking pixel location different after zoom in/out!"

        final_pt1 = pt1 + ofs[:, 1]

        return {"multi_scale": final_pt1, "occ": out["occ"]}

    def forward(self, video: torch.Tensor, pixel_loc0: torch.Tensor):
        """
        Predicts flow and occlusion.

        Arguments:
          video: The video to predict flow, shape B,C=3,T=2,H,W.
          pixel_loc0: The target point (yx) location in frame 0, shape B,D=2

        Returns:
          (dict(str, any)): Model prediction result containing the following keys:
            expec_pred_pixel_loc: Predicted target location, using expectation.
            argmax_pred_pixel_loc: Predicted target location, using argmax.
            multi_scale_pred_pixel_loc: Predicted target location, using multiscale.
              Note that when zoom_iters == 0, this is equivalent to argmax.
            pred_pixel_loc: Final predicted target location. Equivalent to multiscale.
            pred_occ: Final predicted occlusion.
        """

        model_out = self._multi_mask_forward(video, pixel_loc0)

        if self.zoom_iters > 0:
            ms_model_out = self._multi_scale_forward(video, pixel_loc0, model_out["argmax"])
            model_out.update(ms_model_out)
        else:
            model_out["multi_scale"] = model_out["argmax"]

        return {
            "expec_pred_pixel_loc": model_out["expec"],
            "argmax_pred_pixel_loc": model_out["argmax"],
            "multi_scale_pred_pixel_loc": model_out["multi_scale"],
            "pred_pixel_loc": model_out["multi_scale"],
            "pred_occ": model_out["occ"],
        }

    def highres(self):
        self.cwm_model.highres()
        self.perturber.highres()

        self.masking_generator = masking_generator.MultiMaskingGenerator(
            (2, *self.n_patches), self.masking_ratio, self.masking_iters
        )

    def load_pretrained(self, highres=False, force=False):
        if highres:
            self.highres()

        self.cwm_model.load_pretrained(highres, force)

        local_dir = os.path.join(constants.MODEL_LOCAL_CACHE_PATH, "opt_cwm")
        os.makedirs(local_dir, exist_ok=True)

        gcloud_dir = os.path.join(constants.MODEL_GCLOUD_BUCKET_PATH, "opt_cwm")

        opt_cwm_local_path = os.path.join(local_dir, "opt_cwm_ckpt.pt")
        opt_cwm_gcloud_path = os.path.join(gcloud_dir, "opt_cwm_ckpt.pt")

        if force or not os.path.exists(opt_cwm_local_path):
            logger.info(f"Saving opt_cwm model to: {opt_cwm_local_path}.")
            utils.download_from_gcloud(opt_cwm_gcloud_path, opt_cwm_local_path)

        logger.info("Extracting flow_predictor checkpoint from opt_cwm.")
        ckpt = torch.load(opt_cwm_local_path, map_location="cpu")["model"]
        ckpt = {k.replace("flow_predictor.", ""): v for k, v in ckpt.items() if "flow_predictor" in k}

        self.load_state_dict(ckpt, strict=False)

        logger.info("Succesfully loaded checkpoint for flow_predictor.")

        return self
    
    def from_pretrained(self, repo_id: str):
        """
        Loads pretrained weights from the specified Hugging Face repository.

        Args:
            repo_id (str): The Hugging Face repository ID containing the pretrained weights

        Returns:
            FlowPredictor: The current instance with loaded pretrained weights
        """
        # Load the pretrained cwm model
        self.cwm_model.from_pretrained(repo_id, filename="cwm_model.pt")

        # Load the pretrained weights
        filepath = hf_hub_download(repo_id=repo_id, filename="flow_predictor.pt")
        ckpt = torch.load(filepath, map_location="cpu")
        if "model" in ckpt:
            ckpt = ckpt["model"]
        ckpt = {k.replace("flow_predictor.", ""): v for k, v in ckpt.items() if "flow_predictor" in k}
        self.load_state_dict(ckpt, strict=False)
        
        logger.info(f"Successfully loaded pretrained weights from {repo_id}")
        return self