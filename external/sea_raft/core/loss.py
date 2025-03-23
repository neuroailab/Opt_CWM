import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# exclude extremly large displacements
MAX_FLOW = 400
SUM_FREQ = 100
VAL_FREQ = 5000


def sequence_loss(output, flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW):
    """Loss function defined over sequence of flow predictions"""
    n_predictions = len(output["flow"])
    flow_loss = 0.0
    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)
    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)
        loss_i = output["nf"][i]
        final_mask = (~torch.isnan(loss_i.detach())) & (~torch.isinf(loss_i.detach())) & valid[:, None]
        flow_loss += i_weight * ((final_mask * loss_i).sum() / final_mask.sum())

    return flow_loss


# Indexing is broken here
# def avg_pixel_dist(flow, flow_gt, valid, max_flow=MAX_FLOW):
#     """Pixel distance, averaged over batch"""
#     # exlude invalid pixels and extremely large diplacements
#     B,C,H,W = flow.size()
#     assert flow.size() == flow_gt.size() and C == 2 and H == W == 256,(flow.shape, flow_gt.shape)
#     assert valid.size() == (B, H, W),(valid.shape)
#     mag = torch.sum(flow_gt**2, dim=1).sqrt()
#     # print("Total valid (orig):", torch.sum(valid, dim=(1,2)))
#     valid = (valid >= 0.5) & (mag < max_flow)
#     # print("Total valid (after):", torch.sum(valid, dim=(1,2)))
#     flow_errors = torch.sum((flow - flow_gt) ** 2, dim=1).sqrt()
#     # print("Shapes:", final_flow.shape, flow_gt.shape, valid.shape, flow_errors.shape)
#     valid_flow_errors = flow_errors[valid]
#     # print("Valid flow errors shape:", valid_flow_errors.shape)
#     return valid_flow_errors.mean()


def avg_pixel_dist(flow, flow_gt, valid):
    """Pixel distance, averaged over batch"""
    # print("Sizes of pixel dist inputs:", flow.size(), flow_gt.size(), valid.size())
    B, C, H, W = flow.size()
    assert flow.size() == flow_gt.size() and C == 2 and H == W == 256, (flow.shape, flow_gt.shape)
    assert valid.size() == (B, H, W), valid.shape
    valid = valid.bool()

    flow_errors = []
    flow_means = []
    for b in range(B):
        # Can make this indexing more explicit? // Now about as explicit as indexing can get!
        flow_b = flow[b]
        flow_gt_b = flow_gt[b]
        flow_bx, flow_by = flow_b.unbind(0)
        flow_gt_bx, flow_gt_by = flow_gt_b.unbind(0)
        valid_b = valid[b]
        row_indices, col_indices = valid_b.nonzero().unbind(1)
        assert row_indices.size() == col_indices.size() == (51,)
        assert (
            flow_bx.size() == flow_by.size() == flow_gt_bx.size() == flow_gt_by.size() == valid_b.size() == (256, 256)
        )
        valid_flow_x = flow_bx[valid_b]
        valid_flow_y = flow_by[valid_b]
        valid_flow_gtx = flow_gt_bx[valid_b]
        valid_flow_gty = flow_gt_by[valid_b]
        # assert valid_flow.size() == valid_flow_gt.size() == (2,51)
        assert valid_flow_x.size() == valid_flow_y.size() == valid_flow_gtx.size() == valid_flow_gty.size() == (51,)
        # print(valid_flow.size(), valid_flow_gt.size())

        # Original version of this code
        flow_error_orig = (
            torch.sum((flow[b, :, valid[b]] - flow_gt[b, :, valid[b]]) ** 2, dim=0).sqrt().mean().detach().item()
        )

        # New version (wow exact same line length!)
        flow_error = (
            torch.sqrt((valid_flow_x - valid_flow_gtx) ** 2 + (valid_flow_y - valid_flow_gty) ** 2)
            .mean()
            .detach()
            .item()
        )

        # Even newer version
        flow_error_new = (
            torch.sqrt(
                (flow_bx[row_indices, col_indices] - flow_gt_bx[row_indices, col_indices]) ** 2
                + (flow_by[row_indices, col_indices] - flow_gt_by[row_indices, col_indices]) ** 2
            )
            .mean()
            .detach()
            .item()
        )

        assert np.allclose(flow_error, flow_error_orig), (flow_error, flow_error_orig)
        assert np.allclose(flow_error, flow_error_new), (flow_error, flow_error_new)
        # print("Errors are the same regardless of indexing explicitness?", flow_error == flow_error_orig == flow_error_new)
        # print(b, flow_error)
        # print(row_indices)
        # print(col_indices)

        # flow_error_grid = ((flow_b - flow_gt_b)**2).sum(0).sqrt()
        # for row,col in valid_b.nonzero():
        #     print('\n',col.item(), [(i,val.item()) for i,val in enumerate(flow_error_grid[row])])
        #     print([(i,val.item()) for i,val in enumerate(flow_b[0,row])])
        #     print([(i,val.item()) for i,val in enumerate(flow_gt_b[0,row])])
        #     break
        # print((flow_b**2).sum(0).sqrt().mean(), flow_b.abs().sum(0).mean(), flow_error_grid.mean(), flow_error_grid[row_indices, col_indices].mean())
        flow_errors.append(flow_error)
        flow_means.append((flow_b**2).sum(0).sqrt().mean().item())
        # assert False

    mean_flow_error = np.mean(flow_errors)
    # raise Exception(str(mean_flow_error))
    return mean_flow_error, flow_errors, flow_means


def l1_distance(tensor1, tensor2):
    return (tensor1 - tensor2).abs().sum()


def avg_l1_distance(tensor1, tensor2, sum_dim):
    return (tensor1 - tensor2).abs().sum(sum_dim).mean().detach().item()
