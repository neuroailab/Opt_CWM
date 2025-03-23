from dataclasses import dataclass
from typing import List

import torch


@dataclass
class _SingleMaskFlowPredictorOutput:
    expec_pred_loc: torch.FloatTensor
    argmax_pred_loc: torch.FloatTensor
    recon_clean: torch.FloatTensor
    recon_perturbed: torch.FloatTensor
    recon_delta: torch.FloatTensor
    reduced_recon_delta: torch.FloatTensor
    std_dev: torch.FloatTensor
    prob_heatmap: torch.FloatTensor
    pert: torch.FloatTensor
    input_perturbed: torch.FloatTensor


@dataclass
class _MultiMaskFlowPredictorOutput:
    mask_iterations: List[_SingleMaskFlowPredictorOutput]
    final_expec_pred_loc: torch.FloatTensor
    final_argmax_pred_loc: torch.FloatTensor
    final_recon_delta: torch.FloatTensor
    final_reduced_recon_delta: torch.FloatTensor
    final_std_dev: torch.FloatTensor
    final_prob_heatmap: torch.FloatTensor
    final_pred_occ: torch.FloatTensor


@dataclass
class _MultiScaleFlowPredictorOutput:
    zoom_iterations: List[_MultiMaskFlowPredictorOutput]
    final_multi_scale_pred_loc: torch.FloatTensor
    final_pred_occ: torch.FloatTensor


@dataclass
class FlowPredictorOutput:
    final_pred_occ: torch.FloatTensor
    final_expec_pred_loc: torch.FloatTensor
    final_argmax_pred_loc: torch.FloatTensor
    final_multi_scale_pred_loc: torch.FloatTensor
    history: _MultiScaleFlowPredictorOutput
