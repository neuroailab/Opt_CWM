import torch
from torch import nn
from torch.distributions.multivariate_normal import MultivariateNormal

from utils import dist_logging, utils

logger = dist_logging.get_logger(__name__)


class PertMLP(nn.Module):
    def __init__(
        self,
        output_dim,
        input_dim,
        hidden_dim,
    ):
        """
        Generates perturbation parameters.

        Arguments:
          output_dim: MLP output dimension.
          input_dim: MLP input dimension.
          hidden_dim: MLP hidden dimension.
        """
        super().__init__()

        # Two-hidden-layer MLP
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Tanh(),
        )

    def forward(self, f0_emb: torch.Tensor):
        """
        Forward pass.

        Arguments:
          f0_emb: Frame0 patch embeddings extracted from base_cwm.

        Returns:
          (torch.Tensor): Perturbation parameters.
        """
        return self.mlp(f0_emb)


class GaussPerturber(nn.Module):

    params_per_channel = 6

    def __init__(self, pert_size, input_dim, hidden_dim):
        """
        Generates and applies Gaussian perturbations. Uses MLP to generate
        perturbation parameters, which are then transformed into Gaussian
        parameters that create smooth Gaussian perturbations.

        Arguments:
          pert_size: Size of applied perturbation.
          input_dim: MLP input dimension.
          hidden_dim: MLP hidden dimension.
          pert_scale: Dynamically scale perturbation by HW. This is useful
            for applying higher resolution perturbations at inference time.
        """
        super().__init__()

        self.pert_size = pert_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = self.params_per_channel * 3

        self.nn = PertMLP(self.output_dim, self.input_dim, self.hidden_dim)

        self._pert_scale = 1

    def _pert_params_to_gauss_params(self, pert_params):
        max_mean = max_std = self.pert_size / 2
        min_std = 1.0  # some min required otherwise mvn pdf causes nans
        max_amplitude_magnitude = 1.0
        max_corr_magnitude = 0.98  # To prevent rounding errors at corr=1
        # making correlation matrix not positive definite

        B = pert_params.size(0)
        pert_params = pert_params.reshape(B, 3, self.params_per_channel)

        gauss_params = torch.zeros_like(pert_params).cuda()
        gauss_params[:, :, 0:2] += pert_params[:, :, 0:2] * max_mean  # mean
        gauss_params[:, :, 2:4] += (pert_params[:, :, 2:4] + 1) * (  # stdevs
            max_std - min_std
        ) / 2 + min_std  # range (min_std, max_std)
        gauss_params[:, :, 4] += pert_params[:, :, 4] * max_corr_magnitude  # Correlation
        gauss_params[:, :, 5] += pert_params[:, :, 5] * max_amplitude_magnitude  # Amplitude

        return gauss_params

    def _gauss_params_to_perturbation(self, gauss_params):
        h = w = self.pert_size
        B = gauss_params.size(0)
        # TODO: Check edge cases/ off by one for the linspace used in gen_grid
        # (I changed from (0, h-1, h) to (-h/2, h/2, h) to have (0,0) in the middle)
        rowcol_grid = utils.gen_grid(-h / 2, h / 2, -w / 2, w / 2, h, w).cuda()  # shape (hw, 2(row,col))

        # Perturbation: (B, C, H, W)
        perturbation = torch.zeros(B, 3, self.pert_size, self.pert_size).cuda()
        for b in range(B):
            for c in range(3):
                mean = gauss_params[b, c, :2]
                row_std, col_std = gauss_params[b, c, 2:4]
                corr = gauss_params[b, c, 4]
                minn = 0
                maxx = gauss_params[b, c, 5]

                cov_rowcol = row_std * col_std * corr
                cov = torch.zeros(2, 2).cuda()
                cov[0, 0] += row_std**2
                cov[0, 1] += cov_rowcol
                cov[1, 0] += cov_rowcol
                cov[1, 1] += col_std**2

                mvn = MultivariateNormal(loc=mean, covariance_matrix=cov)
                batch_chan_pert = torch.exp(mvn.log_prob(rowcol_grid))

                scale = (maxx - minn) / batch_chan_pert.max()
                batch_chan_pert = scale * batch_chan_pert + minn

                perturbation[b, c] = batch_chan_pert.reshape(h, w)

        return perturbation

    def _apply_perturbation(self, video, f0_pixel_loc, perturbation):

        pert_h, pert_w = perturbation.shape[-2:]

        if self._pert_scale != 1:
            pert_h, pert_w = int(pert_h * self._pert_scale), int(pert_w * self._pert_scale)
            perturbation = torch.nn.functional.interpolate(perturbation, (pert_h, pert_w))

        pert_size = torch.LongTensor([[pert_h, pert_w]]).to(f0_pixel_loc.device)  # (1, 2)

        B, _, _, H, W = video.shape

        video = nn.functional.pad(video, [pert_w // 2, pert_w // 2, pert_h // 2, pert_h // 2])

        for b in range(B):
            mins, maxs = f0_pixel_loc, f0_pixel_loc + pert_size
            video[b, :, 0, mins[b, 0] : maxs[b, 0], mins[b, 1] : maxs[b, 1]] += perturbation[b]

        video = video[:, :, :, pert_h // 2 : -pert_h // 2, pert_w // 2 : -pert_w // 2]

        return video

    def forward(self, video: torch.Tensor, f0_pixel_loc: torch.Tensor, encoder_out: torch.Tensor, return_pert=False):
        """
        Generates and applies Gaussian perturbations.

        Arguments:
          video: The video to perturb, shape B,C=3,T=2,H,W.
            Perturbation will be applied to frame0 only.
          f0_pixel_loc: The target point (yx) location to apply pert.
          encoder_out: Output of base_cwm encoder to generate parameters.
          return_pert: If true, returns generated perturbation.

        Returns:
          (torch.Tensor): Perturbed video.
          (torch.Tensor): If return_pert, the perturbation is returned.
        """
        pert_params = self.nn(encoder_out)
        gauss_params = self._pert_params_to_gauss_params(pert_params)
        perturbation = self._gauss_params_to_perturbation(gauss_params)

        perturbed_video = self._apply_perturbation(video, f0_pixel_loc, perturbation)

        if return_pert:
            return perturbed_video, perturbation
        return perturbed_video

    def highres(self):
        if self._pert_scale == 2:
            return

        self._pert_scale = 2
        logger.info(
            f"Running perturber in high-res mode (2x) with perturbation size: {self.pert_size * self._pert_scale}x{self.pert_size * self._pert_scale}"
        )
