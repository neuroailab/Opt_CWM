import os
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.layers import trunc_normal_ as __call_trunc_normal_
from torch import Tensor

from models.base_cwm.model_utils import Block, PatchEmbed, get_sinusoid_encoding_table
from utils import constants, dist_logging, utils

logger = dist_logging.get_logger(__name__)


def trunc_normal_(tensor, mean=0.0, std=1.0):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


def interpolate_pos_encoding(pos_embed, n_frames, h, w):
    N = pos_embed.shape[1]
    if N == (h * w * n_frames):
        return pos_embed
    old_h = old_w = int((N / n_frames) ** 0.5)
    patch_pos_embed = pos_embed.view(1, n_frames, old_h, old_w, -1).flatten(0, 1).permute(0, 3, 1, 2)

    patch_pos_embed = F.interpolate(
        patch_pos_embed,
        size=(h, w),
        mode="bicubic",
    )
    return patch_pos_embed.permute(0, 2, 3, 1).flatten(0, 2).unsqueeze(0)


class PretrainVisionTransformerEncoder(nn.Module):
    """Vision Transformer with support for patch or hybrid CNN input stage"""

    def __init__(
        self,
        img_size=224,
        patch_size=(16, 16),
        in_chans=3,
        num_classes=0,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_values=None,
        tubelet_size=2,
        use_learnable_pos_emb=False,
        num_frames=16,
        k_bias=False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_size = (tubelet_size,) + patch_size
        self.pt, self.ph, self.pw = self.patch_size
        self.h = int(img_size / self.ph)
        self.w = int(img_size / self.pw)

        self.hw = self.h * self.w

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            tubelet_size=tubelet_size,
            num_frames=num_frames,
        )
        self.num_patches = self.patch_embed.num_patches
        self.num_frames = num_frames

        logger.debug(f"NUM PATCHES IN ENCODER: {self.num_patches}")

        self.pos_embed = get_sinusoid_encoding_table(self.num_patches, embed_dim)

        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(self.pos_embed)

        self.learn_pos_embed = use_learnable_pos_emb

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    init_values=init_values,
                    k_bias=k_bias,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # NOTE: Remove this if we are looking for "consistency"
        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=0.02)

        self.apply(self._init_weights)

    def _set_pos_embed(self, dim=None):
        if dim is None:
            dim = self.embed_dim
        if self.pos_embed is None:
            self.pos_embed = get_sinusoid_encoding_table(self.num_patches, dim)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=""):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def _get_pos_embed(self):
        return self.pos_embed

    def forward_block(self, x, idx):
        return self.blocks[idx](x)

    def interpolate_tensor_with_mask_token(
        self, x: Tensor, mask: Tensor, mask_token: Tensor, invert: bool = True
    ) -> Tensor:
        """
        Where mask == (0 if invert else 1), return x
        where mask == (1 if invert else 0), return mask_token
        Linearly interpolate between these using value of mask.
        """
        # mask_token = mask_token
        # breakpoint()
        B, N, C = x.shape
        assert mask.shape[1] == N, (
            f"Number of tokens in mask ({mask.shape[1]}) does not match " f"number of tokens in input ({N})"
        )

        assert mask_token.shape[-1] == C, (
            f"Dimensionality of mask token ({mask_token.shape[-1]}) does not match "
            f"dimensionality of tokens in input ({C})"
        )

        # convert mask to interpolation weights in range [0., 1.]
        mask = mask.to(x).clip(min=0.0, max=1.0)
        mask = (1.0 - mask) if invert else mask
        mask = mask.unsqueeze(-1)  # [B, N, 1]

        # expand mask token
        mask_token = mask_token.view(1, 1, C).expand(B, N, -1)

        # interpolate
        start = mask_token
        end = x

        return start + mask * (end - start)

    def forward_features(self, x, mask, move_patches, static_patches, delta, mask_token, res=1, res_y=None):

        if res_y is None:
            res_y = res

        x = embed = self.patch_embed(x)

        if res != 1:
            T = 2
            p0 = self.patch_size[-2]
            p1 = self.patch_size[-1]
            pos_embed = interpolate_pos_encoding(self.pos_embed, T, int(256 // p0 * res), int(256 // p1 * res_y))
        else:

            pos_embed = self._get_pos_embed()

        pos_embed = pos_embed.type_as(x)  # .to(x.device).clone()

        if not self.learn_pos_embed:
            pos_embed = pos_embed.to(x.device).clone().detach()

        x = x + pos_embed
        B, _, C = x.shape
        # x_vis = x[~mask].reshape(B, -1, C) # ~mask means visible

        x_vis = self.interpolate_tensor_with_mask_token(x, mask, mask_token, invert=True)

        if move_patches is not None:

            assert B == 1, "Only support batch size 1 for now"
            for px, py in move_patches:
                idx = px * self.w + py
                dx, dy = delta
                nx, ny = px + dx, py + dy
                new_idx = nx * self.w + ny + (self.patch_embed.num_frames - 1) * (self.h * self.w)

                emb = embed[:, idx]
                pos_emb = pos_embed[:, new_idx]
                emb = emb + pos_emb
                x_vis = torch.cat([x_vis, emb[None]], 1)

            if static_patches is not None:
                for px, py in static_patches:
                    idx = px * self.w + py
                    new_idx = px * self.w + py + (self.patch_embed.num_frames - 1) * (self.h * self.w)
                    emb = embed[:, idx]
                    pos_emb = pos_embed[:, new_idx]
                    emb = emb + pos_emb
                    x_vis = torch.cat([x_vis, emb[None]], 1)

        for blk in self.blocks:
            x_vis = blk(x_vis)

        x_vis = self.norm(x_vis)
        return x_vis

    def _set_inputs(self, *args, **kwargs):
        pass

    def forward(
        self,
        x,
        mask,
        mask_token,
        move_patches=None,
        static_patches=None,
        delta=None,
        res=1,
        res_y=None,
    ):

        if res_y is None:
            res = res

        self._set_inputs(x, mask)
        # pass input through the encoder
        x = self.forward_features(x, mask, move_patches, static_patches, delta, mask_token, res=res, res_y=res_y)

        # if we are passing through the entire encoder transformer we apply the head layer
        x = self.head(x)
        return x


class PretrainVisionTransformerDecoder(nn.Module):
    """Vision Transformer with support for patch or hybrid CNN input stage"""

    def __init__(
        self,
        patch_size=(16, 16),
        num_classes=768,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_values=None,
        k_bias=False,
    ):
        super().__init__()

        self.num_classes = num_classes

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_size = patch_size

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    init_values=init_values,
                    k_bias=k_bias,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=""):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_block(self, x, idx):
        return self.blocks[idx](x)

    def get_last_tokens(self, x, return_token_num):
        if return_token_num > 0:
            return self.head(self.norm(x[:, -return_token_num:]))
        elif return_token_num == 0:
            return self.head(self.norm(x))[:, x.size(1) :]
        else:
            return self.head(self.norm(x))

    def forward(self, x, return_token_num):

        # pass input through the decoder
        for blk in self.blocks:
            x = blk(x)

        if return_token_num > 0:
            x = self.head(self.norm(x[:, -return_token_num:]))  # only return the mask tokens predict pixels
        else:
            x = self.head(self.norm(x))

        return x


class PretrainVisionTransformer(nn.Module):
    """Vision Transformer with support for patch or hybrid CNN input stage"""

    def __init__(
        self,
        img_size=224,
        patch_size=(16, 16),
        encoder_in_chans=3,
        encoder_num_classes=0,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        decoder_num_classes=None,  # For pretraining this parameter isn't relevant but must be set according to tube&patch size
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=False,
        k_bias=False,
        qk_scale=None,
        num_frames=16,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_values=0.0,
        tubelet_size=2,
        use_learnable_pos_emb=False,
    ):
        super().__init__()

        self.learn_pos_embed = use_learnable_pos_emb

        self.encoder = PretrainVisionTransformerEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=encoder_in_chans,
            num_classes=encoder_num_classes,
            embed_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            tubelet_size=tubelet_size,
            num_frames=num_frames,
            k_bias=k_bias,
            use_learnable_pos_emb=use_learnable_pos_emb,
        )

        self.decoder = PretrainVisionTransformerDecoder(
            patch_size=patch_size,
            num_classes=(
                3 * tubelet_size * (patch_size[0] * patch_size[1])
                if decoder_num_classes is None
                else decoder_num_classes
            ),
            embed_dim=decoder_embed_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            k_bias=k_bias,
        )

        self.encoder_to_decoder = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=k_bias)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, encoder_embed_dim))
        trunc_normal_(self.mask_token, std=0.02)

        if self.learn_pos_embed:
            self.pos_embed = nn.Parameter(get_sinusoid_encoding_table(self.encoder.num_patches, decoder_embed_dim))
        else:
            self.pos_embed = get_sinusoid_encoding_table(self.encoder.num_patches, decoder_embed_dim)

        self.num_frames = num_frames
        self.num_patches = self.encoder.num_patches
        if self.num_frames is not None:
            self.num_patches_per_frame = self.num_patches // self.num_frames
        else:
            self.num_patches_per_frame = self.num_patches
        self.patch_size = self.encoder.patch_size
        if isinstance(img_size, int):
            self.image_size = (img_size, img_size)
        else:
            assert hasattr(img_size, "__len__"), img_size
            self.image_size = img_size

        # high res
        # may be different with train time config
        # self.pos_emb_scale = pos_emb_scale
        self._pos_emb_scale = 1
        # self.input_size = [size * scale for size, scale in zip(self.image_size, self.pos_emb_scale)]
        # self.n_patches = (self.input_size[0] // self.patch_size[-2], self.input_size[1] // self.patch_size[-1])

    @property
    def mask_size(self):
        return (
            self.num_frames // self.patch_size[0],
            self.image_size[-2] // self.patch_size[-2],
            self.image_size[-1] // self.patch_size[-1],
        )

    @property
    def input_size(self):
        return [size * self._pos_emb_scale for size in self.image_size]

    @property
    def n_patches(self):
        return [isize // psize for isize, psize in zip(self.input_size, self.patch_size[1:])]

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "mask_token"}

    def unpatchify(self, x, mask):
        # Because we modified things at the `get_counterfactual` level,
        # so that it returns the same thing as HardCWM,
        # the unpatchify function should be the same.

        # Define the input tensor
        B, N, C = x.shape  # batch size
        h, w = self.n_patches
        recon = torch.zeros(B, h * w, C).to(x)
        assert mask.sum(1)[0] == x.size(1), f"{mask.sum()}, {mask.size()}, {x.size()})"
        if mask[:, : h * w].sum() == 0:
            # Forward (as usual)
            recon[mask[:, -h * w :]] = x.flatten(0, 1)
        else:
            assert mask[:, -h * w :].sum() == 0
            recon[mask[:, : h * w]] = x.flatten(0, 1)

        rec_imgs = rearrange(recon, "b n (p c) -> b n p c", c=3)
        # Notice: To visualize the reconstruction video, we add the predict and the original mean and var of each patch.
        rec_imgs = rearrange(
            rec_imgs,
            "b (t h w) (p0 p1 p2) c -> b c (t p0) (h p1) (w p2)",
            p0=1,
            p1=self.patch_size[-2],
            p2=self.patch_size[-1],
            h=h,
            w=w,
        )
        return rec_imgs

    def forward(
        self,
        x,
        mask,
        move_patches=None,
        static_patches=None,
        delta=None,
        res=1,
        res_y=None,
        get_encoder_out=False,
    ):

        if res_y is None:
            res_y = res

        _, _, T, _, _ = x.shape

        self.device = x.device

        enc_out = self.encoder(
            x,
            mask,
            self.mask_token,
            move_patches=move_patches,
            static_patches=static_patches,
            delta=delta,
            res=res,
            res_y=res_y,
        )  # [B, N_vis, C_e]

        x_vis = self.encoder_to_decoder(enc_out)

        # add pos embedding
        if res != 1:
            p0 = self.patch_size[-2]
            p1 = self.patch_size[-1]
            pos_embed = interpolate_pos_encoding(self.pos_embed, T, int(256 // p0 * res), int(256 // p1 * res_y))
        else:
            pos_embed = self.pos_embed
        dec_pos_embed = pos_embed.expand(x_vis.size(0), -1, -1).type_as(x)

        if not self.learn_pos_embed:
            dec_pos_embed = dec_pos_embed.to(x.device).clone().detach()

        x_vis = x_vis + dec_pos_embed

        # pass input through the decoder, this will automatically return an intermediate layer if return_feat_layer is set
        x_all = self.decoder(x_vis, 0)

        if get_encoder_out:
            return x_all, enc_out

        return x_all

    def get_counterfactual(self, video, mask, get_encoder_out=False):
        res = res_y = self._pos_emb_scale
        x = self(video, mask=mask, get_encoder_out=get_encoder_out, res=res, res_y=res_y)

        if get_encoder_out:
            x, enc_out = x

        x = x[mask].reshape(x.size(0), torch.count_nonzero(mask[0]), -1)

        if get_encoder_out:
            return x, enc_out
        return x

    def highres(self):
        # high res
        # may be different with train time config
        # self.pos_emb_scale = (2, 2)
        # self.input_size = [size * scale for size, scale in zip(self.image_size, self.pos_emb_scale)]
        # self.n_patches = (self.input_size[0] // self.patch_size[-2], self.input_size[1] // self.patch_size[-1])
        if self._pos_emb_scale == 2:
            return

        self._pos_emb_scale = 2
        logger.info(f"Running base_cwm in high-res mode (2x) with input size {self.input_size[0]}x{self.input_size[1]}")

    def load_pretrained(self, highres=False, force=False):
        if highres:
            base_cwm_suffix = f"base_cwm_512_ckpt.pt"
            self.highres()
        else:
            base_cwm_suffix = f"base_cwm_256_ckpt.pt"

        local_dir = os.path.join(constants.MODEL_LOCAL_CACHE_PATH, "opt_cwm")
        os.makedirs(local_dir, exist_ok=True)

        gcloud_dir = os.path.join(constants.MODEL_GCLOUD_BUCKET_PATH, "opt_cwm")

        base_cwm_local_path = os.path.join(local_dir, base_cwm_suffix)
        base_cwm_gcloud_path = os.path.join(gcloud_dir, base_cwm_suffix)

        if force or not os.path.exists(base_cwm_local_path):
            logger.info(f"Saving base_cwm model to: {base_cwm_local_path}.")
            utils.download_from_gcloud(base_cwm_gcloud_path, base_cwm_local_path)

        ckpt = torch.load(base_cwm_local_path, map_location="cpu")
        self.load_state_dict(ckpt["model"])

        logger.info("Succesfully loaded checkpoint for base_cwm.")

        return self


def pretrain_vit_base_256_scaffold(**kwargs):
    model = PretrainVisionTransformer(
        img_size=256,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_num_classes=0,
        decoder_embed_dim=768,
        decoder_num_heads=12,
        decoder_depth=12,
        mlp_ratio=4,
        qkv_bias=True,
        k_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )

    return model


def vitbase_8x8patch_2frames_1tube_interp_no_noise(**kwargs):
    model = pretrain_vit_base_256_scaffold(
        patch_size=(8, 8),
        num_frames=2,
        tubelet_size=1,
        use_learnable_pos_emb=True,
        **kwargs,
    )
    return model
