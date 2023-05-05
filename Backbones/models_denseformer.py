import math
from functools import partial

import torch
import torch.nn as nn

from collections import OrderedDict


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])  # 14x14
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class Attention(nn.Module):
    def __init__(self,
                 dim,  # 输入token的dim
                 num_heads=8,  # 多头注意力机制   就是分成不同的头 然后分别进行注意力机制，然后再结和合
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]       batch_size,196+1,768
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # qkv
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # import pandas as pd
        # pd.DataFrame(x.mean(dim=-1).cpu().detach().numpy()).to_csv('test.csv', header=False, index=False, mode='a')
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self,
                 dim=196,
                 num_heads=14,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(ChannelAttention, self).__init__()

        ## channel attention
        self.channel_num_heads = 14  # num_heads
        N = 14 * 14  # dim
        channel_head_dim = N // self.channel_num_heads
        self.channel_scale = qk_scale or channel_head_dim ** -0.5
        self.channel_qkv = nn.Linear(N, N * 3, bias=qkv_bias)
        self.channel_attn_drop = nn.Dropout(attn_drop_ratio)
        self.channel_proj = nn.Linear(N, N)
        self.channel_proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]       batch_size,196+1,768
        B, N, C = x.shape

        assert N == 99 or N == 197 or N == 50, 'N must be 14 * 14 * mask_ratio + 1, mask_ratio in [0, 0.5, 0.75]'
        cls_tokens_x = x[:, :1, :]  # [B, 1, N]
        x = x[:, 1:, :].transpose(-2, -1)  # [B, N+1, C]->[B, C, N]
        if N == 99:
            x = x.repeat_interleave(2, 2)  # [B,C,98]->[B,C,196]
        elif N == 50:
            x = x.repeat_interleave(4, 2)  # [B,C,49]->[B,C,196]
        # x = x.repeat_interleave(math.ceil(196 / (N - 1)), 2)[:, :, :196]  # match N to 196

        qkv_channel = self.channel_qkv(x).reshape(B, C, 3, self.channel_num_heads,
                                                  196 // self.channel_num_heads).permute(2, 0, 3, 1, 4)  # N*2=196
        q_channel, k_channel, v_channel = qkv_channel[0], qkv_channel[1], qkv_channel[2]  # [5,14,32,14]
        attn_channel = (q_channel @ k_channel.transpose(-2, -1)) * self.channel_scale  # [5,14,32,32]
        attn_channel = attn_channel.softmax(dim=-1)
        attn_channel = self.channel_attn_drop(attn_channel)

        x = (attn_channel @ v_channel).transpose(1, 2).reshape(B, C, 196)  # [5,14,32,14]->[5,32,14,14]->[5,32,196]
        x = self.channel_proj(x)
        x = self.channel_proj_drop(x)
        if N == 99:
            x = (x[:, :, 0::2] + x[:, :, 1::2]) / 2  # [5,32,196]->[5,32,98]
        elif N == 50:
            x = (x[:, :, 0::4] + x[:, :, 1::4] + x[:, :, 2::4] + x[:, :, 3::4]) / 4  # [5,32,196]->[5,32,49]
        x = torch.cat((cls_tokens_x, x.transpose(-2, -1)), dim=1)  # [5,32,98]->[5,98,32]->[5,99,32]

        return x


class DenseBlock(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 out_feature=32):
        super(DenseBlock, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=32, act_layer=act_layer,
                       drop=drop_ratio)
        # channel attention
        self.norm3 = norm_layer(32)
        self.channelattn = ChannelAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        self.norm4 = norm_layer(32)
        mlp_hidden_dim = int(32 * mlp_ratio)
        self.channelmlp = Mlp(in_features=32, hidden_features=mlp_hidden_dim, out_features=32, act_layer=act_layer,
                       drop=drop_ratio)

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            x = x
        else:
            x = torch.cat(x, 2)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = self.drop_path(self.mlp(self.norm2(x)))

        x = x + self.drop_path(self.channelattn(self.norm3(x)))
        x = x + self.drop_path(self.channelmlp(self.norm4(x)))
        return x


class DenseFormer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3,
                 embed_dim=384, depth=12, num_heads=8, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_c (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_ratio (float): dropout rate
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
        """
        super(DenseFormer, self).__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            DenseBlock(embed_dim + 32 * i, num_heads, mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                       drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                       norm_layer=norm_layer, act_layer=act_layer, )
            for i in range(depth)])
        self.end_dim = embed_dim + 32 * depth
        self.norm = norm_layer(self.end_dim)

        self.decoder_pred = nn.Linear(self.end_dim, patch_size ** 2 * 1, bias=True)  # decoder to patch

        # Weight init
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        out_dim = x.shape[2] // p // p
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, out_dim))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], out_dim, h * p, h * p))
        return imgs

    def freeze_part(self, ):
        for param in self.named_parameters():
            param[1].requires_grad = False
            # print('{} is freezeing.'.format(param[0]))
            # print(param[1].requires_grad)
        self.decoder_pred.weight.requires_grad = True
        self.decoder_pred.bias.requires_grad = True
        print('All model but decoder_pred is freezeing.')
        # print(self.decoder_pred.requires_grad)

    def unfreeze_part(self, ):
        for param in self.named_parameters():
            param[1].requires_grad = True
            # print('{} is unfreezeing.'.format(param[0]))
            # print(param[1].requires_grad)
        print('All model is unfreezeing.')


    def forward_features(self, x):
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x)  # [B, 196, 768]
        # [1, 1, 768] -> [B, 1, 768]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = self.pos_drop(x + self.pos_embed)
        features = [x]
        for blk in self.blocks:
            # x = blk(x)
            new_features = blk(features)
            features.append(new_features)
        x = torch.cat(features, 2)
        # import pandas as pd
        # pd.DataFrame(features[-1].mean(dim=1).cpu().detach().numpy()).to_csv('test.csv', header=False, index=False, mode='a')
        x = self.norm(x)
        x = x[:, 1:, :]

        return x

    def forward(self, x):
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)

        x = self.forward_features(x)
        x = self.decoder_pred(x)
        x = self.unpatchify(x)

        return x


def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


if __name__ == '__main__':
    input = torch.randn(5, 1, 224, 224)
    model = DenseFormer()  # 6.51 GMac 34.74 M
    print(model)
    out = model(input)
    print('out.shape:')
    print(out.shape)
    out = model(input)
    print('difference:')
    print(((out - input) ** 2).sum())
    from ptflops import get_model_complexity_info

    flops, params = get_model_complexity_info(model, (1, 224, 224), as_strings=True, print_per_layer_stat=True)
    print(flops, params)
