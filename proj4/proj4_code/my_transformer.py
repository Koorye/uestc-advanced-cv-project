import math
import torch
from functools import partial
from torch import nn
from torch.nn.init import trunc_normal_, _calculate_fan_in_and_fan_out


class MLP(nn.Module):
    def __init__(self,
                 in_features,
                 hid_features,
                 out_features,
                 dropout=0.):
        super(MLP, self).__init__()

        out_features = out_features or in_features
        hid_features = hid_features or in_features

        self.fc1 = nn.Linear(in_features, hid_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hid_features, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Attention(nn.Module):
    def __init__(self,
                 dim,
                 n_heads=8,
                 qkv_bias=False,
                 attn_dropout=0.,
                 proj_dropout=0.):
        super(Attention, self).__init__()

        self.n_heads = n_heads
        head_dim = dim // n_heads
        self.scale = head_dim ** -.5

        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.attn_dropout = nn.Dropout(attn_dropout)

        self.proj = nn.Linear(dim, dim)
        self.proj_dropout = nn.Dropout(proj_dropout)

    def forward(self, x):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.n_heads, c//self.n_heads).permute(2, 0, 3, 1, 4)
        q,k,v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        x = (attn @ v).transpose(1,2).reshape(b,n,c)
        x = self.proj(x)
        x = self.proj_dropout(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self,
                 dim,
                 n_heads,
                 mlp_ratio,
                 qkv_bias=False,
                 dropout=0.,
                 attn_dropout=0.,
                 proj_dropout=0.,
                 norm_layer=nn.LayerNorm):
        super(EncoderBlock, self).__init__()

        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, n_heads, qkv_bias,
                              attn_dropout, proj_dropout)
        self.dropout = nn.Dropout(dropout)

        self.norm2 = norm_layer(dim)
        mlp_hid_dim = int(dim * mlp_ratio)
        self.mlp = MLP(dim, mlp_hid_dim, dim, dropout)

    def forward(self, x):
        x = x + self.dropout(self.attn(self.norm1(x)))
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):

    def __init__(self, img_size=224,
                 patch_size=16,
                 in_c=3,
                 embed_dim=768,
                 norm_layer=None):
        super(PatchEmbed, self).__init__()

        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0],
                          img_size[1] // patch_size[1])
        self.n_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_c, embed_dim,
                              kernel_size=patch_size,
                              stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x):
        # (b,c,h,w) -> (b,c,hw) -> (b,hw,c)
        x = self.proj(x).flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x


def variance_scaling_(tensor, scale=1.0, mode='fan_in', distribution='normal'):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == 'fan_in':
        denom = fan_in
    elif mode == 'fan_out':
        denom = fan_out
    elif mode == 'fan_avg':
        denom = (fan_in + fan_out) / 2

    variance = scale / denom

    if distribution == "truncated_normal":
        # constant is stddev of standard normal truncated to (-2, 2)
        trunc_normal_(tensor, std=math.sqrt(variance) / .87962566103423978)
    elif distribution == "normal":
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")

def lecun_normal_(tensor):
    variance_scaling_(tensor, mode='fan_in', distribution='truncated_normal')

def init_vit_weights(module, name='', head_bias=0.):
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        elif name.startswith('pre_logits'):
            lecun_normal_(module.weight)
            nn.init.zeros_(module.bias)
        else:
            trunc_normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)


class MyVisionTransformer(nn.Module):
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_c=3,
                 n_classes=1000,
                 embed_dim=768,
                 depth=12,
                 n_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 representation_size=None,
                 proj_dropout=0.,
                 attn_dropout=0.,
                 dropout=0.,
                 norm_layer=None):
        super(VisionTransformer, self).__init__()

        self.n_classes = n_classes
        self.n_features = self.embed_dim = embed_dim
        self.n_tokens = 1

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.patch_embed = PatchEmbed(img_size, patch_size, in_c, embed_dim)
        n_patches = self.patch_embed.n_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(
            1, n_patches+self.n_tokens, embed_dim))
        self.pos_dropout = nn.Dropout(dropout)

        self.blocks = nn.Sequential(*[
            EncoderBlock(embed_dim, n_heads, mlp_ratio, qkv_bias,
                         dropout, attn_dropout, proj_dropout, norm_layer)
            for _ in range(depth)
        ])

        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(self.n_features, n_classes)

        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        self.apply(init_vit_weights)

    def forward(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        x = self.pos_dropout(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        x = self.head(x[:, 0])
        return x
