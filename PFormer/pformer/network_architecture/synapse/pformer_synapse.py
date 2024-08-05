from torch import nn
import torch
import torch.nn.functional as F
from typing import Tuple, Union
from pformer.network_architecture.neural_network import SegmentationNetwork
from timm.models.layers import DropPath, to_3tuple, trunc_normal_


class project(nn.Module):
    def __init__(self, in_dim, out_dim, stride, padding, activate, norm, dilation):
        super().__init__()
        self.out_dim = out_dim
        # self.conv1 = DWconv(in_dim, out_dim, kernel_size=3, stride=stride, padding=padding, dilation=dilation)
        # self.conv2 = DWconv(out_dim, out_dim, kernel_size=3, stride=1, padding=padding, dilation=dilation)
        self.conv1 = nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=stride, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv3d(out_dim, out_dim, kernel_size=3, stride=1, padding=padding, dilation=dilation)
        self.activate = activate()
        self.norm1 = norm(out_dim)
        self.norm2 = norm(out_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activate(x)
        # norm1
        Ws, Wh, Ww = x.size(2), x.size(3), x.size(4)
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.norm1(x)
        x = x.transpose(1, 2).contiguous().view(-1, self.out_dim, Ws, Wh, Ww)

        x = self.conv2(x)
        x = self.activate(x)
        # norm2
        Ws, Wh, Ww = x.size(2), x.size(3), x.size(4)
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.norm2(x)
        x = x.transpose(1, 2).contiguous().view(-1, self.out_dim, Ws, Wh, Ww)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=2, in_chans=4, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_3tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim
        stride1 = [patch_size[0], patch_size[1], patch_size[2]]
        stride2 = [patch_size[0] // 2, patch_size[1], patch_size[2]]
        self.proj1 = project(in_chans, embed_dim // 2, stride1, 1, nn.GELU, nn.LayerNorm, dilation=1)
        self.proj2 = project(embed_dim // 2, embed_dim, stride2, 1, nn.GELU, nn.LayerNorm, dilation=1)
        self.pos_embedding = nn.Parameter(torch.randn(1, 32 * 32 * 32, embed_dim))
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, S, H, W = x.size()
        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if S % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - S % self.patch_size[0]))
        x = self.proj1(x)  # B C Ws Wh Ww
        x = self.proj2(x)  # B C Ws Wh Ww
        if self.norm is not None:
            Ws, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2).contiguous()
            x += self.pos_embedding[:, :(n + 1)]
            x = self.norm(x)
            x = x.transpose(1, 2).contiguous().view(-1, self.embed_dim, Ws, Wh, Ww)
        return x


class LN(nn.Module):
    def __init__(self, dim):
        super(LN, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.dim = dim

    def forward(self, x):
        Ws, Wh, Ww = x.size(2), x.size(3), x.size(4)
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.norm(x)
        x = x.transpose(1, 2).contiguous().view(-1, self.dim, Ws, Wh, Ww)
        return x


class DWconv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding=0, dilation=1):
        super(DWconv, self).__init__()
        self.conv1 = nn.Conv3d(in_ch, in_ch, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_ch,
                               dilation=dilation)
        self.conv2 = nn.Conv3d(in_ch, out_ch, 1, 1)
        # self.conv1=nn.Conv3d(in_ch,out_ch,kernel_size=kernel_size,stride=stride,padding=padding)

    def forward(self, x):
        x = self.conv2(self.conv1(x))
        # x=self.conv1(x)
        return x


class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.dim = in_features

    def forward(self, x):
        Ws, Wh, Ww = x.size(2), x.size(3), x.size(4)
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = x.transpose(1, 2).contiguous().view(-1, self.dim, Ws, Wh, Ww)
        return x


class Attention(nn.Module):
    def __init__(self,
                 dim=256,
                 num_head=32,
                 qkv_bias=False,
                 qkv_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_head = num_head
        head_dim = dim // num_head
        self.scale = head_dim ** -0.5 or qkv_scale
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)
        self.dim = dim

    def forward(self, x):
        Ws, Wh, Ww = x.size(2), x.size(3), x.size(4)
        x = x.flatten(2).transpose(1, 2).contiguous()
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_head, C // self.num_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale  # 矩阵相乘操作
        attn = attn.softmax(dim=-1)  # 每一path进行softmax操作
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # 2,1,48
        x = self.proj(x)
        x = self.proj_drop(x)  # Dropout
        x = x.transpose(1, 2).contiguous().view(-1, self.dim, Ws, Wh, Ww)
        return x


class TransBlock(nn.Module):
    def __init__(self, dim=256, dropR=0., mlp_ratio=4, num_head=16):
        super(TransBlock, self).__init__()
        self.norm1 = LN(dim)
        self.attn = Attention(dim=dim, num_head=num_head)
        self.drop_path = DropPath(dropR)
        self.norm2 = LN(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)  # 隐藏层维度扩张后的通道数
        # 多层感知机
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim)
        self.ch_attn = ChannelAttention(in_planes=dim, ratio=16)

    def forward(self, x):
        x = x + self.norm1(self.drop_path(x * self.ch_attn(x) + self.attn(x)))  # 2,1,48
        x = x + self.norm2(self.drop_path(self.mlp(x)))  # mlp后残差连接
        return x


class Trans(nn.Module):
    def __init__(self, dim=256, dropR=0., mlp_ratio=4, num_head=16, depth=2):
        super(Trans, self).__init__()
        self.t_block = nn.Sequential(*[TransBlock(dim=dim, dropR=dropR, mlp_ratio=mlp_ratio, num_head=num_head)
                                       for i in range(depth)])

    def forward(self, x):
        x = self.t_block(x)
        return x


class CAttention(nn.Module):
    def __init__(self,
                 dim=256,
                 num_head=32,
                 qkv_bias=True,
                 qkv_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.,
                 ):
        super(CAttention, self).__init__()
        self.num_head = num_head
        self.dim = dim
        head_dim = dim // num_head
        self.scale = head_dim ** -0.5 or qkv_scale
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x_kv, x_q):
        Ws, Wh, Ww = x_kv.size(2), x_kv.size(3), x_kv.size(4)
        x_kv = x_kv.flatten(2).transpose(1, 2).contiguous()
        x_q = x_q.flatten(2).transpose(1, 2).contiguous()
        Bl, Nl, Cl = x_kv.shape
        Bh, Nh, Ch = x_q.shape
        kv = self.kv(x_kv).reshape(Bl, Nl, 2, self.num_head, Cl // self.num_head).permute(2, 0, 3, 1, 4)
        q = self.q(x_q).reshape(Bh, Nh, 1, self.num_head, Ch // self.num_head).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        q = q[0]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(Bh, Nh, Ch)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.transpose(1, 2).contiguous().view(-1, self.dim, Ws, Wh, Ww)
        return x


class CTransBlock(nn.Module):
    def __init__(self, dim=256, dropR=0., mlp_ratio=4, num_head=16):
        super(CTransBlock, self).__init__()
        self.norm1 = LN(dim)
        self.attn = CAttention(dim=dim, num_head=num_head)
        self.drop_path = DropPath(dropR)
        self.norm2 = LN(dim)
        mlp_hiddin_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hiddin_dim)
        self.ch_attn = ChannelAttention(in_planes=dim, ratio=16)

    def forward(self, x, x2):
        x = x + self.norm1(self.drop_path(x * self.ch_attn(x) + self.attn(x, x2)))
        x = x + self.norm2(self.drop_path(self.mlp(x)))
        return x, x2


class CTrans(nn.Module):
    def __init__(self, dim=256, dim_up=192, dropR=0., mlp_ratio=4, num_head=16, depth=2):
        super(CTrans, self).__init__()
        self.c_block1 = CTransBlock(dim=dim, dropR=dropR, mlp_ratio=mlp_ratio, num_head=num_head)
        self.c_block2 = CTransBlock(dim=dim, dropR=dropR, mlp_ratio=mlp_ratio, num_head=num_head)
        self.depth = depth
        self.ch_ex = DWconv(dim_up, dim, 1, 1, 0)
        self.norm = LN(dim_up)
        self.ch_re = DWconv(dim, dim_up, 1, 1, 0)
        self.activate = nn.GELU()

    def forward(self, x, x2):
        # x2 = self.ch_ex(x2)
        x_, _ = self.c_block1(x, x2)
        x_2, _ = self.c_block2(x2, x)
        output = torch.concat([x_, x_2], dim=1)
        return output


class CPBAttention(nn.Module):
    def __init__(self,
                 dim=256,
                 num_head=32,
                 qkv_bias=True,
                 qkv_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.,
                 k=512):
        super(CPBAttention, self).__init__()
        self.num_head = num_head
        self.dim = dim
        head_dim = dim // num_head
        self.scale = head_dim ** -0.5 or qkv_scale
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)
        self.Spa = SpatialAttention(kernel_size=7)
        self.k = k
        self.res = DWconv(in_ch=dim, out_ch=dim, kernel_size=3, stride=1, padding=1)

    #         self.res2=DWconv(in_ch=dim,out_ch=dim,kernel_size=3,stride=1,padding=1)

    def forward(self, x_kv, x_q):
        # 取topk
        _, C_, _, _, _ = x_kv.shape
        x_kv_top = self.Spa(x_kv)
        SpA = x_kv_top
        x_kv_top = x_kv_top.flatten(2).transpose(1, 2).contiguous()
        top, index = torch.topk(x_kv_top, k=self.k, dim=1)
        index = index.repeat(1, 1, self.dim)
        # 获得k_g,v_g,q
        Ws, Wh, Ww = x_kv.size(2), x_kv.size(3), x_kv.size(4)
        x_kv = x_kv.flatten(2).transpose(1, 2).contiguous()
        B, N, C = x_kv.shape
        kv = self.kv(x_kv).reshape(B, N, 2, C).permute(2, 0, 1, 3)
        k, v = kv[0], kv[1]
        k_g = torch.gather(k, dim=1, index=index).reshape(B, self.k, self.num_head, C // self.num_head).transpose(1, 2)
        v_g = torch.gather(v, dim=1, index=index).reshape(B, self.k, self.num_head, C // self.num_head).transpose(1, 2)
        x_q = x_q.flatten(2).transpose(1, 2).contiguous()
        Bh, Nh, Ch = x_q.shape
        q = self.q(x_q).reshape(Bh, Nh, 1, self.num_head, Ch // self.num_head).permute(2, 0, 3, 1, 4)
        q = q[0]
        # 计算attn
        attn = (q @ k_g.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn)
        attn = attn.softmax(dim=-1)
        x = (attn @ v_g).transpose(1, 2).reshape(Bh, Nh, Ch)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.transpose(1, 2).contiguous().view(-1, self.dim, Ws, Wh, Ww)
        # output=x+dwconv(v)
        v = self.res(v.transpose(1, 2).contiguous().view(-1, self.dim, Ws, Wh, Ww))
        #         x=self.res2(x+v*SpA)
        x = x + v
        return x


class CPBTransBlock(nn.Module):
    def __init__(self, dim=256, dropR=0., mlp_ratio=4, num_head=16, k=512):
        super(CPBTransBlock, self).__init__()
        self.norm1 = LN(dim)
        self.attn = CPBAttention(dim=dim, num_head=num_head, k=k)
        self.drop_path = DropPath(dropR)
        self.norm2 = LN(dim)
        mlp_hiddin_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hiddin_dim)
        self.ch_attn = ChannelAttention(in_planes=dim, ratio=16)

    def forward(self, x, x2):
        x = x + self.norm1(self.drop_path(x * self.ch_attn(x) + self.attn(x, x2)))
        x = x + self.norm2(self.drop_path(self.mlp(x)))

        return x, x2


class CPBTrans(nn.Module):
    def __init__(self, dim=256, dropR=0., mlp_ratio=4, num_head=16, depth=2, k=512):
        super(CPBTrans, self).__init__()
        self.c_block1 = CPBTransBlock(dim=dim, dropR=dropR, mlp_ratio=mlp_ratio, num_head=num_head, k=k)
        self.c_block2 = CPBTransBlock(dim=dim, dropR=dropR, mlp_ratio=mlp_ratio, num_head=num_head, k=k)
        self.depth = depth
        self.norm = LN(dim)
        self.activate = nn.GELU()

    #         self.para1 = nn.Parameter(torch.tensor([0.5]), requires_grad=True)
    #         self.para2 = nn.Parameter(torch.tensor([0.5]), requires_grad=True)

    def forward(self, x, x2):
        x_, _ = self.c_block1(x, x2)
        x_2, _ = self.c_block2(x2, x)
        output = torch.concat([x_, x_2], dim=1)
        return output


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv3d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.fc = nn.Sequential(nn.Conv3d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv3d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class PBiAttention(nn.Module):
    def __init__(self,
                 dim=256,
                 num_head=32,
                 qkv_bias=False,
                 qkv_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.,
                 Spa_convk=7,
                 k=512):
        super(PBiAttention, self).__init__()
        self.num_head = num_head
        head_dim = dim // num_head
        self.scale = head_dim ** -0.5 or qkv_scale
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # self.kv=nn.Linear(dim,dim*2,bias=qkv_bias)
        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)
        self.dim = dim
        self.Spa = SpatialAttention(kernel_size=Spa_convk)
        self.k = k
        self.res = DWconv(in_ch=dim, out_ch=dim, kernel_size=3, stride=1, padding=1)

    #         self.res2=DWconv(in_ch=dim,out_ch=dim,kernel_size=3,stride=1,padding=1)

    def forward(self, x):
        x_spa = self.Spa(x)
        SpA = x_spa
        x_spa = x_spa.flatten(2).transpose(1, 2).contiguous()
        top, index = torch.topk(x_spa, k=self.k, dim=1)
        index = index.repeat(1, 1, self.dim)
        Ws, Wh, Ww = x.size(2), x.size(3), x.size(4)
        x = x.flatten(2).transpose(1, 2).contiguous()
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, C).permute(2, 0, 1, 3)
        # qkv=self.qkv(x).reshape(B,N,3,self.num_head,C//self.num_head).permute(2,0,3,1,4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q.reshape(B, N, self.num_head, C // self.num_head).transpose(1, 2)
        k_g = torch.gather(k, dim=1, index=index).reshape(B, self.k, self.num_head, C // self.num_head).transpose(1, 2)
        v_g = torch.gather(v, dim=1, index=index).reshape(B, self.k, self.num_head, C // self.num_head).transpose(1, 2)
        attn = (q @ k_g.transpose(-2, -1)) * self.scale  # 矩阵相乘操作
        attn = attn.softmax(dim=-1)  # 每一path进行softmax操作
        attn = self.attn_drop(attn)
        x = (attn @ v_g).transpose(1, 2).reshape(B, N, C)  # 2,1,48
        x = self.proj(x)
        x = self.proj_drop(x)  # Dropout
        x = x.transpose(1, 2).contiguous().view(-1, self.dim, Ws, Wh, Ww)
        v = self.res(v.transpose(1, 2).contiguous().view(-1, self.dim, Ws, Wh, Ww))
        #         x = self.res2(x+v*SpA)
        x = x + v
        return x


class PBTransBlock(nn.Module):
    def __init__(self, dim=256, dropR=0., mlp_ratio=4, num_head=16, qkv_bias=False, qkv_scale=None, attn_drop_ratio=0.,
                 proj_drop_ratio=0.,
                 Spa_convk=3, k=512):
        super(PBTransBlock, self).__init__()
        self.norm1 = LN(dim)
        self.attn = PBiAttention(dim=dim, num_head=num_head, qkv_bias=qkv_bias,
                                 qkv_scale=qkv_scale, attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=proj_drop_ratio,
                                 Spa_convk=Spa_convk, k=k)
        self.drop_path = DropPath(dropR)
        self.norm2 = LN(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)  # 隐藏层维度扩张后的通道数
        # 多层感知机
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim)
        self.ch_attn = ChannelAttention(in_planes=dim, ratio=16)

    def forward(self, x):
        # x = x + self.drop_path(self.attn(self.norm1(x)))  # 2,1,48
        # x = x + self.drop_path(self.mlp(self.norm2(x)))  # mlp后残差连接
        x = x + self.norm1(self.drop_path(x * self.ch_attn(x) + self.attn(x)))
        x = x + self.norm2(self.drop_path(self.mlp(x)))

        return x


class PBTrans(nn.Module):
    def __init__(self, dim=256, dropR=0., mlp_ratio=4, num_head=16, qkv_bias=False, qkv_scale=None, attn_drop_ratio=0.,
                 proj_drop_ratio=0.,
                 Spa_convk=3, k=512, depth=2):
        super(PBTrans, self).__init__()
        self.t_block = nn.Sequential(*[
            PBTransBlock(dim=dim, dropR=dropR, mlp_ratio=mlp_ratio, num_head=num_head, qkv_bias=qkv_bias,
                         qkv_scale=qkv_scale, attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=proj_drop_ratio,
                         Spa_convk=Spa_convk, k=k)
            for i in range(depth)])

    def forward(self, x):
        x = self.t_block(x)
        return x


# class local_block(nn.Module):
#     def __init__(self,in_ch,out_ch):
#         super(local_block, self).__init__()
#         self.norm1=LN(in_ch)
#         self.conv1=DWconv(in_ch,out_ch,3,1,1)
#         self.norm2=LN(out_ch)
#         self.mlp=Mlp(in_ch,in_ch*2,in_ch,act_layer=nn.GELU)
#     def forward(self,x):
#         x=x+self.mlp(self.norm2(x))
#         return x

class local_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(local_block, self).__init__()
        self.conv1 = DWconv(in_ch, out_ch, kernel_size=3, stride=1, padding=1, dilation=1)
        self.norm1 = LN(out_ch)
        self.conv2 = DWconv(out_ch, out_ch, kernel_size=3, stride=1, padding=1, dilation=1)
        self.norm2 = LN(out_ch)
        self.activate = nn.GELU()

    def forward(self, x):
        x = self.activate(self.norm1(self.conv1(x)))
        x = self.activate(self.norm2(self.conv2(x)))
        return x


class global_block(nn.Module):
    def __init__(self, dim, k=2048, num_head=16, depth=1):
        super(global_block, self).__init__()
        self.pbt = PBTrans(dim=dim, k=k, num_head=num_head, depth=depth)

    def forward(self, x):
        x = self.pbt(x)
        return x


class bottle_global_block(nn.Module):
    def __init__(self, dim, num_head, depth):
        super(bottle_global_block, self).__init__()
        self.block = Trans(dim=dim, num_head=num_head, depth=depth)

    def forward(self, x):
        x = self.block(x)
        return x


class bottle_layer(nn.Module):
    def __init__(self, in_ch, out_ch, num_head, depth):
        super(bottle_layer, self).__init__()
        self.layer = nn.Sequential(local_block(in_ch, out_ch),
                                   bottle_global_block(dim=out_ch, num_head=num_head, depth=depth),
                                   #                                  local_block(out_ch,out_ch),
                                   #                                  bottle_global_block(dim=out_ch,num_head=num_head,depth=depth),
                                   # bottle_global_block(dim=in_ch,num_head=num_head,depth=depth),
                                   # local_block(out_ch,out_ch),
                                   # bottle_global_block(dim=out_ch,num_head=num_head,depth=depth),
                                   # local_block(out_ch,out_ch),
                                   # bottle_global_block(dim=out_ch,num_head=num_head,depth=depth)
                                   )

    def forward(self, x):
        x = self.layer(x)
        return x


class encoder_layer(nn.Module):
    def __init__(self, in_ch, out_ch, num_head, depth, k):
        super(encoder_layer, self).__init__()
        self.layer = nn.Sequential(local_block(in_ch, out_ch),
                                   global_block(dim=out_ch, num_head=num_head, depth=depth, k=k),
                                   #                                  local_block(out_ch,out_ch),
                                   #                                  global_block(dim=out_ch,num_head=num_head,depth=depth,k=k)
                                   # global_block(dim=in_ch,num_head=num_head,depth=depth,k=k),
                                   # local_block(out_ch,out_ch),
                                   # global_block(dim=out_ch,num_head=num_head,depth=depth,k=k[2]),
                                   # local_block(out_ch,out_ch),
                                   # global_block(dim=out_ch,num_head=num_head,depth=depth,k=k)
                                   )

    def forward(self, x):
        x = self.layer(x)
        return x


class Downsampling(nn.Module):
    def __init__(self, ch, conv=True, kernel_size=2, stride=2, padding=1):
        super(Downsampling, self).__init__()
        self.conv = conv
        if conv == True:
            self.down = nn.Sequential(
                nn.Conv3d(in_channels=ch, out_channels=ch, kernel_size=kernel_size, stride=stride, padding=padding),
                LN(ch),
                nn.GELU())
        else:
            self.down = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.down(x)
        return x


class Upsampling(nn.Module):
    def __init__(self, ch, conv=True, kernel_size=2, stride=2, padding=0):
        super(Upsampling, self).__init__()
        self.conv = conv
        if conv == False:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=kernel_size, mode='trilinear'),
                DWconv(ch, ch, kernel_size=1, stride=1),
                LN(ch),
                nn.GELU()
            )
        else:
            self.up = nn.Sequential(
                # nn.ConvTranspose3d(in_channels=ch, out_channels=ch, kernel_size=kernel_size, stride=stride),
                nn.ConvTranspose3d(in_channels=ch, out_channels=ch // 2, kernel_size=kernel_size, stride=stride,
                                   output_padding=padding),
                LN(ch // 2),
                nn.GELU()
            )

    def forward(self, x):
        x = self.up(x)
        return x


class final_patch_expanding(nn.Module):
    def __init__(self, dim, num_class, patch_size):
        super().__init__()
        # self.up = nn.ConvTranspose3d(dim, num_class, patch_size, patch_size)
        self.up = nn.Sequential(nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear'),
                                DWconv(dim, out_ch=32, kernel_size=3, stride=1, padding=1), LN(32), nn.GELU(),
                                nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear'),
                                DWconv(32, 14, kernel_size=1, stride=1)
                                )

    def forward(self, x):
        x = self.up(x)

        return x


class final_patch_expanding2(nn.Module):
    def __init__(self, dim, num_class, patch_size):
        super().__init__()
        # self.up = nn.ConvTranspose3d(dim, num_class, patch_size, patch_size)
        self.up = nn.Sequential(nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear'),
                                DWconv(dim, out_ch=32, kernel_size=3, stride=1, padding=1), LN(32), nn.GELU(),
                                nn.Upsample(scale_factor=(1, 1, 1), mode='trilinear'),
                                DWconv(32, 14, kernel_size=1, stride=1)
                                )

    def forward(self, x):
        x = self.up(x)

        return x


class Encoder(nn.Module):
    def __init__(self, ch=[64, 128, 256, 512, 1024], k=[2048, 512], num_head=[4, 8, 16, 32], depth=[2, 2, 1, 1]):
        super(Encoder, self).__init__()
        self.e1 = encoder_layer(in_ch=ch[0], out_ch=ch[1], k=k[0], num_head=num_head[0], depth=depth[0])
        self.d1 = Downsampling(ch[1], conv=True, kernel_size=3, stride=2, padding=1)
        self.e2 = encoder_layer(in_ch=ch[1], out_ch=ch[2], k=k[1], num_head=num_head[1], depth=depth[1])
        self.d2 = Downsampling(ch[2], conv=True, kernel_size=3, stride=2, padding=1)
        self.e3 = encoder_layer(in_ch=ch[2], out_ch=ch[3], k=k[2], num_head=num_head[2], depth=depth[2])
        self.d3 = Downsampling(ch[3], conv=True, kernel_size=3, stride=2, padding=1)
        self.e4 = encoder_layer(in_ch=ch[3], out_ch=ch[4], k=k[3], num_head=num_head[3], depth=depth[3])

    def forward(self, x):
        x1 = self.e1(x)
        x2 = self.e2(self.d1(x1))
        x3 = self.e3(self.d2(x2))
        x4 = self.e4(self.d3(x3))
        return x1, x2, x3, x4


class Decoder(nn.Module):
    def __init__(self, ch=[64, 128, 256, 512, 1024], k=[2048, 512], num_head=[4, 8, 16, 32], depth=[1, 1, 1, 2]):
        super(Decoder, self).__init__()
        # self.d4 = bottle_layer(in_ch=ch[3], out_ch=ch[3],num_head=num_head[3],depth=depth[3])
        self.u3 = Upsampling(ch=ch[4], conv=True, kernel_size=2, stride=2)
        self.d3 = encoder_layer(in_ch=ch[3] * 2, out_ch=ch[3], num_head=num_head[2], depth=depth[2], k=k[2])
        self.u2 = Upsampling(ch=ch[3], conv=True, kernel_size=2, stride=2)
        self.d2 = encoder_layer(in_ch=ch[2] * 2, out_ch=ch[2], num_head=num_head[1], depth=depth[1], k=k[1])
        self.u1 = Upsampling(ch=ch[2], conv=True, kernel_size=2, stride=2)
        self.d1 = encoder_layer(in_ch=ch[1] * 2, out_ch=ch[1], num_head=num_head[0], depth=depth[0], k=k[0])

        # self.cc4 = CTrans(dim=ch[3], mlp_ratio=4, num_head=num_head[3])
        #         self.cc3 = CPBTrans(dim=ch[3], mlp_ratio=4, num_head=num_head[2],k=k[2])
        #         self.cc2 = CPBTrans(dim=ch[2], mlp_ratio=4, num_head=num_head[1],k=k[1])

        #         self.cc1 = CPBTrans(dim=ch[1],mlp_ratio=4,num_head=num_head[0],k=k[0])

        self.seg = final_patch_expanding(dim=ch[1], num_class=14, patch_size=(2, 4, 4))
        self.seg1 = final_patch_expanding2(dim=ch[2], num_class=14, patch_size=(2, 4, 4))
        self.seg2 = final_patch_expanding2(dim=ch[3], num_class=14, patch_size=(2, 4, 4))

    #         self.convblock = nn.Sequential(DWconv(in_ch=1, out_ch=16, kernel_size=3, stride=1, padding=1), LN(16), nn.GELU(),
    #                                 DWconv(16, 16, kernel_size=1, stride=1),LN(16),nn.GELU())

    def forward(self, x, x1, x2, x3, x4):
        output = []
        x4 = self.u3(x4)
        x3 = self.d3(torch.concat([x3, x4], dim=1))
        x3_out = self.seg2(x3)
        x3 = self.u2(x3)
        x2 = self.d2(torch.concat([x2, x3], dim=1))
        x2_out = self.seg1(x2)
        x2 = self.u1(x2)
        x1 = self.d1(torch.concat([x1, x2], dim=1))
        x1_out = self.seg(x1)
        output.append(x3_out)
        output.append(x2_out)
        output.append(x1_out)
        return output


class MyNet(SegmentationNetwork):
    def __init__(self, ch=[64, 128, 256, 512, 1024], k=[512, 256, 128], num_head=[4, 8, 16, 32], depth=[1, 1, 1, 1],
                 deep_supervision=False):
        super(MyNet, self).__init__()
        self.Patch = PatchEmbed(patch_size=2, in_chans=1, embed_dim=ch[0], norm_layer=None)
        self.E = Encoder(ch=ch, k=k, num_head=num_head, depth=depth)
        self.D = Decoder(ch=ch, k=k, num_head=num_head, depth=depth)
        self.do_ds = deep_supervision

    def forward(self, x):
        x_p = self.Patch(x)
        x1, x2, x3, x4 = self.E(x_p)
        output = self.D(x, x1, x2, x3, x4)
        if self.do_ds:
            return output[::-1]
        else:
            return output[2]


if __name__ == "__main__":
    network = MyNet(ch=[128, 128, 256, 512, 1024], k=[512, 128, 64, 32], num_head=[2, 4, 8, 16], depth=[2, 2, 2, 2],
                    deep_supervision=True)
    macs, params = get_model_complexity_info(network, (1, 64, 128, 128), as_strings=True, print_per_layer_stat=True)
    print("|%s |%s" % (macs, params))
