import torch
import torch.nn as nn
import einops
from dominate.tags import output

from torch.nn import functional as F
from torch.jit import Final
from timm.layers import use_fused_attn
from timm.models.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_, get_act_layer

__all__ = ['SVDNoiseUnet', 'SVDNoiseUnet_Concise']


class Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SVDNoiseUnet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, resolution=256):  # resolution = size // 8
        super(SVDNoiseUnet, self).__init__()
        self.upconv = nn.Conv2d(7, in_channels * 3, (1, 1), (1, 1), (0, 0))
        self.downconv = nn.Conv2d(in_channels * 3, out_channels, (1, 1), (1, 1), (0, 0))
        _in = int(resolution * 9 // 3)
        _out = int(resolution * 9 // 3)
        self.mlp1 = nn.Sequential(
            nn.Linear(_in, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, _out),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(_in, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, _out),
        )

        self.mlp3 = nn.Sequential(
            nn.Linear(_in, _out),
        )

        self.attention = Attention(_out)

        self.bn = nn.BatchNorm2d(_out)

        self.mlp4 = nn.Sequential(
            nn.Linear(_out, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, _out),
        )

    def forward(self, x_t, mask, mask_synthetic):
        b, c, h, w = x_t.shape
        input = torch.cat([x_t, mask, mask_synthetic], dim=1)
        x = self.upconv(input)
        x = einops.rearrange(x, "b (a c)h w ->b (a h)(c w)", a=3, c=3)  # x -> [1, 256, 256]
        U, s, V = torch.linalg.svd(x)  # U->[b 256 256], s-> [b 256], V->[b 256 256]
        U_T = U.permute(0, 2, 1)
        out = self.mlp1(U_T) + self.mlp2(V) + self.mlp3(s).unsqueeze(1)  # s -> [b, 1, 256]  => [b, 256, 256]
        out = self.attention(out).mean(1)
        out = self.mlp4(out) + s
        out = U @ torch.diag_embed(out) @ V
        x = einops.rearrange(out, "b (a h)(c w) -> b (a c) h w", a=3, c=3)
        noise = self.downconv(x)
        return mask_synthetic * (1. - mask) + mask * noise


if __name__ == '__main__':
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from data.dataset import InpaintDataset
    import matplotlib.pyplot as plt
    import numpy as np
    from tqdm import tqdm


    dataset = InpaintDataset(data_root='datasets', image_size=[256, 256], pattern='MAP', phase='train')


    dataloader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=4)


    model = SVDNoiseUnet(in_channels=3, out_channels=3, resolution=256).cuda()


    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)


    num_epochs = 1
    output_interval = 100

    for epoch in range(num_epochs):

        with tqdm(total=len(dataloader), desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch') as pbar:
            for step, batch in enumerate(dataloader):
                mask_synthetic = batch['mask_synthetic'].cuda()
                mask = batch['mask'].cuda()
                noise = torch.randn_like(mask_synthetic).cuda()


                output = model(noise, mask, mask_synthetic)


                target = mask_synthetic * (1 - mask) + noise * mask
                loss = criterion(output, target)


                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


                pbar.update(1)
                pbar.set_postfix(loss=loss.item())

                if step % output_interval == 0:
                    print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{step}], Loss: {loss.item():.4f}")

                    output_image = output[0].squeeze().detach().cpu().numpy().transpose(1, 2, 0)
                    target_image = target[0].squeeze().detach().cpu().numpy().transpose(1, 2, 0)

                    output_image = ((output_image + 1) / 2 * 255).astype(np.uint8)
                    target_image = ((target_image + 1) / 2 * 255).astype(np.uint8)

                    plt.figure(figsize=(10, 5))

                    plt.subplot(1, 2, 1)
                    plt.imshow(output_image)
                    plt.title("Model Output")
                    plt.axis('off')

                    plt.subplot(1, 2, 2)
                    plt.imshow(target_image)
                    plt.title("Target Image")
                    plt.axis('off')

                    plt.show()

    torch.save(model.state_dict(), 'svd_noise_unet.pth')
