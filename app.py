import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import io


def make_patches(images, patch_size=16):
    batch_size, channels, height, width = images.shape
    rowwise_patches = height // patch_size
    colwise_patches = width // patch_size

    x = images.reshape(
        batch_size,
        channels,
        rowwise_patches,
        patch_size,
        colwise_patches,
        patch_size
    )

    x = x.permute(0, 2, 4, 1, 3, 5)

    x = x.reshape(
        batch_size,
        rowwise_patches * colwise_patches,
        channels * patch_size * patch_size
    )

    return x


def put_patches_back(patches, patch_size=16, image_size=224):
    batch_size = patches.shape[0]
    channels = 3
    patches_per_side = image_size // patch_size

    x = patches.reshape(
        batch_size,
        patches_per_side,
        patches_per_side,
        channels,
        patch_size,
        patch_size
    )

    x = x.permute(0, 3, 1, 4, 2, 5)

    x = x.reshape(batch_size, channels, image_size, image_size)

    return x


def random_masking(patches, mask_ratio=0.75):
    batch_size, total_patches, patch_dim = patches.shape

    num_keep = int(total_patches * (1 - mask_ratio))

    rand_values = torch.rand(batch_size, total_patches, device=patches.device)

    shuffled_ids = torch.argsort(rand_values, dim=1)

    restore_ids = torch.argsort(shuffled_ids, dim=1)

    keep_ids = shuffled_ids[:, :num_keep]

    visible_patches = torch.gather(
        patches,
        dim=1,
        index=keep_ids.unsqueeze(-1).expand(-1, -1, patch_dim)
    )

    mask = torch.ones(batch_size, total_patches, device=patches.device)
    mask[:, :num_keep] = 0
    mask = torch.gather(mask, dim=1, index=restore_ids)

    return visible_patches, mask, restore_ids


def get_2d_pos_embed(embed_dim, grid_size):
    assert embed_dim % 2 == 0

    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)

    pos_w = grid[0].reshape(-1)
    pos_h = grid[1].reshape(-1)

    dim_each = embed_dim // 2

    omega_dim = dim_each // 2
    omega = np.arange(omega_dim, dtype=np.float32)
    omega = omega / omega_dim
    omega = 1.0 / (10000 ** omega)

    out_h = np.einsum('m,d->md', pos_h, omega)
    out_w = np.einsum('m,d->md', pos_w, omega)

    emb_h = np.concatenate([np.sin(out_h), np.cos(out_h)], axis=1)
    emb_w = np.concatenate([np.sin(out_w), np.cos(out_w)], axis=1)

    emb = np.concatenate([emb_h, emb_w], axis=1)

    return emb


class TransformerBlock(nn.Module):
   def __init__(self, dim, heads, mlp_ratio=4.0, dropout=0.0):
      super().__init__()
      self.norm1 = nn.LayerNorm(dim)
      self.attn  = nn.MultiheadAttention(dim, heads,
                        dropout=dropout, batch_first=True)
      self.norm2 = nn.LayerNorm(dim)
      mlp_hidden = int(dim * mlp_ratio)
      self.mlp   = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(dropout)
      )

   def forward(self, x):
      normed = self.norm1(x)
      attn_out, _ = self.attn(normed, normed, normed)
      x = x + attn_out
      x = x + self.mlp(self.norm2(x))
      return x


class MAE_Encoder(nn.Module):
   def __init__(self,
                img_size    = 224,
                patch_size  = 16,
                in_channels = 3,
                embed_dim   = 768,
                depth       = 12,
                num_heads   = 12):
      super().__init__()

      self.patch_size  = patch_size
      self.embed_dim   = embed_dim
      num_patches      = (img_size // patch_size) ** 2

      patch_dim = in_channels * patch_size * patch_size
      self.patch_embed = nn.Linear(patch_dim, embed_dim)

      self.cls_token   = nn.Parameter(torch.zeros(1, 1, embed_dim))

      pos_embed = get_2d_pos_embed(embed_dim, img_size // patch_size)
      pos_embed = torch.from_numpy(pos_embed).float()
      pos_embed = pos_embed.unsqueeze(0)
      self.register_buffer('pos_embed', pos_embed)

      self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads)
            for _ in range(depth)
      ])

      self.norm = nn.LayerNorm(embed_dim)

      nn.init.normal_(self.cls_token, std=0.02)

   def forward(self, x, mask_ratio=0.75):
      patches = make_patches(x, self.patch_size)

      tokens = self.patch_embed(patches)

      tokens = tokens + self.pos_embed

      tokens_vis, mask, ids_restore = random_masking(tokens, mask_ratio)

      cls = self.cls_token.expand(tokens_vis.shape[0], -1, -1)
      tokens_vis = torch.cat([cls, tokens_vis], dim=1)

      for blk in self.blocks:
            tokens_vis = blk(tokens_vis)

      tokens_vis = self.norm(tokens_vis)

      return tokens_vis, mask, ids_restore


class MAE_Decoder(nn.Module):
   def __init__(self,
                num_patches    = 196,
                encoder_dim    = 768,
                decoder_dim    = 384,
                depth          = 12,
                num_heads      = 6,
                patch_size     = 16,
                in_channels    = 3):
      super().__init__()

      self.decoder_dim  = decoder_dim
      self.num_patches  = num_patches
      patch_dim         = in_channels * patch_size * patch_size

      self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim)

      self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))

      self.dec_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_dim)
      )

      self.blocks = nn.ModuleList([
            TransformerBlock(decoder_dim, num_heads)
            for _ in range(depth)
      ])

      self.norm = nn.LayerNorm(decoder_dim)

      self.head = nn.Linear(decoder_dim, patch_dim)

      nn.init.normal_(self.mask_token, std=0.02)
      nn.init.normal_(self.dec_pos_embed, std=0.02)

   def forward(self, enc_tokens, ids_restore):

      x = self.enc_to_dec(enc_tokens)

      x_no_cls = x[:, 1:, :]
      B         = x.shape[0]
      N_vis     = x_no_cls.shape[1]

      num_masked  = self.num_patches - N_vis
      mask_tokens = self.mask_token.expand(B, num_masked, -1)

      combined = torch.cat([x_no_cls, mask_tokens], dim=1)

      ids_exp  = ids_restore.unsqueeze(-1).expand(-1, -1, self.decoder_dim)
      combined = torch.gather(combined, 1, ids_exp)

      cls_dec  = x[:, :1, :]
      combined = torch.cat([cls_dec, combined], dim=1)

      combined = combined + self.dec_pos_embed

      for blk in self.blocks:
            combined = blk(combined)

      combined = self.norm(combined)

      combined = combined[:, 1:, :]

      pred = self.head(combined)
      return pred


class MAE(nn.Module):
   def __init__(self):
      super().__init__()
      self.encoder = MAE_Encoder(
                  img_size   = 224,
                  patch_size = 16,
                  embed_dim  = 768,
                  depth      = 12,
                  num_heads  = 12
      )
      self.decoder = MAE_Decoder(
                  num_patches  = 196,
                  encoder_dim  = 768,
                  decoder_dim  = 384,
                  depth        = 12,
                  num_heads    = 6
      )

   def forward(self, imgs, mask_ratio=0.75):
      enc_out, mask, ids_restore = self.encoder(imgs, mask_ratio)
      pred = self.decoder(enc_out, ids_restore)
      return pred, mask


def unnormalize(tensor):
      mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
      std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
      img = tensor.cpu() * std + mean
      img = img.clamp(0, 1)
      return img


def show_masked_image(original, mask, patch_size=16):
      patches  = make_patches(original.unsqueeze(0), patch_size)
      mask_exp = mask.unsqueeze(0).unsqueeze(-1)
      patches  = patches * (1 - mask_exp)
      masked_img = put_patches_back(patches, patch_size)
      return masked_img.squeeze(0)


MODEL_PATH = "model_mae.pth"

st.set_page_config(page_title="MAE Reconstruction", layout="wide")
st.title("Masked Autoencoder (MAE) - Image Reconstruction")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def load_model():
    model = MAE().to(device)
    state = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


mask_ratio = st.sidebar.slider("Mask Ratio", 0.5, 0.95, 0.75, 0.05)

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    model = load_model()

    img = Image.open(uploaded).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        pred, mask = model(img_tensor, mask_ratio=mask_ratio)

    pred_img = put_patches_back(pred.float(), patch_size=16).cpu()

    orig = unnormalize(img_tensor[0].cpu())
    masked = show_masked_image(img_tensor[0].cpu(), mask[0].cpu())
    recon = pred_img[0].clamp(0, 1)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Masked Input")
        fig1, ax1 = plt.subplots()
        ax1.imshow(masked.permute(1, 2, 0).numpy())
        ax1.axis("off")
        st.pyplot(fig1)
        plt.close(fig1)

    with col2:
        st.subheader("Reconstruction")
        fig2, ax2 = plt.subplots()
        ax2.imshow(recon.permute(1, 2, 0).numpy())
        ax2.axis("off")
        st.pyplot(fig2)
        plt.close(fig2)

    with col3:
        st.subheader("Original")
        fig3, ax3 = plt.subplots()
        ax3.imshow(orig.permute(1, 2, 0).numpy())
        ax3.axis("off")
        st.pyplot(fig3)
        plt.close(fig3)
