import torch
import torch.nn as nn
import timm


class PatchTransformerModel(nn.Module):
    def __init__(
        self,
        vit_name: str,
        pretrained: bool,
        history_len: int,
        cue_vocab_size: int,
        d_model: int = 192,
        nhead: int = 3,
        num_layers: int = 2,
        ff_dim: int = 384,
        dropout: float = 0.1,
        cue_dim: int = 192,
    ):
        super().__init__()

        self.history_len = history_len
        self.d_model = d_model

        # backbone: use ViT patch embed only
        self.vit = timm.create_model(
            vit_name,
            pretrained=pretrained,
            num_classes=0
        )

        # history token encoder: each fixation [x,y,dur] -> token dim
        self.history_encoder = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        self.cue_embedding = nn.Embedding(cue_vocab_size, cue_dim)

        # positional embeddings for [cue + history + patches]
        # 1 cue + H history + ~196 patches
        self.max_tokens = 1 + history_len + 256
        self.pos_embedding = nn.Embedding(self.max_tokens, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.norm = nn.LayerNorm(d_model)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 2),
            nn.Sigmoid()
        )

    def forward(self, image, history, cue_id):
        """
        image: [B,3,H,W]
        history: [B,H_len,3]
        cue_id: [B]
        """

        B = image.size(0)

        # ---- patch tokens from timm ViT ----
        # patch embedding
        x = self.vit.patch_embed(image)    # [B, N, D]
        N = x.size(1)

        # add original ViT positional embedding for patches
        if hasattr(self.vit, "pos_embed") and self.vit.pos_embed is not None:
            # vit.pos_embed usually includes cls token at pos 0
            patch_pos = self.vit.pos_embed[:, 1:1 + N, :]
            x = x + patch_pos

        patch_tokens = x  # [B, N, D]

        # ---- history tokens ----
        history_tokens = self.history_encoder(history)   # [B, H_len, D]

        # ---- cue token ----
        cue_token = self.cue_embedding(cue_id).unsqueeze(1)  # [B,1,D]

        # ---- concat all tokens ----
        tokens = torch.cat([cue_token, history_tokens, patch_tokens], dim=1)  # [B, 1+H+N, D]
        T = tokens.size(1)

        pos_ids = torch.arange(T, device=tokens.device).unsqueeze(0).expand(B, T)
        tokens = tokens + self.pos_embedding(pos_ids)

        # ---- transformer fusion ----
        out = self.transformer(tokens)
        out = self.norm(out)

        # use cue token output as fused summary
        fused = out[:, 0, :]   # [B,D]

        pred_xy = self.head(fused)
        return pred_xy