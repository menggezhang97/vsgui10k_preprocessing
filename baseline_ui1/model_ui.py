import torch
import torch.nn as nn


class UIBaselineModel(nn.Module):
    def __init__(
        self,
        ui_vocab_size: int,
        cue_vocab_size: int,
        history_len: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        ff_dim: int = 256,
        dropout: float = 0.1,
        xy_feat_dim: int = 32,
    ):
        super().__init__()

        self.history_len = history_len
        self.d_model = d_model

        self.ui_embedding = nn.Embedding(ui_vocab_size, d_model)
        self.cue_embedding = nn.Embedding(cue_vocab_size, d_model)

        self.xy_encoder = nn.Sequential(
            nn.Linear(3, xy_feat_dim),
            nn.ReLU(),
            nn.Linear(xy_feat_dim, d_model),
        )

        self.pos_embedding = nn.Embedding(history_len + 1, d_model)

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
        self.head = nn.Linear(d_model, ui_vocab_size)

    def forward(self, cue_id, history_ui_ids, history_xydur):
        """
        cue_id: [B]
        history_ui_ids: [B, H]
        history_xydur: [B, H, 3]
        """
        bsz = history_ui_ids.size(0)
        H = history_ui_ids.size(1)

        ui_feat = self.ui_embedding(history_ui_ids)          # [B, H, D]
        xy_feat = self.xy_encoder(history_xydur)             # [B, H, D]
        hist_feat = ui_feat + xy_feat

        cue_feat = self.cue_embedding(cue_id).unsqueeze(1)   # [B, 1, D]

        x = torch.cat([cue_feat, hist_feat], dim=1)          # [B, 1+H, D]

        pos_ids = torch.arange(0, H + 1, device=x.device).unsqueeze(0).expand(bsz, H + 1)
        x = x + self.pos_embedding(pos_ids)

        x = self.transformer(x)
        x = self.norm(x)

        # use the last history token representation
        final_feat = x[:, -1, :]                             # [B, D]
        logits = self.head(final_feat)                       # [B, ui_vocab_size]
        return logits