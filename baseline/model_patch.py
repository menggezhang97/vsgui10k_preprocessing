import torch
import torch.nn as nn
import timm


class PatchBaselineModel(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        pretrained: bool,
        history_len: int,
        cue_vocab_size: int,
        dropout: float = 0.1,
        cue_dim: int = 64,
        history_hidden_dim: int = 128,
        fusion_hidden_dim: int = 256,
    ):
        super().__init__()

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0
        )

        backbone_dim = self.backbone.num_features

        self.history_encoder = nn.Sequential(
            nn.Flatten(),  # [B, H, 3] -> [B, H*3]
            nn.Linear(history_len * 3, history_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(history_hidden_dim, history_hidden_dim),
            nn.ReLU(),
        )

        self.cue_embedding = nn.Embedding(cue_vocab_size, cue_dim)

        self.head = nn.Sequential(
            nn.Linear(backbone_dim + history_hidden_dim + cue_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, 2),
            nn.Sigmoid()
        )

    def forward(self, image, history, cue_id):
        img_feat = self.backbone(image)          # [B, backbone_dim]
        hist_feat = self.history_encoder(history)  # [B, history_hidden_dim]
        cue_feat = self.cue_embedding(cue_id)    # [B, cue_dim]

        feat = torch.cat([img_feat, hist_feat, cue_feat], dim=1)
        pred_xy = self.head(feat)
        return pred_xy