import torch
import torch.nn as nn
import timm


class HybridPlainDecoderModel(nn.Module):
    def __init__(
        self,
        vit_name: str,
        pretrained: bool,
        cue_vocab_size: int,
        ui_type_vocab_size: int,
        history_len: int,
        ui_geom_dim: int = 4,
        d_model: int = 192,
        nhead: int = 4,
        num_layers: int = 2,
        ff_dim: int = 384,
        dropout: float = 0.1,
        ui_memory_scale: float = 1.0,
        freeze_patch_backbone: bool = False,
    ):
        super().__init__()

        self.history_len = history_len
        self.d_model = d_model
        self.ui_memory_scale = ui_memory_scale
        self.freeze_patch_backbone = freeze_patch_backbone

        # ============================================================
        # 1) Patch memory branch
        # ============================================================
        self.vit = timm.create_model(
            vit_name,
            pretrained=pretrained,
            num_classes=0,
        )

        if self.freeze_patch_backbone:
            for p in self.vit.parameters():
                p.requires_grad = False

        vit_embed_dim = self.vit.patch_embed.proj.out_channels

        self.patch_proj = (
            nn.Identity()
            if vit_embed_dim == d_model
            else nn.Linear(vit_embed_dim, d_model)
        )

        self.patch_memory_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

        patch_mem_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.patch_memory_transformer = nn.TransformerEncoder(
            patch_mem_layer,
            num_layers=1,
        )
        self.patch_memory_norm = nn.LayerNorm(d_model)

        # ============================================================
        # 2) UI memory branch
        # ============================================================
        self.ui_geom_encoder = nn.Sequential(
            nn.Linear(ui_geom_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

        self.ui_type_embedding = nn.Embedding(ui_type_vocab_size, d_model)

        self.ui_memory_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

        ui_mem_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.ui_memory_transformer = nn.TransformerEncoder(
            ui_mem_layer,
            num_layers=1,
        )
        self.ui_memory_norm = nn.LayerNorm(d_model)

        # ============================================================
        # 3) Shared cue token
        # ============================================================
        self.cue_embedding = nn.Embedding(cue_vocab_size, d_model)

        self.memory_gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )

        # ============================================================
        # 4) History encoder
        # ============================================================
        self.history_encoder = nn.Sequential(
            nn.Linear(3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

        self.history_pos_embedding = nn.Embedding(history_len, d_model)

        hist_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.history_transformer = nn.TransformerEncoder(
            hist_layer,
            num_layers=num_layers,
        )
        self.history_norm = nn.LayerNorm(d_model)

        # ============================================================
        # 5) UI-led dual attention
        # ============================================================
        # step 1: query attends UI memory
        self.ui_cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.ui_cross_norm1 = nn.LayerNorm(d_model)
        self.ui_cross_ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.ui_cross_norm2 = nn.LayerNorm(d_model)

        # step 2: UI-shaped query attends patch memory
        self.patch_cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.patch_cross_norm1 = nn.LayerNorm(d_model)
        self.patch_cross_ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.patch_cross_norm2 = nn.LayerNorm(d_model)

        # ============================================================
        # 6) Search state pooling + regression head
        # ============================================================
        self.query_pool = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1),
        )

        self.state_norm = nn.LayerNorm(d_model)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 2),
            nn.Sigmoid(),
        )

    def build_causal_mask(self, T: int, device: torch.device):
        return torch.triu(
            torch.ones(T, T, device=device, dtype=torch.bool),
            diagonal=1,
        )

    def encode_patch_memory(self, image, cue_token):
        """
        image: [B, 3, H, W]
        return: [B, Np, D]
        """
        patch_tokens = self.vit.patch_embed(image)  # [B, N, C]

        if hasattr(self.vit, "pos_embed") and self.vit.pos_embed is not None:
            N = patch_tokens.size(1)
            pos_embed = self.vit.pos_embed[:, 1:1 + N, :]
            if pos_embed.size(-1) == patch_tokens.size(-1):
                patch_tokens = patch_tokens + pos_embed

        patch_tokens = self.patch_proj(patch_tokens)
        patch_tokens = self.patch_memory_mlp(patch_tokens)

        gate = self.memory_gate(cue_token)
        patch_tokens = patch_tokens * gate.unsqueeze(1)
        patch_tokens = patch_tokens + cue_token.unsqueeze(1)

        patch_tokens = self.patch_memory_transformer(patch_tokens)
        patch_tokens = self.patch_memory_norm(patch_tokens)
        return patch_tokens

    def encode_ui_memory(self, ui_geom, ui_type_id, ui_mask, cue_token):
        """
        ui_geom:    [B, Ku, 4]
        ui_type_id: [B, Ku]
        ui_mask:    [B, Ku], 1=valid, 0=pad
        return:
          ui_tokens: [B, Ku, D]
          ui_key_padding_mask: [B, Ku] True = pad
        """
        geom_emb = self.ui_geom_encoder(ui_geom)
        type_emb = self.ui_type_embedding(ui_type_id)

        ui_tokens = geom_emb + type_emb
        ui_tokens = self.ui_memory_mlp(ui_tokens)

        gate = self.memory_gate(cue_token)
        ui_tokens = ui_tokens * gate.unsqueeze(1)
        ui_tokens = ui_tokens + cue_token.unsqueeze(1)

        ui_key_padding_mask = (ui_mask == 0)

        ui_tokens = self.ui_memory_transformer(
            ui_tokens,
            src_key_padding_mask=ui_key_padding_mask,
        )
        ui_tokens = self.ui_memory_norm(ui_tokens)
        return ui_tokens, ui_key_padding_mask

    def encode_history(self, history_xydur):
        """
        history_xydur: [B, T, 3]
        """
        B, T, _ = history_xydur.shape
        device = history_xydur.device

        hist_tokens = self.history_encoder(history_xydur)

        pos_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        hist_tokens = hist_tokens + self.history_pos_embedding(pos_ids)

        causal_mask = self.build_causal_mask(T, device)
        hist_out = self.history_transformer(hist_tokens, mask=causal_mask)
        hist_out = self.history_norm(hist_out)
        return hist_out

    def pool_sequence(self, x):
        attn_logits = self.query_pool(x)
        attn = torch.softmax(attn_logits, dim=1)
        pooled = torch.sum(attn * x, dim=1)
        return pooled

    def forward(self, image, cue_id, history_xydur, ui_geom, ui_type_id, ui_mask):
        """
        image:         [B, 3, H, W]
        cue_id:        [B]
        history_xydur: [B, T, 3]
        ui_geom:       [B, Ku, 4]
        ui_type_id:    [B, Ku]
        ui_mask:       [B, Ku]
        """
        # A) cue token
        cue_token = self.cue_embedding(cue_id)  # [B, D]

        # B) memories
        patch_memory = self.encode_patch_memory(image, cue_token)  # [B, Np, D]
        ui_memory, ui_key_padding_mask = self.encode_ui_memory(
            ui_geom=ui_geom,
            ui_type_id=ui_type_id,
            ui_mask=ui_mask,
            cue_token=cue_token,
        )  # [B, Ku, D]

        # scale UI memory
        ui_memory = self.ui_memory_scale * ui_memory

        # C) history query
        hist_out = self.encode_history(history_xydur)  # [B, T, D]
        query_seq = hist_out + cue_token.unsqueeze(1)  # [B, T, D]

        # ============================================================
        # Step 1: UI-led attention
        # ============================================================
        ui_attended, _ = self.ui_cross_attn(
            query=query_seq,
            key=ui_memory,
            value=ui_memory,
            key_padding_mask=ui_key_padding_mask,
            need_weights=False,
        )
        x = self.ui_cross_norm1(query_seq + ui_attended)
        x = self.ui_cross_norm2(x + self.ui_cross_ffn(x))

        # ============================================================
        # Step 2: Patch refinement
        # ============================================================
        patch_attended, _ = self.patch_cross_attn(
            query=x,
            key=patch_memory,
            value=patch_memory,
            need_weights=False,
        )
        x = self.patch_cross_norm1(x + patch_attended)
        x = self.patch_cross_norm2(x + self.patch_cross_ffn(x))

        # D) pooled search state
        search_state = self.pool_sequence(x)
        search_state = self.state_norm(search_state)

        # E) next fixation regression
        pred_xy = self.head(search_state)
        return pred_xy