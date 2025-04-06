
import torch
import torch.nn as nn
import torch.nn.functional as F

class SentimentTransformer(nn.Module):
    def __init__(self, vocab_size, emb_dim=256, num_heads=4, hidden_dim=512, num_layers=4, num_classes=3, max_len=128):
        super(SentimentTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = nn.Embedding(max_len + 1, emb_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=0.4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(emb_dim, num_classes)

    def forward(self, input_ids, attention_mask=None):
        B, T = input_ids.shape
        x = self.embedding(input_ids)  # (B, T, D)
        cls_tokens = self.cls_token.expand(B, 1, -1)  # (B, 1, D)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, T+1, D)

        pos = torch.arange(0, T + 1, device=input_ids.device).unsqueeze(0)
        x = x + self.pos_embedding(pos)

        if attention_mask is not None:
            cls_mask = torch.ones((B, 1), dtype=torch.long, device=attention_mask.device)
            attention_mask = torch.cat((cls_mask, attention_mask), dim=1)
            key_padding_mask = attention_mask == 0
        else:
            key_padding_mask = None

        x = self.transformer(x, src_key_padding_mask=key_padding_mask)
        x = self.norm(x[:, 0, :])  # Use [CLS] token output
        x = self.dropout(x)
        logits = self.fc(x)
        return logits
