
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import wandb
import random
import numpy as np
from model import SentimentTransformer
from dataset import ConversationDataset
from transformers import GPT2TokenizerFast
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score

# Reproducibility
SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

# Config
EPOCHS = 15
BATCH_SIZE = 16
BLOCK_SIZE = 128
LEARNING_RATE = 3e-4

wandb.init(project="di725-sentiment-transformer-restore")

# Load tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Load dataset
train_df = pd.read_csv("data/train_cleaned.csv")
val_df = pd.read_csv("data/val_cleaned.csv")

train_dataset = ConversationDataset(train_df, max_len=BLOCK_SIZE)
val_dataset = ConversationDataset(val_df, max_len=BLOCK_SIZE)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Compute and adjust class weights
labels = train_df['label'].values
weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
weights[0] *= 1.3
weights[2] *= 1.6
class_weights = torch.tensor(weights, dtype=torch.float)

# Load model with stable size
model = SentimentTransformer(
    vocab_size=tokenizer.vocab_size,
    emb_dim=256,
    num_heads=4,
    hidden_dim=512,
    num_layers=4,
    num_classes=3,
    max_len=BLOCK_SIZE
)

# Resize embedding if needed
model.embedding = nn.Embedding(tokenizer.vocab_size, 256)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))  # no label smoothing

# Early stopping settings
best_f1 = 0
patience = 4
patience_counter = 0

# Training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for input_ids, attn_mask, labels in train_loader:
        input_ids, attn_mask, labels = input_ids.to(device), attn_mask.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask=attn_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()

    avg_train_loss = total_loss / len(train_loader)

    # Evaluation
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for input_ids, attn_mask, labels in val_loader:
            input_ids, attn_mask, labels = input_ids.to(device), attn_mask.to(device), labels.to(device)
            logits = model(input_ids, attention_mask=attn_mask)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    val_acc = correct / total
    val_f1 = f1_score(all_labels, all_preds, average='macro')

    wandb.log({"epoch": epoch + 1, "train_loss": avg_train_loss, "val_accuracy": val_acc, "val_f1": val_f1})
    print(f"Epoch {epoch+1}: Loss = {avg_train_loss:.4f}, Val Acc = {val_acc:.4f}, F1 = {val_f1:.4f}")

    if val_f1 > best_f1:
        best_f1 = val_f1
        patience_counter = 0
        torch.save(model.state_dict(), "sentiment_transformer_best.pt")
        print("✅ Model saved (new best F1)")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("⏹️ Early stopping triggered.")
            break
