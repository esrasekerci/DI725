
import torch
from torch.utils.data import Dataset
from transformers import GPT2TokenizerFast

class ConversationDataset(Dataset):
    def __init__(self, dataframe, tokenizer_path="gpt2", max_len=128):
        self.texts = dataframe["conversation"].astype(str).tolist()
        self.labels = dataframe["label"].tolist()
        self.tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return (
            enc["input_ids"].squeeze(0),
            enc["attention_mask"].squeeze(0),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )
