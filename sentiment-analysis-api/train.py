"""
Train the sentiment model and save artifacts for the API.
Run this instead of the notebook to generate models/sentiment_model.pt and models/vocab.json.

Usage:
    cd sentiment-analysis-api
    python train.py
"""

import os
import re
import json
import string
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Text preprocessing (same as notebook)
# ---------------------------------------------------------------------------

def normalize(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text

def tokenize(text):
    text = normalize(text)
    return text.split(" ")

def build_vocab(texts, min_freq=5):
    word_counter = Counter()
    for text in texts:
        tokens = tokenize(text)
        word_counter.update(tokens)

    vocab = {}
    vocab["<PAD>"] = 0
    vocab["<UNK>"] = 1
    index = 2

    for word, count in word_counter.items():
        if count >= min_freq:
            vocab[word] = index
            index += 1
    return vocab

def encode(text, vocab):
    tokens = tokenize(text)
    return [vocab.get(token, vocab["<UNK>"]) for token in tokens]

def pad_sequence(sequence, max_length, pad_index):
    if len(sequence) > max_length:
        return sequence[:max_length]
    return sequence + [pad_index] * (max_length - len(sequence))

def create_mask(sequence, pad_index):
    return [1 if token != pad_index else 0 for token in sequence]

def preprocess_sentence(text, vocab, max_length):
    encoded = encode(text, vocab)
    padded = pad_sequence(encoded, max_length, vocab["<PAD>"])
    mask = create_mask(padded, vocab["<PAD>"])
    return padded, mask

def preprocess_dataset(texts, labels, vocab, max_length):
    all_tokens, all_masks, all_labels = [], [], []
    for text, label in zip(texts, labels):
        tokens, mask = preprocess_sentence(text, vocab, max_length)
        all_tokens.append(tokens)
        all_masks.append(mask)
        all_labels.append(label)
    return all_tokens, all_masks, all_labels


# ---------------------------------------------------------------------------
# Dataset class (same as notebook)
# ---------------------------------------------------------------------------

class SentimentDataset(Dataset):
    def __init__(self, tokens, masks, labels):
        self.tokens = tokens
        self.masks = masks
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            "tokens": self.tokens[index],
            "mask": self.masks[index],
            "label": self.labels[index],
        }


# ---------------------------------------------------------------------------
# Model (same as notebook)
# ---------------------------------------------------------------------------

class SentimentModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, tokens, mask):
        embedded = self.embedding(tokens)
        lstm_output, _ = self.lstm(embedded)
        sequence_lengths = mask.sum(dim=1) - 1
        sequence_lengths = torch.clamp(sequence_lengths, min=0)
        batch_indices = torch.arange(lstm_output.size(0), device=lstm_output.device)
        last_outputs = lstm_output[batch_indices, sequence_lengths]
        logits = self.fc(last_outputs)
        return logits


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main():
    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load data
    data_path = os.path.join("..", "data", "imdb_data.csv")
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    df["sentiment_num"] = df["sentiment"].map(lambda x: 1 if x == "positive" else 0)

    # Split
    text_temp, _, label_temp, _ = train_test_split(
        df["review"], df["sentiment_num"], test_size=0.2, random_state=42, stratify=df["sentiment_num"]
    )
    text_train, text_val, label_train, label_val = train_test_split(
        text_temp, label_temp, test_size=0.1, random_state=42
    )
    print(f"Train: {len(text_train)}, Val: {len(text_val)}")

    # Build vocab
    print("Building vocabulary...")
    vocab = build_vocab(text_train, min_freq=5)
    print(f"Vocabulary size: {len(vocab)}")

    # Preprocess
    MAX_LENGTH = 200
    print("Preprocessing datasets...")

    train_tokens, train_masks, train_labels = preprocess_dataset(text_train, label_train, vocab, MAX_LENGTH)
    val_tokens, val_masks, val_labels = preprocess_dataset(text_val, label_val, vocab, MAX_LENGTH)

    train_tokens = torch.tensor(train_tokens, dtype=torch.long)
    train_masks = torch.tensor(train_masks, dtype=torch.long)
    train_labels = torch.tensor(list(train_labels), dtype=torch.long)

    val_tokens = torch.tensor(val_tokens, dtype=torch.long)
    val_masks = torch.tensor(val_masks, dtype=torch.long)
    val_labels = torch.tensor(list(val_labels), dtype=torch.long)

    train_dataset = SentimentDataset(train_tokens, train_masks, train_labels)
    val_dataset = SentimentDataset(val_tokens, val_masks, val_labels)

    BATCH_SIZE = 64
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    # Model
    model = SentimentModel(vocab_size=len(vocab), embedding_dim=100, hidden_dim=128).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train
    EPOCHS = 10  # 10 epochs is enough to get ~86% accuracy (notebook used 100 but overfit)
    print(f"\nTraining for {EPOCHS} epochs...")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            tokens = batch["tokens"].to(device)
            mask = batch["mask"].to(device)
            labels = batch["label"].float().to(device)

            optimizer.zero_grad()
            outputs = model(tokens, mask).squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                tokens = batch["tokens"].to(device)
                mask = batch["mask"].to(device)
                labels = batch["label"].to(device)

                outputs = model(tokens, mask).squeeze(1)
                val_loss += criterion(outputs, labels.float()).item()

                predictions = (torch.sigmoid(outputs) > 0.5).long()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        val_loss /= len(val_loader)
        accuracy = correct / total

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {accuracy:.4f}")

    # Save
    save_dir = "models"
    os.makedirs(save_dir, exist_ok=True)

    model_path = os.path.join(save_dir, "sentiment_model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")

    vocab_path = os.path.join(save_dir, "vocab.json")
    with open(vocab_path, "w") as f:
        json.dump(vocab, f)
    print(f"Vocabulary saved to {vocab_path} ({len(vocab)} words)")

    print("\nDone! You can now start the API with:")
    print("  uvicorn backend.app:app --reload")


if __name__ == "__main__":
    main()
