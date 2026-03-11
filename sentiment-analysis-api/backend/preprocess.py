
import re
import json
import string
import torch



def normalize(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def tokenize(text):

    text = normalize(text)
    return text.split(" ")


def load_vocab(path):
    """Load vocabulary dict from a JSON file."""
    with open(path, "r") as f:
        vocab = json.load(f)
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

def preprocess_text(text, vocab, max_length=200):

    encoded = encode(text, vocab)
    padded = pad_sequence(encoded, max_length, vocab["<PAD>"])
    mask = create_mask(padded, vocab["<PAD>"])

    tokens_tensor = torch.tensor([padded], dtype=torch.long)
    mask_tensor = torch.tensor([mask], dtype=torch.long)

    return tokens_tensor, mask_tensor
