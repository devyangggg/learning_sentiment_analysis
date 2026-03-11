import torch
import torch.nn as nn


class SentimentModel(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, tokens, mask):
        embedded = self.embedding(tokens)

        lstm_output, _ = self.lstm(embedded)

        sequence_lengths = mask.sum(dim=1) - 1
        sequence_lengths = torch.clamp(sequence_lengths, min=0)

        batch_indices = torch.arange(
            lstm_output.size(0),
            device=lstm_output.device
        )

        last_outputs = lstm_output[batch_indices, sequence_lengths]

        logits = self.fc(last_outputs)
        return logits


def load_model(model_path, vocab_size, embedding_dim=100, hidden_dim=128):

    model = SentimentModel(vocab_size, embedding_dim, hidden_dim)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model
