import torch
import torch.nn as nn


class BiLSTM(nn.Module):
    def __init__(
        self,
        embeddings,
        n_classes,
        n_layers=1,
        n_hidden=50,
        dropout=0,
    ) -> None:
        super().__init__()
        self.embeddings = nn.Embedding.from_pretrained(embeddings, freeze=True)
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lstm = nn.LSTM(
            self.embeddings.weight.shape[1],
            hidden_size=n_hidden,
            num_layers=n_layers,
            dropout=dropout,
            bidirectional=True,
            batch_first=True,
        )
        self.fc = nn.Linear(2 * self.n_hidden, n_classes)

    def last_timestep(self, unpacked, lengths):
        # Index of the last output for each sequence.
        idx = (
            (lengths - 1)
            .view(-1, 1)
            .expand(unpacked.size(0), unpacked.size(2))
            .unsqueeze(1)
        )
        return unpacked.gather(1, idx).squeeze()

    def forward(self, x, x_lengths):
        embeddings = self.embeddings(x)
        x_pack = nn.utils.rnn.pack_padded_sequence(
            embeddings, x_lengths, batch_first=True
        )

        out_pack, _ = self.lstm(x_pack)
        out_unpack, out_lengths = nn.utils.rnn.pad_packed_sequence(
            out_pack, batch_first=True
        )
        out = self.last_timestep(out_unpack, out_lengths)
        out = self.fc(out)
        return out
