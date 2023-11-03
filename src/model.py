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
        )
        self.fc = nn.Linear(self.n_hidden, n_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.linear(out)
        return out
