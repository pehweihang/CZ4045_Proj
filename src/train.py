import gensim.downloader
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm

from dataset import TRECDataset
from model import BiLSTM
from pad_sequence import PadSequence


def main():
    w2v = gensim.downloader.load("word2vec-google-news-300")
    train_ds = TRECDataset("../data/x_train.npy", "../data/y_train.npy", w2v)
    dev_ds = TRECDataset("../data/x_dev.npy", "../data/y_dev.npy", w2v)
    test_ds = TRECDataset("../data/x_test.npy", "../data/y_test.npy", w2v)
    torch.manual_seed(420)

    train_loader = DataLoader(
        train_ds, batch_size=32, shuffle=True, collate_fn=PadSequence()
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=32, shuffle=True, collate_fn=PadSequence()
    )
    test_loader = DataLoader(
        test_ds, batch_size=32, shuffle=True, collate_fn=PadSequence()
    )

    model = BiLSTM(w2v.vectors, n_classes=5)

    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=1e4)

    epochs = 5

    for epoch in range(epochs):
        print(f"Epoch {epoch}")
        for inputs, labels in tqdm(train_loader):
            optim.zero_grad()
            out = model(inputs)
            loss = criterion(out, labels)
            loss.backward()
            optim.step()



if __name__ == "__main__":
    main()
