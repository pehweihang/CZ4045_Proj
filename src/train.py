import gensim.downloader
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
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

    model = BiLSTM(torch.from_numpy(w2v.vectors), n_classes=6)

    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)

    epochs = 100

    for epoch in range(epochs):
        model.train()
        loss = 0
        accuracy = 0
        with tqdm(
            train_loader, desc=f"Epoch {epoch+1}", unit="batch"
        ) as tepoch:
            for (inputs, input_lengths), labels in tepoch:
                optim.zero_grad()
                out = model(inputs, input_lengths)
                loss = criterion(out, labels)
                loss.backward()
                optim.step()

                preds = out.argmax(dim=1, keepdim=True).squeeze()
                accuracy = (preds == labels).sum().item() / 32
                tepoch.set_postfix(
                    train_loss=loss.item(), train_accuracy=100.0 * accuracy
                )
            model.eval()
            running_loss = 0
            running_correct_preds = 0
            for (inputs, input_lengths), labels in dev_loader:
                out = model(inputs, input_lengths)
                running_loss += criterion(out, labels).item()
                preds = out.argmax(dim=1, keepdim=True).squeeze()
                running_correct_preds += (preds == labels).sum().item()
            print(
                "dev_loss={} dev_accuracy={}".format(
                    running_loss / len(dev_loader),
                    running_correct_preds / len(dev_ds),
                )
            )


if __name__ == "__main__":
    main()
