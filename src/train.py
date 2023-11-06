import logging
import os

import gensim.downloader
import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import TRECDataset
from model import BiLSTM
from pad_sequence import PadSequence

logger = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    w2v = gensim.downloader.load("word2vec-google-news-300")
    cwd = hydra.utils.get_original_cwd()
    train_ds = TRECDataset(
        os.path.join(cwd, cfg.data.train_data),
        os.path.join(cwd, cfg.data.train_labels),
        w2v,
    )
    dev_ds = TRECDataset(
        os.path.join(cwd, cfg.data.dev_data),
        os.path.join(cwd, cfg.data.dev_labels),
        w2v,
    )
    test_ds = TRECDataset(
        os.path.join(cwd, cfg.data.test_data),
        os.path.join(cwd, cfg.data.test_labels),
        w2v,
    )
    torch.manual_seed(420)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=PadSequence(),
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=PadSequence(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=PadSequence(),
    )

    model = BiLSTM(torch.from_numpy(w2v.vectors), **cfg.model)

    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    for epoch in range(cfg.epochs):
        model.train()
        train_loss = 0
        train_corrects = 0
        dev_loss = 0
        dev_corrects = 0
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
                correct = (preds == labels).sum().item()
                accuracy = correct / cfg.batch_size

                train_loss += loss.item() * inputs.size(0)
                train_corrects += correct
                tepoch.set_postfix(
                    train_loss=loss.item(), train_accuracy=100.0 * accuracy
                )
            model.eval()
            for (inputs, input_lengths), labels in dev_loader:
                out = model(inputs, input_lengths)
                loss = criterion(out, labels)
                dev_loss += loss.item() * inputs.size(0)
                preds = out.argmax(dim=1, keepdim=True).squeeze()
                dev_corrects += (preds == labels).sum().item()
            logger.info(
                "train_loss: {:.5f} train_acc: {:.5f} dev_loss: {:.5f} dev_accuracy: {:.5f}".format(
                    train_loss / len(train_loader.dataset),
                    train_corrects / len(train_loader.dataset),
                    dev_loss / len(dev_loader.dataset),
                    dev_corrects / len(dev_loader.dataset),
                )
            )


if __name__ == "__main__":
    main()
