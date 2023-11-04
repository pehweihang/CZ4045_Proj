import os
import logging

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


@hydra.main(config_path="config", config_name="config")
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
                accuracy = (preds == labels).sum().item() / cfg.batch_size
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
            logger.info(
                "dev_loss={} dev_accuracy={}".format(
                    running_loss / len(dev_loader),
                    running_correct_preds / len(dev_ds),
                )
            )


if __name__ == "__main__":
    main()
