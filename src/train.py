import copy
import logging
import os

import gensim.downloader
import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import TRECDataset
from model import LSTM
from pad_sequence import PadSequence

logger = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    print(os.getcwd())
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

    torch.manual_seed(420)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device {device}")

    embedding = np.array(w2v.vectors)
    model = LSTM(embedding.shape[0], embedding.shape[1], **cfg.model)
    model.set_embedding(torch.from_numpy(embedding))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    best_dev_acc = 0
    best_dev_acc_epoch = 0
    best_model_state_dict = None
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
                inputs, input_lengths, labels = (
                    inputs.to(device),
                    input_lengths.to(device),
                    labels.to(device),
                )
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
            inputs, input_lengths, labels = (
                inputs.to(device),
                input_lengths.to(device),
                labels.to(device),
            )
            out = model(inputs, input_lengths)
            loss = criterion(out, labels)
            dev_loss += loss.item() * inputs.size(0)
            preds = out.argmax(dim=1, keepdim=True).squeeze()
            dev_corrects += (preds == labels).sum().item()
        logger.info(
            "Epoch {} train_loss: {:.5f} train_acc: {:.5f} dev_loss: {:.5f} dev_accuracy: {:.5f}".format(
                epoch + 1,
                train_loss / len(train_loader.dataset),
                train_corrects / len(train_loader.dataset) * 100,
                dev_loss / len(dev_loader.dataset),
                dev_corrects / len(dev_loader.dataset) * 100,
            )
        )
        if dev_corrects / len(dev_loader.dataset) > best_dev_acc:
            best_model_state_dict = {
                k: v.cpu() for k, v in model.state_dict().items()
            }
            best_dev_acc = dev_corrects / len(dev_loader.dataset)
            best_dev_acc_epoch = epoch
        if epoch - best_dev_acc_epoch > cfg.early_stop_patience:
            break

    logger.info(
        f"Saving best model at epoch {best_dev_acc_epoch+1} with accuracy {best_dev_acc*100:.5f}"
    )
    torch.save(
        {
            "state_dict": best_model_state_dict,
            "model_params": {
                "n_embeddings": model.n_embeddings,
                "embedding_dim": model.embedding_dim,
                "n_classes": model.n_classes,
                "n_layers": model.n_layers,
                "n_hidden": model.n_hidden,
                "dropout": model.dropout,
            },
        },
        os.path.join(os.getcwd(), "best_model.pt"),
    )
    return best_dev_acc


if __name__ == "__main__":
    main()
