import logging
import os

import gensim.downloader
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import TRECDataset
from model import BiLSTM
from pad_sequence import PadSequence

logger = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="eval", version_base="1.2")
def main(cfg: DictConfig):
    missing_keys: set[str] = OmegaConf.missing_keys(cfg)
    if missing_keys:
        raise RuntimeError(f"Got missing keys in config:\n{missing_keys}")
    w2v = gensim.downloader.load("word2vec-google-news-300")
    cwd = hydra.utils.get_original_cwd()
    test_ds = TRECDataset(
        os.path.join(cwd, cfg.data.test_data),
        os.path.join(cwd, cfg.data.test_labels),
        w2v,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=PadSequence(),
    )

    saved = torch.load(cfg.model_path)
    model = BiLSTM(**saved["model_params"])
    model.load_state_dict(saved["state_dict"])

    model.eval()
    corrects = 0
    for (inputs, input_lengths), labels in tqdm(
        test_loader, desc="Evaluating model", unit="batch"
    ):
        out = model(inputs, input_lengths)
        preds = out.argmax(dim=1, keepdim=True).squeeze()
        corrects += (preds == labels).sum().item()

    logger.info(
        f"Test set accuracy: {corrects/len(test_loader.dataset)*100:5f}"
    )


if __name__ == "__main__":
    main()
