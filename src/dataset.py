import logging

import gensim.downloader
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

logger = logging.Logger(__name__)


class TRECDataset(Dataset):
    def __init__(self, data_path: str, labels_path: str, w2v) -> None:
        data = np.load(data_path, allow_pickle=True)
        self.labels = np.load(labels_path, allow_pickle=True)
        self.max_len = 0
        self.tokens = []
        for text in tqdm(data, desc="Converting tokens to index"):
            text_tokens = []
            for word in text.split(" "):
                try:
                    text_tokens.append(w2v.key_to_index[word])
                except KeyError:
                    pass
            self.tokens.append(text_tokens)
            self.max_len = max(self.max_len, len(text_tokens))

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, index):
        return self.tokens[index], self.labels[index]


if __name__ == "__main__":
    ds = TRECDataset(
        "../data/x_train.npy",
        "../data/y_train.npy",
        gensim.downloader.load("word2vec-google-news-300"),
    )
    print(ds[0])
