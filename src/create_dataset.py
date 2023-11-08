import argparse
import os

import numpy as np
from datasets import DatasetDict, load_dataset
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/",
        required=False,
        help="directory to save data files to",
    )
    return parser.parse_args()


def main(args):
    os.makedirs(args.data_dir, exist_ok=True)
    dataset: DatasetDict = load_dataset("trec")
    x = np.array(dataset["train"]["text"], dtype=object)
    y = np.array(dataset["train"]["coarse_label"], dtype=np.int8)

    x_test = np.array(dataset["test"]["text"], dtype=object)
    y_test = np.array(dataset["test"]["coarse_label"], dtype=np.int8)

    # combine 0 class with 6 class
    y[y == 6] = 0
    y_test[y_test == 6] = 0

    # random split
    x_train, x_dev, y_train, y_dev = train_test_split(
        x, y, test_size=500, stratify=y
    )

    np.save(os.path.join(args.data_dir, "x_train"), x_train)
    np.save(os.path.join(args.data_dir, "x_dev"), x_dev)
    np.save(os.path.join(args.data_dir, "x_test"), x_test)
    np.save(os.path.join(args.data_dir, "y_train"), y_train)
    np.save(os.path.join(args.data_dir, "y_dev"), y_dev)
    np.save(os.path.join(args.data_dir, "y_test"), y_test)


if __name__ == "__main__":
    args = parse_args()
    main(args)
