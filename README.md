# CZ4045 Project Question 2
## Requirements
Install required python packages
```bash
conda env create -f environment.yml
conda activate nlp-proj
```
Or
```bash
pip install -r requirements.txt
```
## Usage
Download and prepare the dataset
```bash
python src/create_dataset.py
```

Train and tune model:
```bash
python src/train.py -m \
    hydra.sweeper.n_trials=<NUMBER OF TUNING TRIALS> \
    epochs=<NUMBER OF EPOCHS>
    lr=<LEARNING RATE>
    early_stop_patience=<EARLY STOP PATIENCE>
    model.n_layers=<NUMBER OF LSTM LAYERS>
    model.n_hidden=<LSTM HIDDEN SIZE>
    model.dropout=<LSTM DROPOUT>
    model.bidirectional=<true IF BIDIRECTIONAL LSTM>
```

evaluate model:
```bash
python src/evaluate.py model_path=<PATH TO SAVED MODEL>
```
