import torch


class PadSequence:
    def __call__(self, batch):
        sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
        sequences, labels = zip(*sorted_batch)
        sequences_padded = torch.nn.utils.rnn.pad_sequence(
            sequences, batch_first=True
        )

        lengths = torch.LongTensor([len(x) for x in sequences])
        labels = torch.LongTensor(labels)

        return (sequences_padded, lengths), labels
