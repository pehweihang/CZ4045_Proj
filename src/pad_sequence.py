import torch


class PadSequence:
    def __call__(self, batch):
        sorted_batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)
        sequences = [x[0] for x in sorted_batch]
        sequences_padded = torch.nn.utils.rnn.pad_sequence(
            sequences, batch_first=True
        )

        lengths = torch.LongTensor([len(x) for x in sequences])
        labels = torch.LongTensor(map(lambda x: x[1], sorted_batch))

        return sequences_padded, lengths, labels
