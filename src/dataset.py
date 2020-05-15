import json
import torch
import tqdm
from multiprocessing import cpu_count
import numpy as np
from pathlib import Path
from src.vectorizer import Vectorizer
from torch.utils.data import Dataset, DataLoader


def pad_char_sequence(seq, ln, dtset):
    padding = [dtset.vect.encode(pad_symb)[0] for pad_symb in [dtset.vect.PAD_TOKEN] * ln]
    return torch.cat([seq, torch.Tensor(padding).long()])


def pad_mask_sequence(msk, ln, dtset):
    padding = [dtset.vect.encode(pad_symb)[1] for pad_symb in [dtset.vect.PAD_TOKEN] * ln]
    return torch.cat([msk, torch.Tensor(padding).bool()])


def pad_token_sequence(seq, ln, dtset):
    padding = [dtset.vocab[pad_symb] for pad_symb in [dtset.vect.PAD_TOKEN] * ln]
    return torch.cat([seq, torch.Tensor(padding).long()])


def custom_collate_fn(batch, max_chars, dtset):
    bsz = len(batch)

    batch_chars = []
    batch_masks = []
    batch_tokens = []
    batch_seq_lengths = []

    for elem in batch:
        # break

        chars, masks, tokens = elem
        seq_len = len(tokens)

        torch_chars = torch.cat([torch.Tensor(ch).long() for ch in chars]).view(seq_len, max_chars)
        torch_masks = torch.cat([torch.Tensor(m).bool() for m in masks]).view(seq_len, max_chars)
        torch_tokens = torch.Tensor(tokens).long()

        batch_chars.append(torch_chars)
        batch_masks.append(torch_masks)
        batch_tokens.append(torch_tokens)
        batch_seq_lengths.append(seq_len)

    batch_seq_lengths = np.array(batch_seq_lengths)
    batch_max_seq_len = batch_seq_lengths.max()

    padded_batch_chars = []
    padded_batch_masks = []
    padded_batch_tokens = []

    for idx, bc in enumerate(batch_chars):

        if len(bc) < batch_max_seq_len:
            padding_len = batch_max_seq_len - len(bc)

            padded_chars = pad_char_sequence(bc, padding_len, dtset)
            padded_masks = pad_mask_sequence(batch_masks[idx], padding_len, dtset)
            padded_tokens = pad_token_sequence(batch_tokens[idx], padding_len, dtset)

            padded_batch_chars.append(padded_chars)
            padded_batch_masks.append(padded_masks)
            padded_batch_tokens.append(padded_tokens)

        else:
            padded_batch_chars.append(bc)
            padded_batch_masks.append(batch_masks[idx])
            padded_batch_tokens.append(batch_tokens[idx])

    X = torch.cat(padded_batch_chars).view(bsz, batch_max_seq_len, max_chars)  # chars
    m = torch.cat(padded_batch_masks).view(bsz, batch_max_seq_len, max_chars)  # masks
    y = torch.cat(padded_batch_tokens).view(bsz, batch_max_seq_len)  # target tokens
    lenghts = torch.Tensor(batch_seq_lengths).int()  # lenghts of sequences for packing

    # sorting the data in decreasing order for further use in pack_padded_sequence and reverse
    srt_idx = (-lenghts).argsort()
    X = X[srt_idx]
    m = m[srt_idx]
    y = y[srt_idx]
    lenghts = lenghts[srt_idx]

    return X, m, y, lenghts


class SPCInMemDataset(Dataset):

    def __init__(self, data_path, vocab_path, max_chars, noise_type):
        super(SPCInMemDataset, self).__init__()

        # load data in memory
        self.data = [ln.strip() for ln in data_path.open(mode='r', encoding='utf-8')]
        self.vocab = json.load(vocab_path.open(mode='r', encoding='utf-8'))
        self.rev_vocab = {v: k for k, v in self.vocab.items()}
        self.vect = Vectorizer(max_chars=max_chars, noise_type=noise_type)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.process_line(self.data[item])

    def process_line(self, ln):

        seq, masks, tokensIdx = [], [], []
        for t in ln.strip().split(' '):

            s, m = self.vect.encode(t)
            seq.append(s)
            masks.append(m)
            tokensIdx.append(self.vocab[t])

        seq = np.array(seq).astype(np.int)
        masks = np.array(masks).astype(np.bool)
        tokensIdx = np.array(tokensIdx).astype(np.int)
        return seq, masks, tokensIdx


if __name__ == '__main__':

    TASK_DATA_TRAIN_PATH = Path('../data/task-data-noner-train.txt')
    TASK_DATA_VAL_PATH = Path('../data/task-data-noner-val.txt')
    TASK_DATA_TEST_PATH = Path('../data/task-data-noner-test.txt')
    TASK_DATA_VOCAB_PATH = Path('../data/task-data-vocab.json')

    TYPE = "W-SUB"
    MAX_CHARS = 24
    ve = Vectorizer(max_chars=MAX_CHARS, noise_type=TYPE)

    # vectorization check
    fi = TASK_DATA_TRAIN_PATH.open(mode='r', encoding='utf-8', newline='\n')
    for line in tqdm.tqdm(fi):
        [ve.encode(t) for t in line.strip().split(' ')]

    fi = TASK_DATA_VAL_PATH.open(mode='r', encoding='utf-8', newline='\n')
    for line in tqdm.tqdm(fi):
        [ve.encode(t) for t in line.strip().split(' ')]

    fi = TASK_DATA_TEST_PATH.open(mode='r', encoding='utf-8', newline='\n')
    for line in tqdm.tqdm(fi):
        [ve.encode(t) for t in line.strip().split(' ')]

    TYPE = "PER"
    MAX_CHARS = 24
    BATCH_SIZE = 16
    NUM_WORKERS = cpu_count()

    # TRAIN
    dataset_train_split = SPCInMemDataset(
        data_path=TASK_DATA_TRAIN_PATH,
        vocab_path=TASK_DATA_VOCAB_PATH,
        max_chars=MAX_CHARS,
        noise_type=TYPE
    )

    train_loader = DataLoader(
        dataset=dataset_train_split,
        batch_size=BATCH_SIZE,
        collate_fn=lambda x: custom_collate_fn(x, max_chars=MAX_CHARS, dtset=dataset_train_split),
        num_workers=NUM_WORKERS
    )

    for btch in tqdm.tqdm(train_loader, desc='Reading train split data'):
        bX, bm, by, blenghts = btch
        assert bX.shape == bm.shape

    # VAL
    dataset_val_split = SPCInMemDataset(
        data_path=TASK_DATA_VAL_PATH,
        vocab_path=TASK_DATA_VOCAB_PATH,
        max_chars=MAX_CHARS,
        noise_type=TYPE
    )

    val_loader = DataLoader(
        dataset=dataset_val_split,
        batch_size=BATCH_SIZE,
        collate_fn=lambda x: custom_collate_fn(x, max_chars=MAX_CHARS, dtset=dataset_val_split),
        num_workers=NUM_WORKERS
    )

    for batch in tqdm.tqdm(val_loader, desc='Reading val split data'):
        bX, bm, by, blenghts = btch
        assert bX.shape == bm.shape

    # TEST
    dataset_test_split = SPCInMemDataset(
        data_path=TASK_DATA_TEST_PATH,
        vocab_path=TASK_DATA_VOCAB_PATH,
        max_chars=MAX_CHARS,
        noise_type=TYPE
    )

    test_loader = DataLoader(
        dataset=dataset_test_split,
        batch_size=BATCH_SIZE,
        collate_fn=lambda x: custom_collate_fn(x, max_chars=MAX_CHARS, dtset=dataset_test_split),
        num_workers=NUM_WORKERS
    )

    for btch in tqdm.tqdm(test_loader, desc='Reading test split data'):
        bX, bm, by, blenghts = btch
        assert bX.shape == bm.shape
