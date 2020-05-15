import torch
import random
import regex as re
import numpy as np
from pathlib import Path
from colorama import Fore, Style
from train import MUDE, get_data_loaders


# PATHS
CHECKPOINTS_PATH = Path('checkpoints')


def pred_pretty_printer(inp):
    n, t, p = inp

    n = n[1:-1]
    t = t[1:-1]
    p = p[1:-1]

    colored_p = []
    for idx, token in enumerate(p):
        if t[idx] == token:
            s = f"{Style.BRIGHT}{Fore.GREEN}{token}{Style.RESET_ALL}"
        else:
            s = f"{Style.BRIGHT}{Fore.RED}{token}{Style.RESET_ALL}"
        colored_p.append(s)

    print(
        f"NOISED: {' '.join(n)}\n"
        f"TARGET: {' '.join(t)}\n"
        f"PRED: {' '.join(colored_p)}"
    )


def best_checkpoint_selector(directory):
    pattern = re.compile(r'__(\d+)_WRA')
    chkpnts = [f for f in directory.iterdir() if f.name.endswith('pth.tar')]
    leidx = np.argmax([int(pattern.search(chk.name).group(1)) for chk in chkpnts])
    return chkpnts[int(leidx)]


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    TYPES = [
        # "PASS",  # PASS WITHOUT MODIFICATIONS
        "PER",    # PERMUTATION
        "DEL",    # DELETION
        "INS",    # INSERTION
        "SUB",    # SUBTRACTION
        "W-PER",  # WHOLE WORD PERMUTATION
        "W-DEL",  # WHOLE WORD DELETION
        "W-INS",  # WHOLE WORD INSERTION
        "W-SUB"   # WHOLE WORD SUBTRACTION
    ]

    MAX_CHARS = 24
    BSZ = 8
    NT = random.choice(TYPES)
    ntype_chckp = CHECKPOINTS_PATH.joinpath(f"MUDE_{NT}")
    checkpoint_path = best_checkpoint_selector(ntype_chckp)

    test_ld = get_data_loaders(NT, BSZ, MAX_CHARS)[2]
    CHAR_VOCAB_SIZE = len(test_ld.dataset.vect.chars)
    TGT_VOCAB_SIZE = len(test_ld.dataset.vocab)

    DIM = 512
    DIM_FFT = int(DIM * 4)
    ATTN_HEADS = 8
    DEPTH = 2
    DIM_HIDDEN = 650
    DROPOUT_RATE = .01
    LR = 1e-4

    mude = MUDE(
        dim=DIM,
        characters_vocab_size=CHAR_VOCAB_SIZE,
        tokens_vocab_size=TGT_VOCAB_SIZE,
        encoder_depth=DEPTH,
        encoder_attn_heads=ATTN_HEADS,
        encoder_dimff=DIM_FFT,
        encoder_dropout=DROPOUT_RATE,
        top_rec_hidden_dim=DIM_HIDDEN,
        top_rec_proj_dropout=DROPOUT_RATE
    )
    mude = mude.to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    mude.load_state_dict(state_dict)

    X, m, y, lens = next(iter(test_ld))
    X = X.to(device)
    y = y.to(device)
    m = m.to(device)
    lens = lens.to(device)

    mude.eval()
    with torch.no_grad():
        _, y_pred = mude(X, m, lens)
    y_pred = y_pred.argmax(dim=-1)

    ridx = random.choice(range(BSZ))
    noised = [test_ld.dataset.vect.decode(t) for t in X[ridx]][0:lens[ridx]]
    tgt = [test_ld.dataset.rev_vocab[int(t)] for t in y[ridx]][0:lens[ridx]]
    pred = [test_ld.dataset.rev_vocab[int(t)] for t in y_pred[ridx]][0:lens[ridx]]

    pred_pretty_printer((noised, tgt, pred))
