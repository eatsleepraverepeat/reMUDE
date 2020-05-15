import torch
import regex as re
from pathlib import Path
from razdel import sentenize

from control import best_checkpoint_selector
from train import TASK_DATA_TEST_PATH, TASK_DATA_VOCAB_PATH
from src.model import MUDE
from src.dataset import pad_char_sequence, pad_mask_sequence, SPCInMemDataset
from utils.text import text_normalizer, m_analyzer


EOS_TOKEN = '<EOS>'
BOS_TOKEN = '<BOS>'
NUM_TOKEN = '<NUM>'
UNK_TOKEN = '<UNK>'


def re_tokenizer(x):
    return RPATTERN.findall(x)


def isnum(x):
    if x.isdigit() or ROMAN_NUMERALS.match(x):
        return NUM_TOKEN
    else:
        return x


def islatin(x):
    if x not in [BOS_TOKEN, EOS_TOKEN, NUM_TOKEN, UNK_TOKEN]:
        return 'LATN' in m_analyzer.parse(x)[0].tag
    else:
        return False


def process_record(rec):
    sentences = []

    text = text_normalizer(rec)
    for sentence in sentenize(text):
        txt = sentence.text

        tokens = re_tokenizer(txt)
        tokens.insert(0, BOS_TOKEN)
        tokens.append(EOS_TOKEN)
        tokens = [isnum(token) for token in tokens]

        # single lang sentences and minimum length of the sentence is 3 tokens, ignoring SPECIAL
        latin_block = any([islatin(token) for token in tokens])

        if not latin_block:
            line = ' '.join(tokens)
            sentences.append(line)

    return sentences


RPATTERN = re.compile(r'[а-яa-zё\d]+|\p{Punct}', re.UNICODE | re.MULTILINE | re.DOTALL | re.IGNORECASE)
ROMAN_NUMERALS = re.compile('(^(?=[MDCLXVI])M*(C[MD]|D?C{0,3})(X[CL]|L?X{0,3})(I[XV]|V?I{0,3})$)')


if __name__ == '__main__':

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = best_checkpoint_selector(Path('checkpoints/MUDE_PER'))

    MAX_CHARS = 24
    NOISE_TYPE = "PASS"

    dataset = SPCInMemDataset(
        data_path=TASK_DATA_TEST_PATH,
        vocab_path=TASK_DATA_VOCAB_PATH,
        max_chars=MAX_CHARS,
        noise_type=NOISE_TYPE
    )

    CHAR_VOCAB_SIZE = len(dataset.vect.chars)
    TGT_VOCAB_SIZE = len(dataset.vocab)

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

    inp = 'На всчрете с Птуиным Сичен сакзал, что в 2008 гдоу стотмосиь нфети в рбулях составялла поркдяа 1100 руб., ' \
          'сечйас — 1200 руб., при эотм траиф "Транснефти" на прочакку нтфеи ' \
          'в это же вермя ворыс с 822 до 2,1 тыс. руб. за тнноу на 100 км. ' \
          'Совеинтк перзидента "Трафсненти" Иогрь Димен овтетил на заявлеине гнавлого исполнитеньлого ' \
          'директроа "Росенфти" Иогря Счеина, ' \
          'который на вртсече с пзеридентом Воадимирлм Питуным во виорнтк, 12 мая, ' \
          'попсорил птчои вовде снзиить тафиры трубопрдвооной мпнооолии, поскокьлу, ' \
          'по его расаетчм, рхсаоды на тррнспоат счйеас сосватляют 32% от смоитости нтфеи. ' \
          'В резутьтале расхдоы на транспорт счйеас соютавляст 32% от стоитосми нтфеи, ' \
          'а это "чувствитлеьно", заюлкчил гвала "Росфенти".'

    sentences = process_record(inp)
    lens = [len(sentence.split(' ')) for sentence in sentences]
    max_len = max(lens)

    X, m = [], []
    for sentence in sentences:

        pack = [dataset.vect.encode(t) for t in sentence.split(' ')]
        encoded = torch.Tensor([p[0] for p in pack]).long()
        masks = torch.Tensor([p[1] for p in pack]).bool()

        padding_len = max_len-len(encoded)
        encoded = pad_char_sequence(encoded, padding_len, dataset)
        masks = pad_mask_sequence(masks, padding_len, dataset)

        X.append(encoded.unsqueeze(0))
        m.append(masks.unsqueeze(0))

    X = torch.cat(X).long().to(device=device)
    m = torch.cat(m).bool().to(device=device)
    lens = torch.Tensor(lens).long().to(device=device)

    X = X[torch.argsort(-lens)]
    m = m[torch.argsort(-lens)]
    lens = lens[torch.argsort(-lens)]

    mude.eval()
    with torch.no_grad():
        _, y_pred = mude(X, m, lens)

    y_pred = y_pred.argmax(dim=-1)
    restored = ""
    for idx, y in enumerate(y_pred):
        restored += ' '.join(
            [
                dataset.rev_vocab[int(t)]
                for t in y[:lens[idx]]
                if dataset.rev_vocab[int(t)] not in [
                    dataset.vect.PAD_TOKEN, dataset.vect.EOS_TOKEN, dataset.vect.BOS_TOKEN
                ]
            ]
        ) + ' '

    print(restored)
