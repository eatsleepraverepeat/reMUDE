import random
import warnings
import numpy as np
import regex as re
from itertools import permutations
from utils.qwerty_layout import qwerty_coords


class Vectorizer:

    def __init__(self, max_chars, noise_type, att_len=4):
        """
        Vectorizer takes token, applies specified type of transformation to it and encode as a sequence of symbols.
        max_chars (int): maximum output sequence length (padded to right)
        noise_type (str): type of noise to apply
        att_len (int): minimum token length to apply noise transformation
        """

        self.noise_type = noise_type
        self.whole = True if re.match(r'^W-', self.noise_type) else False
        self.att_len = att_len

        self.qwerty_chars = [*qwerty_coords]
        self.points = np.array([*qwerty_coords.values()])

        self.mxch = max_chars  # for c0 token
        self.symbols = ',."-():%?/!;[]#*'
        self.alphabet = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ'
        self.special_chars = '^@_<>#~'  # <SEQ-REPR> <PAD_CHAR> <NUM> <BOS> <EOS> <UNK> <PAD_TOKEN>
        self.chars = self.alphabet + self.symbols + self.special_chars

        self.SEQ_REPR_INDEX = self.chars.index('^')  # c0 char index
        self.PAD_CHAR_INDX = self.chars.index('@')

        self.NUM_TOKEN = '<NUM>'
        self.BOS_TOKEN = '<BOS>'
        self.EOS_TOKEN = '<EOS>'
        self.UNK_TOKEN = '<UNK>'
        self.PAD_TOKEN = '<PAD>'

        self.SP2INDEX = {
            self.NUM_TOKEN: self.chars.index('_'),
            self.BOS_TOKEN: self.chars.index('<'),
            self.EOS_TOKEN: self.chars.index('>'),
            self.UNK_TOKEN: self.chars.index('#'),
            self.PAD_TOKEN: self.chars.index('~')
        }
        self.INDEX2SP = {v: k for k, v in self.SP2INDEX.items()}

    def char_distance(self, char):
        dists = np.sqrt(np.power(qwerty_coords[char] - self.points, 2).sum(axis=1))
        dists[dists == 0] = 100
        return dists

    def permutation(self, w):
        chars = list(w)

        if not self.whole:
            perms = [list(reversed(c)) for c in permutations(range(1, len(chars) - 1), 2)]
        else:
            perms = [list(reversed(c)) for c in permutations(range(0, len(chars)), 2)]

        perm = random.choice(perms)
        fr = chars[perm[0]]
        sc = chars[perm[1]]
        chars[perm[0]] = sc
        chars[perm[1]] = fr

        return chars

    def deletion(self, w):
        chars = list(w)

        if not self.whole:
            dele = random.choice(range(1, len(chars) - 1))
        else:
            dele = random.choice(range(0, len(chars)))

        del chars[dele]
        return chars

    def insertion(self, w):
        chars = list(w)

        if not self.whole:
            inser = random.choice(range(1, len(chars) - 1))
        else:
            inser = random.choice(range(0, len(chars)))

        chars.insert(inser, random.choice(self.alphabet))
        return chars

    def subtraction(self, w):
        chars = list(w)

        if not self.whole:
            sb = random.choice(range(1, len(chars) - 1))
        else:
            sb = random.choice(range(0, len(chars)))

        c = chars[sb]
        dists = self.char_distance(c.lower())
        closest_chars = np.where(dists == dists.min())[0]
        r_adjacent = np.random.choice([self.qwerty_chars[it] for it in closest_chars])
        chars[sb] = r_adjacent
        return chars

    def encode(self, w):
        """DOC+"""

        bin_all = [self.PAD_CHAR_INDX] * self.mxch

        mask_all = [1] * self.mxch
        mask_all[0] = 0

        bin_all[0] = self.SEQ_REPR_INDEX

        if w in self.SP2INDEX:
            bin_all[1] = self.SP2INDEX[w]
            mask_all[1] = 0

        else:
            if self.noise_type == "PASS" or len(w) < self.att_len:
                wchars = list(w)

            elif self.noise_type in ["PER", "W-PER"]:
                wchars = self.permutation(w)

            elif self.noise_type in ["DEL", "W-DEL"]:
                wchars = self.deletion(w)

            elif self.noise_type in ["INS", "W-INS"]:
                wchars = self.insertion(w)

            elif self.noise_type in ["SUB", "W-SUB"]:
                wchars = self.subtraction(w)

            else:
                warnings.warn("CHECK THE AVAILABLE METHODS")
                raise

            for idx, wchar in enumerate(wchars):
                try:
                    bin_all[idx + 1] = self.chars.index(wchar)  # first entry is c0, for the encoder representation
                except ValueError:
                    print(wchar)
                    raise
                mask_all[idx + 1] = 0

        return bin_all, mask_all

    def decode(self, bins):
        s = ''.join([self.chars[b] for b in bins if b not in [self.SEQ_REPR_INDEX, self.PAD_CHAR_INDX]])
        if len(s) == 1 and s in list(self.special_chars):
            return self.INDEX2SP[self.chars.index(s)]
        else:
            return s

    def decode_as_is(self, bins):
        """For debugging proposes"""
        return ' '.join([self.chars[b] for b in bins])


if __name__ == '__main__':

    # types
    TYPES = [
        "PASS",   # PASS WITHOUT MODIFICATIONS
        "PER",    # PERMUTATION
        "DEL",    # DELETION
        "INS",    # INSERTION
        "SUB",    # SUBSTRACTION
        "W-PER",  # WHOLE WORD PERMUTATION
        "W-DEL",  # WHOLD WORD DELETION
        "W-INS",  # WHOLE WORD INSERTION
        "W-SUB"   # WHOLE WORD SUBSTRACTION
    ]

    MAX_CHAR_LEN = 24

    # PASS
    TYPE = 'PASS'
    vect = Vectorizer(MAX_CHAR_LEN, TYPE)
    words = [
        'Играючи',
        'Мальчишка',
        ';',
        '"',
        ".",
        "!",
        vect.NUM_TOKEN,
        vect.BOS_TOKEN,
        vect.EOS_TOKEN,
        vect.UNK_TOKEN,
        vect.PAD_TOKEN
    ]

    for word in words:
        wbins, wmask = vect.encode(word)
        decoded = vect.decode(wbins)
        assert word == decoded

    # CHECKS
    TYPE = 'PER'
    vect = Vectorizer(MAX_CHAR_LEN, TYPE)
    word = 'Играючи'
    wbins, wmask = vect.encode(word)
    vect.decode_as_is(wbins)

    # PER
    TYPE = 'PER'
    vect = Vectorizer(MAX_CHAR_LEN, TYPE)
    assert not vect.whole
    word = 'Играючи'
    wbins, wmask = vect.encode(word)
    decoded = vect.decode(wbins)
    assert not set(word).difference(decoded)

    # W-PER
    TYPE = 'W-PER'
    vect = Vectorizer(MAX_CHAR_LEN, TYPE)
    assert vect.whole
    word = 'Играючи'
    wbins, wmask = vect.encode(word)
    decoded = vect.decode(wbins)
    assert not set(word).difference(decoded)

    # DEL
    TYPE = 'DEL'
    vect = Vectorizer(MAX_CHAR_LEN, TYPE)
    assert not vect.whole
    word = 'Играючи'
    wbins, wmask = vect.encode(word)
    decoded = vect.decode(wbins)

    # W-DEL
    TYPE = 'W-DEL'
    vect = Vectorizer(MAX_CHAR_LEN, TYPE)
    assert vect.whole
    word = 'Играючи'
    wbins, wmask = vect.encode(word)
    decoded = vect.decode(wbins)

    # INS
    TYPE = 'INS'
    vect = Vectorizer(MAX_CHAR_LEN, TYPE)
    assert not vect.whole
    word = 'Играючи'
    wbins, wmask = vect.encode(word)
    decoded = vect.decode(wbins)

    # W-INS
    TYPE = 'W-INS'
    vect = Vectorizer(MAX_CHAR_LEN, TYPE)
    assert vect.whole
    word = 'Играючи'
    wbins, wmask = vect.encode(word)
    decoded = vect.decode(wbins)

    # SUB
    TYPE = 'SUB'
    vect = Vectorizer(MAX_CHAR_LEN, TYPE)
    assert not vect.whole
    word = 'Играючи'
    wbins, wmask = vect.encode(word)
    decoded = vect.decode(wbins)

    # W-SUB
    TYPE = 'W-SUB'
    vect = Vectorizer(MAX_CHAR_LEN, TYPE)
    assert vect.whole
    word = 'Играючи'
    wbins, wmask = vect.encode(word)
    decoded = vect.decode(wbins)
