import regex as re
import unidecode
import ftfy
import pymorphy2


m_analyzer = pymorphy2.MorphAnalyzer()


def text_normalizer(text_sentence):

    # drop slashes and newline symbol
    text_sentence = re.sub(r'\\r\\n$|\\t+|\\', '', text_sentence)

    #
    text_sentence = ftfy.fix_text(text_sentence)

    # replace bytes symbols
    text_sentence = re.sub('xa0', ' ', text_sentence)

    # replace repetitab and newline symbols
    text_sentence = re.sub('\n+|\t+', ' ', text_sentence)

    # quotes normalizer
    text_sentence = re.sub(r"[«»‘“„”']+", '"', text_sentence)
    # unicode normalizer (@ character level)
    if re.compile(r'(?i)[^-a-zа-яё\d".,]').search(text_sentence) is not None:
        for m in re.compile(r'(?i)[^-a-zа-яё\d".,]').finditer(text_sentence):
            text_sentence = text_sentence.replace(m.group(), unidecode.unidecode(m.group()))

    # :) pattern avoiding unicode characters
    check_characters = {
        'ı': 'i',
        'İ': 'I'
    }
    for chk, repl in check_characters.items():
        if re.search(chk, text_sentence):
            text_sentence = re.sub(chk, repl, text_sentence)

    # No concatenated before
    w_no_before_pattern = re.compile(r'(?i)[а-яa-zё]+No\.')
    if w_no_before_pattern.search(text_sentence):
        text_sentence = re.sub(r'(?i)([а-яa-zё]+)(No\.)', r'\1 \2', text_sentence)

    # concatenated number after
    w_num_pattern = re.compile(r'(?i)[а-яa-zё]+[\d]+')
    if w_num_pattern.search(text_sentence):
        text_sentence = re.sub(r'(?i)([a-zа-яё]+)([\d]+)', r'\1 \2', text_sentence)

    # concatenated number before
    w_num_pattern = re.compile(r'(?i)[\d]+[а-яa-zё]+')
    if w_num_pattern.search(text_sentence):
        text_sentence = re.sub(r'(?i)([\d]+)([а-яa-zё]+)', r'\1 \2', text_sentence)

    # concatenate w\s-w
    dash_first = re.compile(r'(?i)[а-яa-zё]+\s+-[a-zа-яё]+')
    if dash_first.search(text_sentence):
        text_sentence = re.sub(r'(?i)([а-яa-zё]+)\s+-([a-zа-яё]+)', r'\1-\2', text_sentence)

    # concatenate w-\sw
    dash_first = re.compile(r'(?i)[а-яa-zё]+-\s+[a-zа-яё]+')
    if dash_first.search(text_sentence):
        text_sentence = re.sub(r'(?i)([а-яa-zё]+)-\s+([a-zа-яё]+)', r'\1-\2', text_sentence)

    # divide w-d
    wd_pattern = re.compile(r'(?i)[a-zа-яё]+-\d+')
    if wd_pattern.search(text_sentence):
        text_sentence = re.sub(r'(?i)([a-zа-яё]+)-(\d+)', r'\1 - \2', text_sentence)

    # divide d-w
    wd_pattern = re.compile(r'(?i)\d+-[a-zа-яё]+')
    if wd_pattern.search(text_sentence):
        text_sentence = re.sub(r'(?i)(\d+)-([a-zа-яё]+)', r'\1 - \2', text_sentence)

    # divide d-d
    wd_pattern = re.compile(r'(?i)\d+-\d+')
    if wd_pattern.search(text_sentence):
        text_sentence = re.sub(r'(?i)(\d+)-(\d+)', r'\1 - \2', text_sentence)

    # divide -d
    wd_pattern = re.compile(r'(?i)-\d+')
    if wd_pattern.search(text_sentence):
        text_sentence = re.sub(r'(?i)-(\d+)', r'- \1', text_sentence)

    # divide d-
    wd_pattern = re.compile(r'(?i)\d+-')
    if wd_pattern.search(text_sentence):
        text_sentence = re.sub(r'(?i)(\d+)-', r'\1 -', text_sentence)

    # punctuation repetition
    text_sentence = re.sub(r',{2,}', ',', text_sentence)
    text_sentence = re.sub(r'"{2,}', '"', text_sentence)
    text_sentence = re.sub(r'\.{2,}', '.', text_sentence)

    # remove list like notations
    text_sentence = re.sub(r'^"[\d]+\.', '"', text_sentence)
    text_sentence = re.sub(r'^[\d]+\.', '', text_sentence)

    # replace zero-as-char things
    text_sentence = re.sub(r'(?i)([a-zа-яё])0([a-zа-яё])', r'\1o\2', text_sentence)

    return text_sentence
