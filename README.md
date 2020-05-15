# (re)MUDE
### (re)Implementation of [Learning Multi-level Dependencies for Robust Word Recognition](https://arxiv.org/pdf/1911.09789.pdf)


# Summary
The original paper introduce  a  robust  word  recognition framework  that  captures multi-level sequential dependencies in noised sentence. Practical application of such framework addresses to a challenging task of [Grammatical Error Correction](https://en.wikipedia.org/wiki/Grammar_checker) and [improving robustness of modern NLP setups](https://arxiv.org/abs/1905.11268).

Model architecture:
<p align='center'><img width='1024' src='https://i.ibb.co/2jK20VW/mude-arc.png'/></p>

# Why
Despite a clearly written paper the released [original code](https://github.com/zw-s-github/MUDE) lacks of structure, guidance and reproducibility. There are also critical bugs found by community members in original implementation. Here's my attempt to organize bits and pieces in intuitive way.

# Details
It's a fully [Pytorch](https://github.com/pytorch/) based implementation, using `torch.nn.TransformerEncoder`, `torch.nn.data.Dataset`, `torch.nn.data.Dataloader` and blazing [Ignite](https://github.com/pytorch/ignite) for personal and your convinience.

# Data
Authors said:
```
Lastly, as this work primarily focuses on English, it would be very meaningful to experiment the proposed framework on other languages.
```
So I took it seriously and has trained/evaluated experimental runs on low size corpus of [**russian** news texts](https://github.com/natasha/corus). Preprocessed respective train, valid and test splits with vocabulary placed in `data`.

# Train, evaluation
All hyperparameter values in those runs are copied from original code. Experiment runs evaluated wrt to noise type in terms of Word Recognition Accuracy on test split.

Result table are shown below. 
|PER   |DEL   |INS   |SUB   |W-PER |W-DEL |W-INS |W-SUB |
|-----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
|0.998 |0.976 |0.987 |0.974 |0.998 |0.956 |0.987 |0.965 |


Checkpoints are not included in repo because of them size.
It's not that bad, you can train your own copy of MUDE easily. Also, I'm a not a big fan of huge number of arguments in usual `train.py` script, so their number overwhelmly small here, sorry for that.
```bash
python3 train_noise-type -n "NOISE-TYPE"
```

*PER checkpoint in action*:

[original article](https://www.rbc.ru/business/14/05/2020/5ebc1efa9a79471be026dc51)
> На всчрете с Птуиным Сичен сакзал, что в 2008 гдоу стотмосиь нфети в рбулях составялла поркдяа 1100 руб., сечйас — 1200 руб., при эотм траиф "Транснефти" на прочакку нтфеи в это же вермя ворыс с 822 до 2,1 тыс. руб. за тнноу на 100 км. Совеинтк перзидента "Трафсненти" Иогрь Димен овтетил на заявлеине гнавлого исполнитеньлого директроа "Росенфти" Иогря Счеина, который на вртсече с пзеридентом Воадимирлм Питуным во виорнтк, 12 мая, попсорил птчои вовде снзиить тафиры трубопрдвооной мпнооолии, поскокьлу, по его расаетчм, рхсаоды на тррнспоат счйеас сосватляют 32% от смоитости нтфеи. В резутьтале расхдоы на транспорт счйеас соютавляст 32% от стоитосми нтфеи, а это "чувствитлеьно", заюлкчил гвала "Росфенти".

...

> На встрече с Путиным Сечин сказал, что в _NUM_ году стоимость нефти в рублях составляла порядка _NUM_ руб., сейчас — _NUM_ руб., при этом тариф "Транснефти" на ***парковку*** нефти в это же время вырос с _NUM_ до _NUM_, _NUM_ тыс. руб. за тонну на _NUM_ км. Советник президента "Транснефти" Игорь ***Днем*** ответил на заявление главного исполнительного директора "Роснефти" Игоря Сечина, который на встрече с президентом Владимиром Путиным во вторник, _NUM_ мая, попросил почти вдвое снизить тарифы ***противовоздушной*** монополии, поскольку, по его расчетам, расходы на транспорт сейчас составляют _NUM_% от стоимости нефти. В результате расходы на транспорт сейчас составляют _NUM_% от стоимости нефти, а это "***чувствовал***", заключил глава "Роснефти".

to reproduce (having **PER** checkpoint)
```bash
python3 correction-example.py
```

# Project structure
```
data/ # contains the dataset and vocab
src/  # contains the MUDE, dataset and vectorizer

train.py                # train model to solve for selected noise type
control.py              # visualize predictions on test set
correction-example.py   # to reproduce example above
```

# Notes
1. No sliding windows. Variable length input with padding, packaging with `pack_padded_sequence` for top recurrent unit processing;
2. `c0` token used to compute representation of characters sequence here - is a separate symbol; 
3. All runs has fixed β (contribution of seq2seq loss) and does not change its value during training as opposed to original idea of gradually reducing it;
4. `SUB` (subtraction) type of noise implemented as replacing with randomly selected char from adjacent ones given QWERTY layout.
5. Most of the word prediction errors occurs because of the vocabulary size problem. Such predictions usually has low score. Are there any chances to build such system with BPE/subword-unit vocab compression?