import warnings
import argparse
from pathlib import Path
from src.model import MUDE
from utils.metrics import WordRecognitionAccuracy

import torch
from torch.nn import NLLLoss
from torch.optim.rmsprop import RMSprop
from torch.utils.data import DataLoader
from multiprocessing import cpu_count

from ignite.contrib.metrics import GpuInfo
from ignite.engine import Engine, Events
from ignite.handlers import global_step_from_engine, ModelCheckpoint, TerminateOnNan
from ignite.contrib.handlers import ProgressBar
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger
from ignite.contrib.handlers.tensorboard_logger import OutputHandler
from ignite.contrib.handlers.tensorboard_logger import OptimizerParamsHandler
from ignite.contrib.handlers.tensorboard_logger import WeightsHistHandler
from ignite.contrib.handlers.param_scheduler import CosineAnnealingScheduler

from src.dataset import SPCInMemDataset, custom_collate_fn


BASIC_PATH = Path(__file__).parent
TASK_DATA_TRAIN_PATH = BASIC_PATH.joinpath('data/task-data-noner-train.txt')
TASK_DATA_VAL_PATH = BASIC_PATH.joinpath('data/task-data-noner-val.txt')
TASK_DATA_TEST_PATH = BASIC_PATH.joinpath('data/task-data-noner-test.txt')
TASK_DATA_VOCAB_PATH = BASIC_PATH.joinpath('data/task-data-vocab.json')


def get_data_loaders(noise_type, batch_size, max_chars):
    NUM_WORKERS = cpu_count()

    dataset_train_split = SPCInMemDataset(
        data_path=TASK_DATA_TRAIN_PATH,
        vocab_path=TASK_DATA_VOCAB_PATH,
        max_chars=max_chars,
        noise_type=noise_type
    )

    dataset_val_split = SPCInMemDataset(
        data_path=TASK_DATA_VAL_PATH,
        vocab_path=TASK_DATA_VOCAB_PATH,
        max_chars=max_chars,
        noise_type=noise_type
    )

    dataset_test_split = SPCInMemDataset(
        data_path=TASK_DATA_TEST_PATH,
        vocab_path=TASK_DATA_VOCAB_PATH,
        max_chars=max_chars,
        noise_type=noise_type
    )

    train_loader = DataLoader(
        dataset_train_split,
        batch_size=batch_size,
        collate_fn=lambda x: custom_collate_fn(x, max_chars=max_chars, dtset=dataset_train_split),
        num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        dataset_val_split,
        batch_size=batch_size,
        collate_fn=lambda x: custom_collate_fn(x, max_chars=max_chars, dtset=dataset_val_split),
        num_workers=NUM_WORKERS
    )
    test_loader = DataLoader(
        dataset_test_split,
        batch_size=batch_size,
        collate_fn=lambda x: custom_collate_fn(x, max_chars=max_chars, dtset=dataset_test_split),
        num_workers=NUM_WORKERS
    )

    return train_loader, val_loader, test_loader


def prepare_batch(batch):
    X, m, y, lengths = batch

    X = X.to(device)
    m = m.to(device)
    y = y.to(device)
    lengths = lengths.to(device)

    return X, m, y, lengths


def create_trainer(model,
                   optimizer,
                   seq2seq_loss_fn,
                   pred_loss_fn,
                   char_vocab_size,
                   tgt_vocab_size):

    def _update(engine, batch):
        X, m, y, lengths = prepare_batch(batch)

        model.train()
        optimizer.zero_grad()
        yseq_pred, y_pred = model(X, m, lengths)

        s2s_loss = seq2seq_loss_fn(yseq_pred.view(-1, char_vocab_size), X[:, :, 1:].contiguous().view(-1))
        recog_loss = pred_loss_fn(y_pred.view(-1, tgt_vocab_size), y.view(-1))

        final_loss = recog_loss + BETA*s2s_loss

        final_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), .8)
        optimizer.step()

        return {
            "loss": final_loss.item(),
            "loss_pred": recog_loss.item(),
            "loss_seq2seq": s2s_loss.item()
        }

    return Engine(_update)


def create_evaluator(model, metrics):
    metrics = metrics or {}

    def _inference(engine, batch):

        model.eval()
        with torch.no_grad():
            X, m, y, lengths = prepare_batch(batch)
            _, y_pred = model(X, m, lengths)
        return y_pred, y, lengths

    engine = Engine(_inference)
    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


if __name__ == '__main__':

    TYPES = [
        "PASS",   # PASS WITHOUT MODIFICATIONS
        "PER",    # PERMUTATION
        "DEL",    # DELETION
        "INS",    # INSERTION
        "SUB",    # SUBTRACTION
        "W-PER",  # WHOLE WORD PERMUTATION
        "W-DEL",  # WHOLE WORD DELETION
        "W-INS",  # WHOLE WORD INSERTION
        "W-SUB"   # WHOLE WORD SUBTRACTION
    ]

    # BASIC PATH
    TENSOBOARD_LOGS_DIR_PATH = Path('tensorboard_logs').absolute()
    CHECKPOINTS_DIR_PATH = Path('checkpoints').absolute()
    EVALUATION_RESULTS_FILE_PATH = Path('data/evaluation_res/report').absolute()

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', '-e', default=24, type=int, help='number of epochs to learn')
    parser.add_argument('--noise_type', '-n', type=str, default="PER", help='type of noise')
    parser.add_argument('--beta', type=float, default=1.0, help='contribution of seq2seq loss')
    parser.add_argument('--gpu', type=int, default=0, help='which GPU device to use')
    args = parser.parse_args()

    if args.noise_type not in TYPES:
        warnings.warn(f"{args.noise_type} TYPE OF NOISING IS NOT IMPLEMENTED")
        raise NotImplementedError

    if torch.cuda.is_available():
        device = "cuda"
        if torch.cuda.device_count() > 1:
            device = f"cuda:{args.gpu}"
    else:
        device = "cpu"

    EPOCHS = args.epochs
    NOISE_TYPE = args.noise_type
    BETA = args.beta

    DIM = 512
    DIM_FFT = int(DIM * 4)
    ATTN_HEADS = 8
    DEPTH = 2
    DIM_HIDDEN = 650
    DROPOUT_RATE = .01
    LR = 1e-4

    MAX_CHARS = 24
    BATCH_SIZE = 16

    train_ld, val_ld, test_ld = get_data_loaders(NOISE_TYPE, BATCH_SIZE, MAX_CHARS)

    TOTAL_UPDATE_STEPS = int(len(train_ld) * EPOCHS)
    CHAR_VOCAB_SIZE = len(train_ld.dataset.vect.chars)
    TGT_VOCAB_SIZE = len(train_ld.dataset.vocab)

    # PATH
    RUN_NAME = f"MUDE_{NOISE_TYPE}"
    TENSORBOARD_RUN_LOG_DIR_PATH = TENSOBOARD_LOGS_DIR_PATH.joinpath(RUN_NAME)
    CHECKPOINTS_RUN_DIR_PATH = CHECKPOINTS_DIR_PATH.joinpath(RUN_NAME)

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
    opt = RMSprop(mude.parameters(), lr=LR)

    # losses and metric definition
    seq2seq_criterion = NLLLoss(ignore_index=train_ld.dataset.vect.PAD_CHAR_INDX)
    recog_criterion = NLLLoss(ignore_index=train_ld.dataset.vocab['<PAD>'])
    mtrcs = {"WRA": WordRecognitionAccuracy()}  # word recognition accuracy, i guess? :)

    trainer = create_trainer(
        model=mude,
        optimizer=opt,
        seq2seq_loss_fn=seq2seq_criterion,
        pred_loss_fn=recog_criterion,
        char_vocab_size=CHAR_VOCAB_SIZE,
        tgt_vocab_size=TGT_VOCAB_SIZE
    )

    evaluator = create_evaluator(
        model=mude,
        metrics=mtrcs
    )

    LOG_TRAINING_PROGRESS_EVERY_N = 32

    @trainer.on(Events.EPOCH_COMPLETED)
    def valid_evaluate(engine):
        print("VAL EVAL")
        epoch = engine.state.epoch
        evaluator.run(val_ld)
        val_wra_vle = round(evaluator.state.metrics['WRA'], 3)
        print(f"EPOCH:[{epoch}] VAL WRA:{val_wra_vle}")

    @trainer.on(Events.COMPLETED)
    def test(engine):
        print("TEST EVAL")
        evaluator.run(test_ld)
        test_wra_vle = round(evaluator.state.metrics["WRA"], 3)
        report = f"{RUN_NAME};{test_wra_vle}\n"
        with EVALUATION_RESULTS_FILE_PATH.open(mode='a') as f:
            f.writelines(report)
        print(f"TRAINING IS DONE FOR {RUN_NAME} RUN.")

    pbar = ProgressBar()

    checkpointer = ModelCheckpoint(
        CHECKPOINTS_RUN_DIR_PATH,
        filename_prefix=RUN_NAME.lower(),
        n_saved=None,
        score_function=lambda engine: round(engine.state.metrics['WRA'], 3),
        score_name='WRA',
        atomic=True,
        require_empty=True,
        create_dir=True,
        archived=False,
        global_step_transform=global_step_from_engine(trainer)
    )
    nan_handler = TerminateOnNan()
    coslr = CosineAnnealingScheduler(opt, "lr", start_value=LR, end_value=LR / 4, cycle_size=TOTAL_UPDATE_STEPS // 1)

    evaluator.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'_': mude})

    trainer.add_event_handler(Events.ITERATION_COMPLETED, nan_handler)
    trainer.add_event_handler(Events.ITERATION_COMPLETED, coslr)

    GpuInfo().attach(trainer, name='gpu')
    pbar.attach(
        trainer,
        output_transform=lambda output: {'loss': output['loss']},
        metric_names=[f"gpu:{args.gpu} mem(%)"]
    )

    # FIRE
    tb_logger = TensorboardLogger(log_dir=TENSORBOARD_RUN_LOG_DIR_PATH)
    tb_logger.attach(
        trainer,
        log_handler=OutputHandler(
            tag='training',
            output_transform=lambda output: {'loss': output['loss']}
        ),
        event_name=Events.ITERATION_COMPLETED(every=LOG_TRAINING_PROGRESS_EVERY_N)
    )
    tb_logger.attach(
        evaluator,
        log_handler=OutputHandler(
            tag='validation',
            metric_names='all',
            global_step_transform=global_step_from_engine(trainer)
        ),
        event_name=Events.EPOCH_COMPLETED
    )
    tb_logger.attach(
        trainer,
        log_handler=OptimizerParamsHandler(opt),
        event_name=Events.ITERATION_STARTED
    )
    tb_logger.attach(
        trainer,
        log_handler=WeightsHistHandler(mude),
        event_name=Events.EPOCH_COMPLETED
    )

    trainer.run(train_ld, max_epochs=EPOCHS)
    tb_logger.close()
    torch.save(mude.state_dict(), CHECKPOINTS_RUN_DIR_PATH.joinpath(f"{RUN_NAME}-last.pth"))
