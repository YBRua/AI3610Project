import os
import sys
import torch
import logging
import numpy as np
import torch.nn as nn

from datetime import datetime

import models
import data_prep
from trainer import Trainer
from common import parse_args


def setup_memtorch_logger(args):
    logger = logging.getLogger('MDNN Memtorch Simulation')
    logger.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_formatter = logging.Formatter("%(message)s")

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    model_name = os.path.splitext(args.model_save)[0]
    file_handler = logging.FileHandler(
        filename=f'./logs/{model_name}-memtorch-{args.seed}-{timestamp}.log',
        mode='a',
        encoding='utf-8')
    file_formatter = logging.Formatter(
        "[%(levelname)s %(asctime)s]: %(message)s")

    stream_handler.setFormatter(stream_formatter)
    file_handler.setFormatter(file_formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger


def main(args):
    # set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    logger = setup_memtorch_logger(args)
    logger.info(' '.join(sys.argv))

    # experiment setup
    BATCH_SIZE = args.batch_size
    MODEL_SAVE = os.path.join('model_save', args.model_save)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    _, test_set = data_prep.load_mnist('./data')
    test_loader = data_prep.wrap_dataloader(
        test_set, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    # initialization
    model = models.MLP()
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    trainer = Trainer()

    # evaluation
    logger.info(f'Loading model from {MODEL_SAVE}')
    model.load_state_dict(torch.load(MODEL_SAVE))
    model.eval()

    # deploy to memristor
    if args.use_memtorch:
        model.load_state_dict(torch.load(MODEL_SAVE))
        model.eval()
        mloss, macc = trainer.validate_memtorch(
            model, loss_fn, test_loader, device)
        logger.info(f'| Memtorch | Loss {mloss:8.4f} | Acc {macc:5.4f} |')


if __name__ == "__main__":
    args = parse_args()
    main(args)
