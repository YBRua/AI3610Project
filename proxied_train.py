import os
import sys
import torch
import logging
import numpy as np
import torch.nn as nn
import torch.optim as optim

from datetime import datetime

import models
import data_prep
import perturbators
from trainer import Trainer
from common import parse_args


def setup_train_logger(args):
    logger = logging.getLogger('MDNN Proxied Training')
    logger.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_formatter = logging.Formatter("%(message)s")

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    model_name = os.path.splitext(args.model_save)[0]
    file_handler = logging.FileHandler(
        filename=f'./logs/proxy-{model_name}-{args.seed}-{timestamp}.log',
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
    logger = setup_train_logger(args)
    logger.info(' '.join(sys.argv))

    # experiment setup
    N_EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    NOISE_MEAN = args.mean
    NOISE_STD = args.std
    MODEL_SAVE = os.path.join('model_save', args.model_save)
    PROXY_SAVE = os.path.join('model_save', args.proxy_save)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # data prep
    train_set, test_set = data_prep.load_mnist('./data')
    train_loader = data_prep.wrap_dataloader(
        train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    test_loader = data_prep.wrap_dataloader(
        test_set, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    # initialization
    model = models.build_model(args.model)
    proxy = models.build_model(args.model)
    model.to(device)
    proxy.to(device)
    proxy.load_state_dict(torch.load(PROXY_SAVE))
    model.load_state_dict(torch.load(PROXY_SAVE))
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    trainer = Trainer()

    # perturbation
    if args.perturbator == 'exp':
        perturbator = perturbators.ExpGaussianPerturbator(
            model, device, NOISE_MEAN, NOISE_STD)
    elif args.perturbator == 'df':
        perturbator = perturbators.DeviceFaultPerturbator(
            model, device,
            lrs_proportion=0.05 if args.scheduled_perturb else 0.1,
            hrs_proportion=0.05 if args.scheduled_perturb else 0.1,
            scheduled=args.scheduled_perturb)
    elif args.perturbator == 'scheduled':
        perturbator = perturbators.ScheduledExpGaussianPerturbator(
            model, device, NOISE_MEAN, std_end=NOISE_STD, steps=N_EPOCHS - 1)
    else:
        perturbator = None

    # train
    best_val_loss = float('inf')
    for e in range(N_EPOCHS):
        trainer.proxy_train(
            e, model, proxy, optimizer,
            nn.MSELoss(),
            # nn.KLDivLoss(reduction='batchmean'),
            train_loader, device,
            perturbator, args)

        val_loss, val_acc = trainer.validate(
            model, loss_fn, test_loader, device)

        if perturbator is not None:
            perturbator.step()

        logger.info(
            f'| Epoch {e} '
            f'| Val Loss {val_loss:3.4f} '
            f'| Val Acc {val_acc:.4f} |')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE)
            logger.info(f'| Model saved as {MODEL_SAVE} |')


if __name__ == "__main__":
    args = parse_args()
    main(args)
