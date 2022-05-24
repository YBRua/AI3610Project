import os
import json
import torch
import numpy as np
import torch.nn as nn

import logging
from datetime import datetime

import noise
import models
import data_prep
from trainer import Trainer
from common import parse_args


def setup_wlls_logger(args):
    logger = logging.getLogger('MDNN Weight-Loss Landscape')
    logger.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_formatter = logging.Formatter("%(message)s")

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    model_name = os.path.splitext(args.model_save)[0]
    file_handler = logging.FileHandler(
        filename=f'./logs/weightloss-{model_name}-{args.seed}-{timestamp}.log',
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

    # experiment setup
    EVAL_RUNS = args.eval_runs
    BATCH_SIZE = args.batch_size
    NOISE_MEAN = args.mean
    NOISE_STD = np.arange(0, 1.6, 0.1)
    MODEL_SAVE = os.path.join('model_save', args.model_save)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = setup_wlls_logger(args)

    _, test_set = data_prep.load_mnist('./data')
    test_loader = data_prep.wrap_dataloader(
        test_set, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    # initialization
    save_name = os.path.splitext(args.model_save)[0]
    model = models.build_model(args.model)
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    trainer = Trainer()

    # evaluation
    logger.info(f'Loading model from {MODEL_SAVE}')

    full_accs, full_losses = [], []
    for std in NOISE_STD:
        logger.info(f'Evaluation with std {std}')

        accs, losses = [], []
        for n in range(EVAL_RUNS):
            model.load_state_dict(torch.load(MODEL_SAVE))
            model.eval()
            logger.info(f'Evaluation Run {n}')
            with torch.no_grad():
                # regular eval
                test_loss, test_acc = trainer.validate(
                    model, loss_fn, test_loader, device)
                logger.info('=' * 84)
                logger.info(
                    f"| Run {n} | W/o noise "
                    f"| Loss {test_loss:8.4f} | Acc {test_acc:5.4f} |")

                # add noise
                noise.add_noise_to_weights(
                    model, device, NOISE_MEAN, std)

                # eval with exponential gaussian noise
                test_loss, test_acc = trainer.validate(
                    model, loss_fn, test_loader, device)

                accs.append(test_acc)
                losses.append(test_loss)

                logger.info(
                    f"| Run {n} | W/  noise "
                    f"| Loss {test_loss:8.4f} | Acc {test_acc:5.4f} |")
                logger.info('=' * 84)

        avg_acc = np.mean(accs)
        avg_loss = np.mean(losses)
        full_accs.append(avg_acc)
        full_losses.append(avg_loss)
        logger.info(f'Average Accuracy: {avg_acc:.4f}')
        logger.info(f'Average Loss: {avg_loss:.4f}')

    logger.info("Done")
    JSON_DUMP_PATH = os.path.join(
        'weight_losses',
        save_name + '.json')
    with open(JSON_DUMP_PATH, 'w') as f:
        json.dump({
            'stds': NOISE_STD.tolist(),
            'accs': full_accs,
            'losses': full_losses
        }, f)


if __name__ == "__main__":
    args = parse_args()
    main(args)
