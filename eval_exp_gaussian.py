import os
import sys
import torch
import logging
import numpy as np
import torch.nn as nn

from datetime import datetime

import noise
import models
import data_prep
from trainer import Trainer
from common import parse_args


def setup_eval_logger(args):
    logger = logging.getLogger('MDNN Evaluation')
    logger.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_formatter = logging.Formatter("%(message)s")

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    model_name = os.path.splitext(args.model_save)[0]
    file_handler = logging.FileHandler(
        filename=f'./logs/eval-{model_name}-{args.seed}-{timestamp}.log',
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
    logger = setup_eval_logger(args)
    logger.info(' '.join(sys.argv))

    # experiment setup
    EVAL_RUNS = args.eval_runs
    BATCH_SIZE = args.batch_size
    NOISE_MEAN = args.mean
    NOISE_STD = args.std
    MODEL_SAVE = os.path.join('model_save', args.model_save)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    _, test_set = data_prep.load_mnist('./data')
    test_loader = data_prep.wrap_dataloader(
        test_set, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    # initialization
    model = models.build_model(args.model)
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    trainer = Trainer()

    # evaluation
    logger.info(f'Loading model from {MODEL_SAVE}')

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
            # ori_param = copy.deepcopy(model.state_dict())

            # add noise
            noise.add_noise_to_weights(
                model, device, NOISE_MEAN, NOISE_STD)

            # eval with exponential gaussian noise
            # noise_param = copy.deepcopy(model.state_dict())
            test_loss, test_acc = trainer.validate(
                model, loss_fn, test_loader, device)

            accs.append(test_acc)
            losses.append(test_loss)

            logger.info(
                f"| Run {n} | W/  noise "
                f"| Loss {test_loss:8.4f} | Acc {test_acc:5.4f} |")
            logger.info('=' * 84)

            # perturbed model info
            # largest_deviations = []
            # logger.info(
            #     f'| {"Layer":<20} '
            #     f'| {"Deviation":17} '
            #     f'| {"Original":17} '
            #     f'| {"Noisy":17} |')
            # logger.info('-' * 84)
            # for key in ori_param.keys():
            #     largest_deviations.append(
            #         torch.max(
            #             torch.abs(ori_param[key] - noise_param[key])
            #         ).item())
            #     ori_max = torch.max(ori_param[key]).item()
            #     ori_min = torch.min(ori_param[key]).item()
            #     noise_max = torch.max(noise_param[key]).item()
            #     noise_min = torch.min(noise_param[key]).item()
            #     logger.info(
            #         f'| {key:<20} '
            #         f'| {largest_deviations[-1]:17.2f} '
            #         f'| {ori_max:8.2f} {ori_min:8.2f} '
            #         f'| {noise_max:8.2f} {noise_min:8.2f} |')
            # logger.info('=' * 84)

    logger.info(f'Done {EVAL_RUNS} runs')
    logger.info(f'Average Accuracy: {np.mean(accs):.4f}')
    logger.info(f'Average Loss: {np.mean(losses):.4f}')


if __name__ == "__main__":
    args = parse_args()
    main(args)
