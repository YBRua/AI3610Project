from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--seed', type=int, default=1234)
    parser.add_argument(
        '--mean', type=float, default=0)
    parser.add_argument(
        '--std', type=float, default=1)
    parser.add_argument(
        '--model_save', '-s', type=str,
        default='model.pt')
    parser.add_argument(
        '--model', '-m',
        choices=['mlp', 'cnn'], default='mlp')
    parser.add_argument(
        '--perturbator', '-p',
        choices=['none', 'exp', 'df', 'scheduled'], default='none')
    parser.add_argument(
        '--regularization', '-r', action='store_true')
    parser.add_argument(
        '--batch_size', '-b', type=int, default=32)
    parser.add_argument(
        '--epochs', '-e', type=int, default=5)
    parser.add_argument(
        '--nonideality',
        choices=['none', 'device_faults'],
        default='none')
    parser.add_argument(
        '--eval_runs', type=int, default=5)

    return parser.parse_args()
