# AI3610 Brain Inspired Intelligence Final Project

## Quick Start

### Setting up the environment

```sh
pip install -r requirements.txt
```

### Setting up the workspace

The training scripts assume that a `./logs` directory exists for storing logs, and that a `./model_save` directory exists for storing checkpoints. If they do not exist, they must be manually created before running training scripts.

```sh
mkdir logs
mkdir model_save
```

### Training

For example, to train an MLP model with batch size 64, and exponential gaussian perturbation (of std 1.0, by default)

```sh
python train.py --model mlp --model_save mlp_egwp_bs64.pt --batch_size 64 -p exp
```

- `-m` or `--model` specifies the architecture of the model (Can be one of `mlp`, `cnn`, `dmlp` (MLP with dropouts), `qmlp` (MLP with quantization), `mmlp` (Ensembled MLP) or `xzr` (another CNN, from another group))
- `--model_save` specifies the file to save the model. Models will be saved in `./model_save` directory
- `-p` or `--perturbator` specifies the perturbator (Can be one of `none`, `exp` (Exponential Gaussian) or `df` (Simulated MemTorch Device Fault))
- For more commandline arguments, please see the `parse_args` function in `common.py`

### Evaluation

#### Exponential Gaussian Noise

For example, to evaluate a trained CNN with ExpGaussian noise of std 1.0

```sh
python eval_exp_gaussian.py --model_save cnn_baseline_bs32.pt -m cnn --std 1.0
```

- `model_save` should indicate a path to a saved model (stored in `./model_save` directory)
- `-m` or `--model` should specify the architecture of the model
- `--std` is the standard deviation of the exponential gaussian noise to be added

#### MemTorch Deployment Simulation

For example, to evaluate a MLP model on MemTorch (without device faults)

```sh
python ./eval_memtorch.py --model_save mlp_egwp_bs64.pt -m mlp
```

To evaluate the same model with device faults

```sh
python ./eval_memtorch.py --model_save mlp_egwp_bs64.pt -m mlp --nonideality device_faults
```