import copy
import torch
import memtorch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torch.utils.data import DataLoader

from memtorch.mn.Module import patch_model
from memtorch.map.Input import naive_scale
from memtorch.map.Parameter import naive_map
from memtorch.bh.nonideality.NonIdeality import apply_nonidealities

import perturbators


def weight_mixer_factory(end_w: float, begin_w: float, steps: int):
    def weight_mixer(epoch: int):
        return end_w + (begin_w - end_w) * (steps - epoch) / steps

    return weight_mixer


class Trainer:
    def __init__(self) -> None:
        pass

    def proxy_train(
            self,
            e: int,
            model: nn.Module,
            proxy: nn.Module,
            optimizer: optim.Optimizer,
            loss_fn: nn.Module,
            train_loader: DataLoader,
            device: torch.device,
            perturbator: perturbators.Perturbator,
            args):

        model.train()
        proxy.eval()
        progress = tqdm(train_loader)
        tot_loss = 0
        tot_acc = 0

        for bid, (x, y) in enumerate(progress):
            x, y = x.to(device), y.to(device)

            if perturbator is not None:
                perturbator.perturb_model()

            optimizer.zero_grad()
            model_out = torch.softmax(model(x), dim=-1)
            proxy_out = torch.softmax(proxy(x), dim=-1)
            pred = model_out.argmax(dim=1)

            loss = loss_fn(model_out, proxy_out)
            loss.backward()
            optimizer.step()
            tot_loss += loss.item()
            tot_acc += (pred == y).float().mean().item()

            if perturbator is not None:
                perturbator.restore_perturbation()

            avg_loss = tot_loss / (bid + 1)
            avg_acc = tot_acc / (bid + 1)
            progress.set_description(
                f'| Epoch {e} | Loss {avg_loss:3.4f} | Acc {avg_acc:.4f}')

    def train(
            self,
            e: int,
            model: nn.Module,
            optimizer: optim.Optimizer,
            loss_fn: nn.Module,
            train_loader: DataLoader,
            device: torch.device,
            perturbator: perturbators.Perturbator,
            args):

        model.train()
        progress = tqdm(train_loader)
        tot_loss = 0
        tot_acc = 0
        for bid, (x, y) in enumerate(progress):
            x, y = x.to(device), y.to(device)

            if perturbator is not None:
                perturbator.perturb_model()

            optimizer.zero_grad()
            output = model(x)
            pred = output.argmax(dim=1)
            loss = loss_fn(output, y)

            if args.regularization:
                reg_penalty = 0
                for param in model.parameters():
                    if param.requires_grad:
                        reg_penalty += torch.norm(param, p=2)
                loss += 0.001 * reg_penalty

            loss.backward()
            optimizer.step()

            if perturbator is not None:
                perturbator.restore_perturbation()

            tot_loss += loss.item()
            tot_acc += (pred == y).float().mean().item()

            avg_loss = tot_loss / (bid + 1)
            avg_acc = tot_acc / (bid + 1)
            progress.set_description(
                f'| Epoch {e} | Loss {avg_loss:3.4f} | Acc {avg_acc:.4f}')

    def validate(
            self,
            model: nn.Module,
            loss_fn: nn.Module,
            test_loader: DataLoader,
            device: torch.device,
            use_tqdm: bool = False):
        model.eval()
        with torch.no_grad():
            tot_loss = 0
            tot_acc = 0
            if use_tqdm:
                prog = tqdm(test_loader)
            else:
                prog = test_loader
            for bid, (x, y) in enumerate(prog):
                x, y = x.to(device), y.to(device)

                output = model(x)
                pred = output.argmax(dim=1)
                loss = loss_fn(output, y)

                tot_loss += loss.item() * x.shape[0]
                tot_acc += (pred == y).float().sum().item()

            avg_loss = tot_loss / len(test_loader.dataset)
            avg_acc = tot_acc / len(test_loader.dataset)

        return avg_loss, avg_acc

    def validate_memtorch(
            self,
            model: nn.Module,
            loss_fn: nn.Module,
            test_loader: DataLoader,
            device: torch.device,
            nonideality: str = 'none'):
        # memristor setup
        reference_memristor = memtorch.bh.memristor.VTEAM
        reference_memristor_params = {'time_series_resolution': 1e-10}
        # memristor = reference_memristor(**reference_memristor_params)
        # memristor.plot_hysteresis_loop()
        # memristor.plot_bipolar_switching_behaviour()

        # convert DNN to MDNN
        patched_model = patch_model(
            copy.deepcopy(model),
            memristor_model=reference_memristor,
            memristor_model_params=reference_memristor_params,
            module_parameters_to_patch=[nn.Linear, nn.Conv2d],
            mapping_routine=naive_map,
            transistor=True,
            programming_routine=None,
            tile_shape=(128, 128),
            max_input_voltage=0.3,
            scaling_routine=naive_scale,
            ADC_resolution=8,
            ADC_overflow_rate=0.,
            quant_method='linear')
        patched_model.tune_()

        if nonideality == 'device_faults':
            patched_model_ = apply_nonidealities(
                copy.deepcopy(patched_model),
                non_idealities=[
                    memtorch.bh.nonideality.NonIdeality.DeviceFaults],
                lrs_proportion=0.1,
                hrs_proportion=0.1,
                electroform_proportion=0)
        elif nonideality == 'none':
            patched_model_ = patched_model
        else:
            raise ValueError(f'Unknown nonideality: {nonideality}')

        val_loss, val_acc = self.validate(
            patched_model_, loss_fn, test_loader, device)
        return val_loss, val_acc
