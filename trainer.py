"""
Author: Mohamed Abdelaziz
Matr.Nr.: K12137202
Exercise 5
"""

import os
import numpy as np

import torch

from torch.utils.data import Subset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets import AugmentedData, collater
from architectures import *
from utils import plot

from tqdm import tqdm


class Trainer:
    def __init__(self, network_config, input_dir='inputs', output_dir='results', print_at=10, plot_at=1000, validate_at=500, save_at=500,
                input_image_size=(100,100), min_offset=0, max_offset=8, min_spacing=2, max_spacing=6, min_pixels=144,
                trainset_size=0.7, validset_size=0.15, testset_size=0.15,
                 batch_size=64, learning_rate=1e-3, weight_decay=1e-5, n_updates=1000, resume=0):

        super(Trainer, self).__init__()

        self.input_dir = input_dir
        self.output_dir = output_dir
        self.plot_dir = os.path.join(self.output_dir, "plots")
        self.tensorboard_logdir = os.path.join(self.output_dir, "tensorboard")
        self.best_model_path = os.path.join(self.output_dir, 'best_model.pt')
        self.checkpoint_path = os.path.join(self.output_dir, 'checkpoint.pth')
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)

        self.print_at = print_at
        self.plot_at = plot_at
        self.validate_at = validate_at
        self.save_at = save_at
        self.best_validation_loss = np.inf

        self.batch_size = batch_size
        self.trainset_size, self.validset_size, self.testset_size = trainset_size, validset_size, testset_size

        self.set_seed()

        self.input_image_size = input_image_size
        self.min_offset, self.max_offset, self.min_spacing, self.max_spacing = min_offset, max_offset, min_spacing, max_spacing
        self.min_pixels = min_pixels

        self.get_loaders()

        self.writer = SummaryWriter(log_dir=self.tensorboard_logdir)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # self.net = SimpleCNN(**network_config)
        self.net = VariantCNN(**network_config)
        self.net.to(self.device)


        self.mse = MSELoss()
        self.target_mse = MSELoss()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        self.update = 0
        self.n_updates = n_updates
        self.update_to_resume(resume)

        print(self.net)

    # Seeder
    def set_seed(self, seed=0):
        np.random.seed(seed)
        torch.manual_seed(seed)

    # Create Loaders
    def get_loaders(self):
        self.dataset = AugmentedData(self.input_dir, self.input_image_size, self.min_offset, self.max_offset, self.min_spacing, self.max_spacing, self.min_pixels)
        train_indices, valid_indices, test_indices = self.get_subset_indices()
        self.train_loader = DataLoader(Subset(self.dataset, indices=train_indices), batch_size=self.batch_size, shuffle=True, pin_memory=True, collate_fn=collater)
        self.valid_loader = DataLoader(Subset(self.dataset, indices=valid_indices), batch_size=self.batch_size, shuffle=False, pin_memory=True, collate_fn=collater)
        self.test_loader = DataLoader(Subset(self.dataset, indices=test_indices), batch_size=self.batch_size, shuffle=False, pin_memory=True, collate_fn=collater)

    # Create/Load/Save Subset Indices for Reproducibility
    def get_subset_indices(self):
        indices_path = os.path.join(self.input_dir, 'indeces.npz')
        try:
            indices = np.load(indices_path)
            training, validation, test = indices['train'].astype(np.int), indices['valid'].astype(np.int), indices['test'].astype(np.int)
        except FileNotFoundError:
            tr_up_bound = len(self.dataset) * self.trainset_size
            vn_up_bound = len(self.dataset) * (self.trainset_size + self.validset_size)
            training = np.arange(tr_up_bound)
            validation = np.arange(tr_up_bound, vn_up_bound)
            test = np.arange(vn_up_bound, len(self.dataset)-1)
            np.savez(indices_path, train=training, valid=validation, test=test)
        except:
            raise NotImplementedError('\t\t\tSomething Terrible Just Happened while loading/creating indices!!')
        return training, validation, test

    # Calculate RMSE [CPU ONLY]
    def rmse(self, outputs, targets):
        losses = []
        for i in range(len(outputs)):
            losses.append(torch.sqrt(self.target_mse(outputs[i].cpu(), targets[i])))
        return sum(losses) / len(outputs)

    # Get Targets
    def get_targets(self, outputs, knowns):
        return [outputs[i][knowns[i] == 0].clone() for i in range(len(outputs))]

    # Update Model/Optimizer/Losses if resuming
    def update_to_resume(self, resume):
        if resume:
            checkpoint = torch.load(self.checkpoint_path)
            print(checkpoint)
            self.net.load_state_dict(checkpoint['model_state_dict'])
            self.net.to(self.device)
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.loss_fn = checkpoint['loss']
            self.rmse_fn = checkpoint['target_loss']
            self.update = checkpoint['updates']
            print(f'Resuming from checkpoint: {self.update}')
        else:
            torch.save(self.net, self.checkpoint_path)

    # Train Model [Training Loop]
    def train(self):
        self.net.train()
        update_progress_bar = tqdm(total=self.n_updates, desc=f"loss: {np.nan:7.5f}", position=0, initial=self.update)
        while self.update < self.n_updates:
            for data in self.train_loader:
                self.optimizer.zero_grad()
                inputs, targets_, targets, knowns = data

                outputs = self.net(inputs.to(self.device))
                self.loss = self.mse(outputs, targets.to(self.device))
                self.loss.backward()
                self.optimizer.step()

                outputs_ = self.get_targets(outputs, knowns.to(self.device))
                self.target_loss = self.rmse(outputs_, targets_)

                if (self.update + 1) % self.print_at == 0:
                    self.write_stats(self.loss.cpu(), self.target_loss.cpu())
                self.plot_batch(inputs, targets, outputs)
                self.validate()
                self.save_checkpoint(self.loss, self.target_loss)

                update_progress_bar.set_description(f"loss: {self.loss:7.5f}", refresh=True)
                update_progress_bar.update()

                self.update += 1
                if self.update >= self.n_updates:
                    break
        update_progress_bar.close()
        self.writer.close()

        self.get_best_model_stats()
        self.save_checkpoint(self.loss, self.target_loss)
        return self.net

    # Evaluate dataset
    def evaluate_set(self, model, dataloader):
        model.eval()
        loss = 0
        target_loss = 0
        with torch.no_grad():
            for data in tqdm(dataloader, desc="scoring", position=0):
                inputs, targets_, targets, knowns = data

                outputs = model(inputs.to(self.device))
                loss += self.mse(outputs, targets.to(self.device)).item()

                outputs_ = self.get_targets(outputs, knowns.to(self.device))
                target_loss += self.rmse(outputs_, targets_)

        loss /= len(dataloader)
        target_loss /= len(dataloader)
        model.train()
        return loss, target_loss

    # Validate Model
    def validate(self):
        if (self.update + 1) % self.validate_at == 0:
            val_loss, val_target_loss = self.evaluate_set(self.net, self.valid_loader)
            self.write_stats(val_loss, val_target_loss, tags=['validation/loss', 'validation-target/loss'])
            self.write_histogram()
            if val_loss < self.best_validation_loss:
                self.best_validation_loss = val_loss
                torch.save(self.net, self.best_model_path)

    # Calculate Stats for Best Model
    def get_best_model_stats(self):
        print(f"Computing scores for best model")
        net = torch.load(self.best_model_path)
        train_loss = self.evaluate_set(net, self.train_loader)
        val_loss = self.evaluate_set(net, self.valid_loader)
        test_loss = self.evaluate_set(net, self.test_loader)
        print(f"Scores:\n\ttrain loss: {train_loss}\n\tvalid loss: {val_loss}\n\ttest loss: {test_loss}")
        with open(os.path.join(self.output_dir, "results.txt"), "w") as rf:
            print(f"Scores:\n\ttrain loss: {train_loss}\n\tvalid loss: {val_loss}\n\ttest loss: {test_loss}", file=rf)
        return net

    # Write Stats [Tensorboard]
    def write_stats(self, loss, target_loss, tags=['training/loss', 'training-target/loss']):
        self.writer.add_scalar(tag=tags[0], scalar_value=loss, global_step=self.update)
        self.writer.add_scalar(tag=tags[1], scalar_value=target_loss, global_step=self.update)

    def write_histogram(self):
        for i, (name, param) in enumerate(self.net.named_parameters()):
            self.writer.add_histogram(tag=f"validation/param_{i} ({name})", values=param.cpu(), global_step=self.update)
            self.writer.add_histogram(tag=f"validation/gradients_{i} ({name})", values=param.grad.cpu(), global_step=self.update)

    # Plot Batch
    def plot_batch(self, inputs, targets, outputs):
        if (self.update + 1) % self.plot_at == 0:
            plot(inputs.detach().cpu().numpy(), targets.detach().cpu().numpy(), outputs.detach().cpu().numpy(), self.plot_dir, self.update)

    # Save Checkpoint
    def save_checkpoint(self, loss, target_loss):
        if (self.update + 1) % self.save_at == 0:
            torch.save({'updates': self.update+1, 'model_state_dict': self.net.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': loss, 'target_loss': target_loss}, self.checkpoint_path)

    def __repr__(self):
        return f'Trainer:\n=======\nDataset:\n\tTotal Size:\t{len(self.dataset)}\n\tTrain Size:\t{len(self.train_loader)}\n\tValid Size:\t{len(self.valid_loader)}\n\tTest Size:\t{len(self.test_loader)}\nModel:\n{self.net}\nLoss:\t{self.mse}\nOptimizer:\n\t{self.optimizer}\nTotal Iterations:\t{self.n_updates}\nCurrent Iteration:\t{self.update}'
