"""
Author: Mohamed Abdelaziz
Matr.Nr.: K12137202
Exercise 5
"""

import os

import pickle
import torch
import numpy as np

from datasets import AugmentedData

from architectures import *

from tqdm import tqdm
import matplotlib.pyplot as plt


class Tester:
    """docstring for Tester."""

    def __init__(self, network_config, input_dir, output_dir, type, plot):
        super(Tester, self).__init__()
        self.predictions = []

        self.input_dir = input_dir
        self.output_dir = output_dir
        self.models_dir = models_dir
        self.type = type
        self.plot = plot

        self.plot_dir = os.path.join(self.output_dir, "test_plots")
        self.best_model_path = os.path.join(self.models_dir, 'pretrained.pt')

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)

        if self.type == 'pkl':
            self.inputs, self.knowns = self.load_pickle()
        elif self.type == 'images':
            # Under Dev.
            self.inputs, self.knowns, _, __ = AugmentedData(**network_config)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torch.load(self.best_model_path)

        self.model.to(self.device)

    def load_pickle(self):
        testset = pickle.load(open(os.path.join(self.input_dir, 'test_inputs.pkl'), 'rb'))
        return testset['input_arrays'], testset['known_arrays']

    def test(self):
        with open(os.path.join(self.output_dir, 'predictions.pkl'), 'wb') as f:
            with torch.no_grad():
                for idx in tqdm(range(len(self.inputs))):
                    input = torch.from_numpy(self.inputs[idx]).unsqueeze(dim=0).float()
                    output = self.model(input.to(self.device)).detach().cpu().numpy()
                    if self.plot:
                        self.plot_set(input.numpy(), output, idx)
                    target = self.get_target(output, self.knowns[idx])
                    self.predictions.append(target)
            pickle.dump(self.predictions, f)
        return self.predictions

    def get_target(self, output, known):
        output = output.squeeze()
        return output[known == 0].copy().astype(dtype=np.uint8)

    def plot_set(self, inputs, predictions, update):
        """Plotting the inputs, targets and predictions to file `path`"""
        # os.makedirs(path, exist_ok=True)
        fig, axes = plt.subplots(ncols=2, figsize=(15, 5))

        for i in range(len(inputs)):
            for ax, data, title in zip(axes, [inputs, predictions], ["Input", "Prediction"]):
                ax.clear()
                ax.set_title(title)
                ax.imshow(data[i].astype(np.uint).T)
                ax.set_axis_off()
            fig.savefig(os.path.join(self.plot_dir, f"{update:07d}_{i:02d}.png"), dpi=100)
        plt.close(fig)
