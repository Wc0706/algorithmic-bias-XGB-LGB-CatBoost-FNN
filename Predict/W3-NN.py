import numpy as np
import pandas as pd
import os
import torch
from collections import OrderedDict

path = os.path.dirname(os.path.dirname(__file__))


class NN_Model(torch.nn.Module):
    def __init__(self, insize, hidsize, outsize):
        super().__init__()
        self.main = torch.nn.Sequential(OrderedDict([
            ('fc1', torch.nn.Linear(insize, hidsize)),
            ('relu1', torch.nn.ReLU()),
            ('fc2', torch.nn.Linear(hidsize, hidsize)),
            ('relu2', torch.nn.ReLU()),
            ('fc3', torch.nn.Linear(hidsize, hidsize)),
            ('relu3', torch.nn.ReLU()),
            ('fc4', torch.nn.Linear(hidsize, outsize)),
        ]))

    def forward(self, input_data):
        return self.main(input_data)


def load_weights_from_txt(model, weight_dir):
    fc_layers = ['fc1', 'fc2', 'fc3', 'fc4']
    for layer_name in fc_layers:
        weight_file = os.path.join(weight_dir, f'trained.{layer_name}.weight.bias.txt')
        df = pd.read_csv(weight_file, sep='\t', index_col=0)
        bias_col = df.columns[-1]
        bias = df[bias_col].values.astype(np.float32)
        weight = df.drop(columns=[bias_col]).values.astype(np.float32)
        layer = getattr(model.main, layer_name)
        layer.weight.data = torch.tensor(weight, dtype=torch.float32)
        layer.bias.data = torch.tensor(bias, dtype=torch.float32)


train = ''

xx = pd.read_csv(train, header=None)
xx = pd.DataFrame(xx)
tmp = xx.iloc[:, 5]
t = xx.shape[0]

arr1 = np.zeros((t, 3))

fnn = []

INPUT_SIZE = 6
HIDDEN_SIZE = 32
OUTPUT_SIZE = 1

for j in range(0, 3):
    print(f"Processing model {fnn[j]}...")

    weight_dir = os.path.join(path, 'predict', f'{fnn[j]}.fnn', 'testR1', '1.FNN', 'Model_1')

    model = NN_Model(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    load_weights_from_txt(model, weight_dir)
    model.eval()

    yy = []

    for i in range(xx.shape[0]):
        x = xx.iloc[i, :INPUT_SIZE].values.astype(np.float32)
        x = x.reshape(1, -1)

        input_tensor = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            output = model(input_tensor).item()

        yy.append(output)

    arr1[:, j] = yy

arr2 = np.mean(arr1, axis=1).reshape(-1, 1)

f = []
ff = []
t = pd.concat([tmp, pd.DataFrame(arr2)], axis=1)
t.columns = ['Temp', 'Fit']

c5_folder = os.path.join(path, "")
if not os.path.exists(c5_folder):
    os.makedirs(c5_folder)

txt_filename = ""
txt_save_path = os.path.join(c5_folder, txt_filename)

t.to_csv(txt_save_path, sep='\t', encoding='utf-8', index=None, header=True)
print(f"Results saved to: {txt_save_path}")
print(f"Three-model ensemble prediction completed!")