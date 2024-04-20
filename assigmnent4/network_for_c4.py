import torch 
import torch.nn as nn
import torch.nn.functional as F

from imp import reload

import DataLoader
reload(DataLoader)

from DataLoader import InMemDataLoader
from DataLoader import C4DataSet


class Model(nn.Module):
    def __init__(self, dp=0.45):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(2, 25, 2, padding=1)
        self.bn1 = nn.BatchNorm2d(25)

        self.conv2 = nn.Conv2d(25, 15, 2, padding=1)
        self.bn2 = nn.BatchNorm2d(15)

        self.fc1 = nn.Linear(15*2*2, 12)
        self.bn3 = nn.BatchNorm1d(12)

        self.fc2 = nn.Linear(12, 3)

        self.do = nn.Dropout(dp)

    def forward(self, x):

        x = F.max_pool2d(self.conv1(x), 2)
        x = F.relu(self.bn1(x))

        x = F.max_pool2d(self.conv2(x), 2)
        x = F.relu(self.bn2(x))

        x = x.view(x.shape[0], -1)

        x = self.fc1(x)
        x = F.relu(self.bn3(x))

        x = self.do(x) # dropout

        x = self.fc2(x)

        x = nn.Softmax()(x)

        return x

    def loss(self, Out, Targets):
      return F.cross_entropy(Out, Targets)

def load_data():
    amount_of_games = 20000 
    moves_observed  = 25 
    all_samples = amount_of_games * moves_observed

    batch_size = 128
    train_size, val_size, test_size = int(all_samples/3), int(all_samples/3), int(all_samples/3)

    dataset = C4DataSet(amount_of_games, moves_observed).create_data_set()

    train_set = dataset[:train_size]
    val_set = dataset[train_size:train_size+val_size] 
    test_set = dataset[train_size+val_size:]

    data_loaders = {
        "train": InMemDataLoader(train_set, batch_size=batch_size, shuffle=True),
        "valid": InMemDataLoader(val_set, batch_size=batch_size, shuffle=False),
        "test": InMemDataLoader(test_set, batch_size=batch_size, shuffle=False),
    }

    return data_loaders

def initialize_weights(model):
    with torch.no_grad():
        for name, p in model.named_parameters():
            if 'weight' in name:
                if 'conv' in name:
                    f_in = p.shape[1]*p.shape[2]*p.shape[3]
                    p.normal_(0, torch.sqrt(torch.tensor(2./f_in)))
                elif 'bn' in name:
                    p = torch.ones_like(p)
                elif 'fc' in name:
                    f_in = p.shape[1]
                    p.normal_(0, torch.sqrt(torch.tensor(2./f_in)))
                else:
                    raise Exception('weird weight')

            elif 'bias' in name:
                p.zero_()
            else:
                raise Exception('weird parameter')
            

def train(model, data_loaders, num_of_epochs, train_loader, opt, device="cpu"):
  model.train()

  for data_loader in data_loaders.values():
    if isinstance(data_loader, InMemDataLoader):
        data_loader.to(device)

  iter = 0
  for e in range(num_of_epochs):
    model.train()
    print(f"Epoch {e+1}")

    for batch in train_loader:
      x = batch[0].to(device)
      y = batch[1].to(device)
      opt.zero_grad()
      iter += 1

      out = model(x)
      loss = nn.CrossEntropyLoss()(out, y)
      loss.backward()
      opt.step()

def run_training(model, data_loaders):
    lr = 0.00002
    weight_decay = 0.000001
    momentum = 0.9
    epochs = 12
    _device = "cpu"

    # opt = torch.optim.SGD(model.parameters(), lr=lr, weight_decay = weight_decay, momentum = momentum)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train(model=model, data_loaders=data_loaders, num_of_epochs=epochs, train_loader=data_loaders["train"], opt=opt, device=_device)


def find_network():
    data_loaders = load_data()

    model = Model()

    initialize_weights(model)
    run_training(model, data_loaders=data_loaders)

    return model 