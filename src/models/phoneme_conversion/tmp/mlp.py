
# Structured after this resource: https://github.com/raminnakhli/HMM-DNN-Speech-Recognition

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

# The DNN portion of the DNN-HMM

# Essentially, this is an MLP that looks as such (feature_size -> 40 -> 30 -> 20 -> class_count)
class NeuralNetwork(nn.Module):
    # input features i.e. <feature_size> are generally from extracted windows of audio (up to 1-2 seconds)
    def __init__(self, feature_size, class_count):
        super(NeuralNetwork, self).__init__()
        mid1_neuron = 40
        mid2_neuron = 30
        mid3_neuron = 20

        self.layer1 = nn.Sequential(
            nn.Linear(feature_size, mid1_neuron),
            nn.LeakyReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(mid1_neuron, mid2_neuron),
            nn.LeakyReLU(),
        )
        self.layer3 = nn.Sequential(
            nn.Linear(mid2_neuron, mid3_neuron),
            nn.LeakyReLU(),
        )
        self.layer4 = nn.Sequential(
            nn.Linear(mid3_neuron, class_count)
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out

class MLP():
    def __init__(self, feature_size, class_count):
        self.net = NeuralNetwork(feature_size, class_count)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.RMSprop(self.net.parameters())
        self.epoch_count = 20
    
    def train(self, data, label, epoch=None):
        self.net.train()
        number_of_epochs = epoch

        # train for the number of epochs
        for epoch in range(number_of_epochs):
            # split into batches
            for i, (batch_data, batch_label) in enumerate(zip(data, label)):
                batch_data = batch_data.reshape(1, -1)

                batch_data = Variable(torch.from_numpy(batch_data)).float()
                batch_label = Variable(torch.from_numpy(batch_label)).long()

                self.optimizer.zero_grad()
                score = self.net(batch_data)

                loss = self.loss_function(score, batch_label)

                loss.backware()
                self.optimizer.step()
    
    def log_probability(self, data):
        self.net.eval()

        data = Variable(torch.from_numpy(data)).float()

        scores = self.net(data)
        prob = softmax_module(scores)
        return prob.data.numpy()
    
    def predict(self, data):
        self.net.eval()

        data = Variable(torch.from_numpy(data)).float()

        scores = self.net(data)
        _, predicted = torch.max(scores.data, 1)
        return predicted