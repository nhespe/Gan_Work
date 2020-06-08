""" Gans are pretty simple - this one is the core example of MNIST classifciatino"""
import torch
import numpy as numpy
import pandas as pd

from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets

EPOCHS = 100
LEARNING_RATE = 0.0002

class Discriminator(torch.nn.Module):
    """ This is simply just a MNIST classifier 

    """
    def __init__(self, dropout=.2, relu_param=.2):
        super().__init__()
        self.relu_param = relu_param
        self.dropout = dropout
        
        # [(784, 1024), (1024, 512), (512, 256), (512, 256)]
        self.input = nn.Sequential( 
            nn.Linear(784, 512),
            nn.LeakyReLU(self.relu_param),
            nn.Dropout(self.dropout)
        )
        self.h1 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(self.relu_param),
            nn.Dropout(self.dropout)
        )
        self.output = nn.Sequential(
            torch.nn.Linear(256, 1), 
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = self.input(x)
        x = self.h1(x)
        x = self.output(x)
        return x

class Generator(torch.nn.Module):
    """
        Noise goes in, image goes out 100 -> 784 (28*28)
    """
    def __init__(self, relu_param=.2):
        super().__init__()
        self.relu_param = relu_param
        
        self.input = nn.Sequential(
            nn.Linear(100, 256),
            nn.LeakyReLU(self.relu_param)
        )
        self.h1 = nn.Sequential(            
            nn.Linear(256, 512),
            nn.LeakyReLU(self.relu_param)
        )
        self.output = nn.Sequential(
            nn.Linear(512, 784),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.input(x)
        x = self.h1(x)
        x = self.output(x)
        return x

def train_discriminator(optimizer, real_data, fake_data):
    """ Wrapper function for the training of the disc function

    """
    batch_size = real_data.size(0)
    optimizer.zero_grad()
    
    # Train on MNSIT Data
    prediction_real = discriminator(real_data)
    error_mnist = loss(prediction_real, ones_target(batch_size) )
    error_mnist.backward()

    # Train on Generated Data
    prediction_fake = discriminator(fake_data)
    error_gen = loss(prediction_fake, zeros_target(batch_size))
    error_gen.backward()
    
    optimizer.step()
    return (error_mnist + error_gen)


def train_generator(optimizer, gen_output):
    """ Wrapper function for the training of the generator function 

    """
    batch_size = gen_output.size(0)
    optimizer.zero_grad()

    # generate fake data
    prediction = discriminator(gen_output)

    # Calculate error and backpropagate
    error = loss(prediction, ones_target(batch_size))
    error.backward()
    optimizer.step()

    return error

def main():
    gen_optimizer = optim.Adam(generator.parameters(), lr=LEARNING_RATE)
    desc_optimizer = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE)
    loss = nn.BCELoss()

    # this is a test set we can see it each interval
    test_sample_rand_noise = Variable(torch.randn(16, 100))

    # Total number of epochs to train
    for epoch in range(EPOCHS):
        for idx, (image_batch,_) in enumerate(data_loader):
            ##  Train Discriminator
            real_data = Variable(images_to_vectors(image_batch))
            gen_output = generator(Variable(torch.randn(image_batch.size(0), 100))).detach()
            disc_error = train_discriminator(desc_optimizer, real_data, gen_output)

            ## Train Generator
            gen_output = generator(Variable(torch.randn(image_batch.size(0), 100)))
            g_error = train_generator(gen_optimizer, gen_output)
        
        print(f"IDX: {idx}, disc_error: {disc_error}, gen_error: {gen_error}")
        # image_vectors = generator(test_sample_rand_noise)
        # image_vectors.view(image_vectors.size(0), 1, 28, 28)
        # test_images = test_images.data
        # pkl.dump(f"batch_{idx}_sample_out.pkl", test_images)
