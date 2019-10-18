import numpy as np
import pandas as pd
import torch
import scipy.stats
import matplotlib.pyplot as plt
from scipy.integrate import trapz
from data_loader import data_loader
from mc_dropout import MCDropout, training_MC


    
# load the data
# remove none value
df = pd.read_csv("california_housing.csv", sep = ",")
df = df.dropna(axis = 0)
# remove ocean_proximity
df = df.drop('ocean_proximity', axis = 1)

# split data into training and test set
np.random.seed(129)
msk = np.random.rand(len(df)) < 0.8
traindf = df[msk]
testdf = df[~msk]
x, y, x_test, y_test = data_loader(traindf, testdf)
num_data, num_feature = x.shape

# set training parameters
l = 1e-4
wr = l**2. / num_data
dr = 2. / num_data
learning_rate = 0.001
batch_size = 50
num_epoch = 1000
tolerance=0.002
patience = 20

skip_training = False  # Set this flag to True before validation

mlp = MCDropout()

# training model
mlp.train()
training_MC(mlp, x, y, x_test, y_test, learning_rate, batch_size, num_epoch, tolerance, patience)

if skip_training:
    mlp.load_state_dict(torch.load('MC_mlp_01.pth'))
    
    
#Monte Carlo Sample
# MC sample 
K_test = 100 # sample 20 times 
mlp.train() 
MC_samples = [mlp(x_test) for _ in range(K_test)]  
# calculate the means 
mean_samples = torch.stack([tup for tup in MC_samples]).view(K_test, x_test.shape[0]).cpu().data.numpy()   # shape K_test * val_set_size
mean = np.mean(mean_samples, 0)
epistemic_uncertainty = np.std(mean_samples, 0)

def hit_probability(y_test, mean, epistemic_uncertainty, confidence):
    confidence_lower, confidence_higher = scipy.stats.norm.interval(confidence , mean, epistemic_uncertainty)
    return np.sum([1 if y_test[i] <= confidence_higher[i] and y_test[i] >= confidence_lower[i] else 0 for i in range(len(y_test))]) / len(y_test)

confidence = np.linspace(0, 1.0, 21, endpoint = True)
#print(confidence)
plt.plot(confidence, confidence, '-*')
hit_ratio = [hit_probability(y_test, mean, epistemic_uncertainty, c) for c in confidence]
plt.plot(confidence, hit_ratio)
plt.xlabel("Confidence")
plt.ylabel("Probability")
plt.savefig("mcdropout.eps", format="eps", dpi=1000)
area1 = trapz(confidence, dx = 0.05)
area2 = trapz(hit_ratio, dx = 0.05)
print("Deviation = ", area1 - area2)

