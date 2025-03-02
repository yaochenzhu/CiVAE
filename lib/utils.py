import torch
import numpy as np

def sigmoid(x):
    z = 1/(1 + np.exp(-x))
    return z

'''
    Simply take the difference of y in the
    treatment group and non-treatment group.
'''

def naiveEstimator(t_train, y_train):
    if type(t_train) is torch.Tensor:
        t_train = t_train.numpy()
    if type(y_train) is torch.Tensor:
        y_train = y_train.numpy()
    t_train = t_train.astype(bool)
    ate = y_train[t_train].mean() - y_train[~t_train].mean()
    return ate


def findConfounders(p_before, p_after, number):
    assert p_before.shape == p_after.shape
    latent_dim = p_before.shape[0]
    assert latent_dim >= number
    score_ijs = []
    for i in range(latent_dim):
        for j in range(i+1, latent_dim):
            score = p_after[i][j] - p_before[i][j]
            score_ijs.append((score, (i, j)))
    score_ijs.sort()
    confounders = set()
    for _, (i, j) in score_ijs:
        confounders.add(i)
        confounders.add(j)
        if len(confounders) >= number:
            break
    confounders = list(confounders)
    confounders.sort()
    return confounders