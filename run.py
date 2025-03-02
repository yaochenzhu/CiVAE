import sys
sys.path.append("lib")

import pyro
import torch
import numpy as np
from matplotlib import pyplot as plt


from model import CEVAE
from utils import sigmoid, naiveEstimator, findConfounders
from causallearn.utils.KCI.KCI import KCI_UInd, KCI_CInd
from disentangle_model import nfI2VAE 

def main():
    ### Simulate the dataset
    num_samples = 5000
    confounder_dim = 3
    mediator_dim = 3
    feature_dim = 20
    latent_dim = confounder_dim + mediator_dim

    causal_effects = 2 ### Direct Effects
    confounding_effects = np.array([1, 1, 1])
    ### Mediated Effects
    mediated_effects_a = np.array([-1,-1,-1])
    mediated_effects_b = np.array([1,1,1])
    
    for i in range(10):                             
        x_train, y_train, t_train, c_train, m_train = generateSamplesWithMultipleMixedConfounderMediator(
            num_samples=num_samples, 
            confounder_dim=confounder_dim,
            mediator_dim=mediator_dim,
            feature_dim=feature_dim, 
            causal_effects=causal_effects,
            confounding_effects=confounding_effects,
            mediated_effects_a=mediated_effects_a,
            mediated_effects_b=mediated_effects_b)
        x_test = x_train

        ### Train the vanilla CEVAE on observed features
        pyro.clear_param_store()
        cevae = CEVAE(outcome_dist="normal", 
                      latent_dim=latent_dim,
                      feature_dim=feature_dim)
        cevae.fit(x_train, t_train, y_train)
        ite = cevae.ite(x_test)  #individual treatment effect
        ate = ite.mean()

        ### Train the CiVAE model to infer the latent variables
        pyro.clear_param_store()
        nfi2_vae = nfI2VAE(outcome_dist="normal", 
                           latent_dim=latent_dim, 
                           feature_dim=feature_dim,
                           hidden_dim=latent_dim,
                           num_layers=2)
        nfi2_vae.fit(x_train, t_train, y_train, num_epochs=10)
        z_train = nfi2_vae.infer(x_train, t_train, y_train)

        ### Identify the latent confounders
        num_indtest = 2000
        z = z_train[:num_indtest, :]
        id_obj = KCI_UInd()
        cid_obj = KCI_CInd()
        pvalue_before = np.zeros((latent_dim, latent_dim), dtype=np.float32)
        pvalue_after  = np.zeros((latent_dim, latent_dim), dtype=np.float32)
        t_train_ = t_train.unsqueeze(-1).numpy()[:num_indtest]
        for i in range(latent_dim):
            for j in range(latent_dim):
                zi = z[...,i].unsqueeze(-1).numpy()
                zj = z[...,j].unsqueeze(-1).numpy()
                pvalue_before[i, j] = id_obj.compute_pvalue(zi, zj)[0]
                pvalue_after[i, j] = cid_obj.compute_pvalue(zi, zj, t_train_)[0]
        c_dims = findConfounders(pvalue_before, pvalue_after, number=confounder_dim)
        
        ### Train ATE estimation model on inferred confounders
        c_train_new = z_train[:, c_dims]
        c_test_new = c_train_new
        pyro.clear_param_store()
        cevae = CEVAE(outcome_dist="normal",
                      latent_dim=confounder_dim, 
                      feature_dim=len(c_dims))
        cevae.fit(c_train_new, t_train, y_train)
        ite = cevae.ite(c_test_new)  #individual treatment effect
        dis_ate = ite.mean()
        print("-"*5+"Trained on Inferred Confounders"+"-"*5)
        print("ATE by Naive: {:.3f}".format(naiveEstimator(t_train, y_train)))
        print("ATE by CEVAE: {:.3f}".format(ate.numpy()))
        print("ATE by CiVAE: {:.3f}".format(dis_ate.numpy()))
        
if __name__ == "__main__":
	main()
