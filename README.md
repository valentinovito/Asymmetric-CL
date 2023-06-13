# Asymmetric-CL
This repository contains code implementations of the asymmetric focal contrastive loss (AFCL) and the model architecture from the paper "[An Asymmetric Contrastive Loss for Handling Imbalanced Datasets](https://arxiv.org/abs/2207.07080)". The model consists of a two-stage training strategy, as outlined in the following figure:

![alt text](https://github.com/valentinovito/Asymmetric-CL/blob/main/architecture.jpg)

AFCL is a generalization of both the vanilla contrastive loss (CL) and the focal contrastive loss (FCL). The loss is proposed as a way to directly address the problem of class imbalance in a dataset. Experimental results show that AFCL is able to outperform CL and FCL in terms of weighted and unweighted classification accuracies.

In the implementation, the parameters 'gamma' and 'eta' have a default value of 0. When dealing with significantly imbalanced datasets, it is advised to increase the value of 'gamma' and 'eta' to up to 10 and 300, respectively. Read the paper for more details on suggested hyperparameter values depending on the class imbalance scenario.
