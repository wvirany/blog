---
title: "Understanding the Neural Network Gaussian Process"  
date: "2024-08-20"  
summary: ""  
description: ""  
draft: false  
toc: false  
readTime: true  
autonumber: false  
math: true  
tags: ["math"]
showTags: true  
hideBackToTop: false
---

Despite their overwhelming success in modern machine learning, deep neural networks remain poorly understood from a theoretical perspective. Classical statistical wisdom dictates that overparameterized models (i.e., models with more degrees of freedom than data samples) should overfit noisy data and thus generalize poorly. Yet, even in cases in which deep neural networks fit noisy training data almost perfectly, they still exhibit good generalizability. This contradiction has highlighted a serious gap between the theory and practice of deep learning, motivating the need for a more complete theoretical framework for deep learning.

An interesting result in the theoretical study of deep learning is the Neural Network Gaussian Process (NNGP), which shows the equivalence between neural networks and Gaussian processes (GPs). Indeed, \cite{neal1996priors} first showed that the distribution over functions represented by neural networks with a single hidden layer converges to a GP in the limit as the number of hidden units is taken to infinity[^fn1]. More recently, \cite{lee2018deep} extended this work to the case of arbitrary depth.

In this blog, I briefly explain some of the background mathematics necessary in the development of the NNGP. Then, I show that an infinitely wide neural network of any depth is Gaussian process. Previous

I aim to clarify mathematical arguments by detailing a thorough notation.

#### Notation

Consider an $L$-Layer neural network, with each layer $l$ consisting of $n_l$ hidden units. Let $\mathbf{x} = (x_1, \dots, x_{n_0})$ denote the input of the network. Then, the forward pass is defined by the following series of computations:

$$
\begin{align*}
\text{Input:} \ \ A_j^{(0)} &= x_j \\\\
\text{Pre-activation:} \ \ Z_j^{(l+1)} &= b_j^{(l)} + \sum_{i=0}^{n_l}w_{ji}^{(l)}A_i^{(l)} \\\\
\text{Post-activation:} \ \ A_j^{(l+1)} &= h \left( Z_j^{(l+1)} \right) \\\\
\text{Output:} \ \ f_j(\mathbf{x}; \theta) &= b_j^{(L)} + \sum_{i=0}^{n_L}w_{ji}^{(L)}A_i^{(L)}
\end{align*}
$$

Here,

* $Z_j^{(l)}$ denotes the $j$th hidden unit in layer $l$ before the activation function,
* $A_j^{(l)}$ denotes the $j$th hidden unit in layer $l$ after the activation function,
* $h(\cdot)$ is some nonlinear activation function,
* and $w_{ji}^{(l)}, b_j^{(l)}$ denote the weights and biases at layer $l$, respectively.

Often, we write the $f_j(\mathbf{x}; \theta)$ as $f_j(\mathbf{x})$, where it is implied that the output function is parameterized by the vector $\theta$. Similarly, each hidden unit is itself a function of the input **x**, so we can write, e.g., $A_j^{(l)} = A_j^{(l)}(\mathbf{x})$. It is also sometimes convenient to talk about computations involving entire layers instead of individual hidden units. As such, it is common to remove the subscript and denote an entire layer by, e.g., $A^{(l)}$, whereas the corresponding hidden units are denoted $A_j^{(l)}$.

For the sake of clarity, I will consider the case in which there is only one output unit. Thus, the function represented by the neural network can be written as $f(\mathbf{x})$, where I have removed the subscript $j$. The more general case is a fairly straightforward extension, in which each output unit $f_j(\mathbf{x})$ is itself a GP. The nuances arise in how the output unit is processed; e.g., a classification task in which the prediction is decided by taking the unit with the largest value, corresponding to a probability distribution under the softmax function.

Finally, if a function $f(x)$ is a GP with mean function $m(x)$ and covariance function $k(x, x^{\prime})$, I will denote this by $f(x) \sim \mathcal{GP}(m, k)$.



#### Single-Layer Neural Networks as Gaussian Processes

First, we assume the weight and bias parameters are drawn i.i.d. from a Normal distribution with $\mu_b = \mu_w = 0, \sigma_b^2 = 1$, and $\sigma_w^2 = 1/n_l$. Following the notation previously introduced, we begin with the computation of the pre-activation units in the first layer:

$$
Z_j^{(1)} = b_j^{(0)} + \sum_{i=1}^{n_0}w_{ji}^{(0)}x_i.
$$

Since each $w_{ji}, b_j$ is i.i.d. Normal, then $Z_j^{(1)}$ is i.i.d. Normal[^fn2]&nbsp;with mean

$$
\begin{align*}
\mathbb{E} \left[ Z_j^{(1)} \right] &= \mathbb{E} \left[ b_j^{(0)} + \sum_{i=1}^{n_0} w_{ji}^{(0)} x_i \right] \\\\
&= \mathbb{E} \left[ b_j^{(0)} \right] + \sum_{i=1}^{n_0} x_i \mathbb{E} \left[ w_{ji}^{(0)} \right] = 0,
\end{align*}
$$

where I've used the fact that the mean of each $w_{ji}, b_j$ is zero. Then, the distribution of the pre-activation units in the first layer can be wholly described by the covariance:

$$
\begin{align*}
& \mathbb{E} \left[ Z_j^{(1)}(\mathbf{x}) \\, Z_j^{(1)}(\mathbf{x}^{\prime}) \right] = \mathbb{E} \left[ \left( b_j^{(0)} + \sum_{i=1}^{n_0} w_{ji}^{(0)} x_i \right) \left( b_j^{(0)} + \sum_{i=1}^{n_0} w_{ji}^{(0)} x_i^{\prime} \right) \right] \\\\
    &= \mathbb{E} \left[ b_j^{(0)} \right]^2 + \mathbb{E} \left[ b_j^{(0)} \sum_{i=1}^{n_0} w_{ji}^{(0)} x_i \right] + \mathbb{E} \left[ b_j^{(0)} \sum_{i=1}^{n_0} w_{ji}^{(0)} x_i^{\prime} \right] + \sum_{i=1}^{n_0} \sum_{k=1}^{n_0} x_i \, x_k^{\prime} \mathbb{E} \left[ w_{ji}^{(0)} \right] \\\\
    &= \sigma_b^2 + \mathbb{E} \left[ b_j^{(0)} \right] \mathbb{E} \left[ w_{ji}^{(0)} \right] \sum_{i=1}^{n_0} (x_i + x_i') + \sigma_w^2 \sum_{i=1}^{n_0} \sum_{k=1}^{n_0} x_i \, x_k^{\prime}.
\end{align*}
$$

The second term is achieved as a result of the fact that the parameters are i.i.d., and since $\mu_b = \mu_w = 0$, it vanishes. Then, since $\sigma_b^2 = 1$ and $\sigma_w^2 = 1/n_l$, this becomes 

$$
\mathbb{E} \left[ Z_j^{(1)}(\mathbf{x}) \\, Z_j^{(1)}(\mathbf{x}^\prime) \right] = 1 + \frac{1}{n_0} \\, \mathbf{x}^\top \mathbf{x}^\prime.
$$

We can then compute the post-activation units via

$$
A_j^{(1)} = h \left( Z_j^{(1)} \right).
$$

The distribution of each $A_j^{(1)}$ for arbitrary activation functions is complicated, and in general it no longer follows a Gaussian distribution. However, we can say that each post-action unit is i.i.d. Then, we compute the output of the single-layer nueral network via another affine transformation:

$$
f(\mathbf{x}) = b^{(1)} + \sum_{i=1}^{n_1} w_i^{(1)} \\, A_i^{(1)},
$$

and since each $A_i^{(1)}$ is i.i.d., then it follows from the Central Limit Theorem (CLT) that $f(\mathbf{x})$ takes a Normal distribution in the limit as $n_1 \to \infty$. Thus, for any finite set of inputs **x**, $f(\mathbf{x})$ will follow a multivariate Normal distribution; this is precisely the definition of a Gaussian process. It follows that

$$
f(\mathbf{x}) \sim \mathcal{GP}(m, k),
$$

where

$$
m(\mathbf{x}) = \mathbf{E} \left[ f(\mathbf{x}) \right] = 0,
$$

and

$$
\begin{align*}
    k(\mathbf{x}, \mathbf{x}^\prime) &= \text{Cov}{\left( f(\mathbf{x}), f(\mathbf{x}') \right)} \\\\
        &= \mathbb{E} \left[ \left( b^{(1)} + \sum_{i=1}^{n_1} w_{ji}^{(1)} A_i^{(1)}(\mathbf{x}) \right) \left( b^{(1)} + \sum_{i=1}^{n_1} w_{ji}^{(1)} A_i^{(1)} (\mathbf{x}') \right) \right] \\\\
        &= \sigma_b^2 + \sigma_w^2 \mathbb{E} \left[ A^{(1)}(\mathbf{x}), A^{(1)}(\mathbf{x}') \right] \\\\
        &= 1 + \frac{1}{n_1} C(\mathbf{x}, \mathbf{x}^\prime),
\end{align*}
$$

where I have introduced the covariance function $C(\mathbf{x}, \mathbf{x}^\prime)$ as in \cite{neal1996priors}. This covariance function is often difficult to compute, and depends on the specified activation function. See [Comments on the Covariance Function](#comments-on-the-covariance-function) for further discussion.


#### Deep Neural Networks as Gaussian Processes


The case for neural networks with arbitrary depth can be extended via an argument of mathematical induction. First, the base case follows from the previous section. Then, we assume that $Z_j^{(l)} \sim \mathcal{GP}(0, k^{(l)})$[^fn3] , and that each $Z_j^{(l)}$ are i.i.d. Hence, each $A_j^{(l)}$ are i.i.d. as well. Then, we can compute

$$
Z_j^{(l+1)} = b_j^{l} + \sum_{i=1}^{n_l}w_{ji}^{(l)}A_j^{(l)}.
$$

Once again, since each $A_j^{(l)}$ are i.i.d., then as $n_l \to \infty$, the CLT implies that $Z_j^{(l+1)}$ will take a Normal distribution. Thus, $Z_j^{(l+1)}$ is also a GP, and we have our result.

Specifically, the mean function of the corresponding GP is given by $m(\mathbf(x)) = \mathbb{E} \left[ Z_j^{(l+1)} \right]$, and since the weights and biases have mean zero, then $m(\mathbf{x}) = 0$. Then, we have

$$
Z_j^{(l+1)} \sim \mathcal{GP}(0, k^{(l+1)}),
$$

where

$$
\begin{align*}
k^{(l+1)}(\mathbf{x}, \mathbf{x}^\prime) &= \mathbb{E} \left[ Z_j^{(l+1)} (\mathbf{x}) \\, Z_j^{(l+1)}(\mathbf{x}^\prime) \right] \\\\
&= 1 + \frac{1}{n_l} \\, C\left( A^{(l)}(\mathbf{x}) A^{(l)}(\mathbf{x}^\prime) \right).
\end{align*}
$$

The covariance function of the corresponding GP at each layer is defined recursively, and I restrict discussion of the covariance function to the single-layer case. This is addressed in the following section.


#### Comments on the Covariance Function


As was previously alluded to the computation of the covariance function is often difficult to evaluate, and depends on the specifica architecture and choice of activation functions in the neural network. Computin the covariance function involves integrating over the distributions of the weights and biases for each pair of training samples. For many architectures, this requires sophisticated numerical integration techniques, and is often not practical from a computational perspective. Furthermore, this becomes increasingly challenging with larger datasets.

However, in the case of a single hidden layer, certain choices of activation functions do yield analytic covariance functions. \cite{williams1996computing} gives on such example; the "error function", defined by

$$
\text{erf } x = \frac{2}{\sqrt{\pi}} \int_0^x e^{-t^2}dt.
$$

The error function is related to the cumulative distribution function for the Gaussian distribution. Furthermore, it closely resembles the tanh function, making it a reasonable choice for an activation function in a neural network. The corresponding covariance function is then given by

$$
\mathbb{E} \left[ \text{erf } (\mathbf{x}) \\, \text{erf } (\mathbf{x}^\prime) \right] = \frac{2}{\sqrt{\pi}} \sin^{-1} \left( \frac{2\mathbf{x}^\top \Sigma \mathbf{x}^\prime}{\sqrt{ \left( 1 + 2\mathbf{x}^\top \Sigma \mathbf{x} \right)  \left( 1 + 2 \mathbf{x}^{\prime\top}\Sigma \mathbf{x}^\prime \right) } } \right),
$$

where $\Sigma$ denotes the covariance matrix of the input-to-hidden layer weights (\cite{williams1996computing}). Note that this covariance function is not stationary, i.e., it is not translation invariant, which is often a nice property in kernel functions for GP regression.


#### Experiments

In this section, I produce some empirical results to help visualize and understand the NNGP. 


![fig:bivariate_distributions](figures/bivariate_distributions.png)


#### References

1. [Visual exploration of Gaussian Processes](https://distill.pub/2019/visual-exploration-gaussian-processes/)
2. [Some math behind the NTK](https://lilianweng.github.io/posts/2022-09-08-ntk/)


[^fn1]: This could benefit from more explanation of what is meant by "distribution of functions represented by NNs...
[^fn2]: This follows from the fact that the linear combination of i.i.d. Normal random variables is itself i.i.d. Normal.
[^fn3]: Note that the covariance function of the GP for the units in each layer depends on the activation functions in all the previous layers.