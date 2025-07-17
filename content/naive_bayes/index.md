---
title: "Naive Bayes"
draft: true
math: true
comments: false
---

First, let's establish the Naive Bayes classification problem.

Supppose we are given a dataset of $n$ iid observations $\cD = \\{\(\x^{(i)}, y_i)\\}_{i=1}^n$, where each $\x^{(i)} \in \R^d$ is a $d$-dimensional input vector and $y_i \in \\{1, 2, \ldots, K\\}$ is the corresponding class label. Given a new input $\x$, the goal of classification is to correctly assign $\x$ to one of $K$ classes.

Let's distringuish between two approaches to classification: *discriminative* vs. *generative* modeling.

In the **discriminative modeling** approach, we wish to model the distribution over class labels $p(y \mid \x)$. Thus, given an input $\x$, we should be able to assign probabilities to each of the $K$ classes, and choose the $y$ which maximizes this probabilitiy.

In the **generative modeling** approach, we wish to model the sample generating process by estimating each class-conditional probability density. That is, using Bayes' rule, we can write

$$
\begin{equation}\label{1}
p(y \mid \x) \propto p(\x \mid y)p(y).
\end{equation}
$$

We see that this achieves the same result as the discriminative modeling approach, but requires a bit more work - instead of skipping straight to the step of modeling the class probabilities given an input, we aim to model $p(\x \mid y)$ for each class $K$. Moreover, we also wish to estimate the prior class probabilities $p(y)$. One might ask: why bother? Well, in doing so, we gain the ability to generate new data points by first sampling $y \sim p(y)$, then sampling $x \sim p(\x \mid y)$.

Naive Bayes takes a generative modeling approach to classification.

There are many ways to estimate the densities $p(\x \mid y)$ and $p(y)$. One such method is maximum likelihood estimation (MLE), where we assume a particular parametric form of each distribution---for example, $\x$ might be drawn from a Gaussian distribution, with parameters $\bmu_k$ and $\bSigma_k$, and $y$ might be drawn from a multinomial distribution with $K$ classes, with probabilities $p(y = k) = \pi_k$. Then, we could find the parameters which maximize the likelihood of the observed dataset and use these to estimate the probability densities. However, for the sake of understanding Naive Bayes, I won't focus on density-estimation techniques here.

As I previously said, our goal is to find $y$, given an input $\x$, which maximizes $p(y \mid \x)$. That is, we wish to find

$$
\begin{align*}
\hat{y} &\in \argmax{y} p(y \mid \x) \\\\
&= 
\end{align*}
$$


$$
p(\x \mid y) = p(x_1 \mid y) \cdots p(x_d \mid y)
$$

However, if some $x_i$ never existed in the training set, this will be 0. This could be a problem (see spam vs. ham example)