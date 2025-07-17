---
title: "Naive Bayes"
draft: true
math: true
comments: false
---

First, let's establish the Naive Bayes classification problem.

Supppose we are given a dataset of $n$ iid observations $\cD = \\{\(\x^{(i)}, y_i)\\}_{i=1}^n$, where each $\x^{(i)}$ is some $d$-dimensional input vector and $y_i \in \\{1, 2, \ldots, K\\}$ is the corresponding class label. Given a new input $\x$, the goal of classification is to correctly assign $\x$ to one of $K$ classes.

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

Now, let's assume that the features of $\x$ are binary. Then, we can write the joint distribution over the features of $\x$, conditioned on $y$ as

$$
\begin{equation}\label{2}
p(\x \mid y) = p(x_1, x_2, \ldots, x_d \mid y),
\end{equation}
$$

where each $x_i \in \\{0, 1\\}$. Then, if we want to estimate the probability density of $\x$ being in a particular class $y$, we would need to assign a probability to each possible configuration of $\x$ --- there are $2^d$ possible such configurations! This requires estimating $2^d - 1$ parameters. [^fn1]  In order to gain an accurate estimation for such a distribution, we would need **a lot of data** ---  this is an example of the *curse of dimensionality*. Moreover, this is just for a single class $y$.

This calls for a need to reduce the dimensionality of our parameter space.


**What is conditional independence?**

The "naive" assumption in Naive Bayes is that the features of our inputs are conditionally independent. Under this assumption, we can factor the joint distribution in $\eqref{2}$ as

$$
p(x_1, \ldots, x_d \mid y) = p(x_1 \mid y) \cdots p(x_2 \mid y).
$$

One way to say this is "$x_1, x_2, \ldots, x_d$ are independent, conditioned on $y$." Now, we can model each individual distribution $p(x_i \mid y)$ independently. In the case of binary features, this only requires $d$ parameters for each class instead of $2^d-1$.

As an example, consider three random variables $a, b, c \in \R$.


**When does Naive Bayes work?**

From here, it's natural to ask the question "when does Naive Bayes work?" Let's look at some different examples.


**Example:** Spam vs. ham


**Example:** Medical diagnosis



[^fn1]: Estimating a discrete probability distribution with $n$ possible inputs only requires $n-1$ parameters due to the summation constraint:
$$
\sum_{i=1}^n p_i = 1,
$$
where $p_i$ is the probability of the $i^{\text{th}}$ event. The $n^{\text{th}}$ probability is defined implicitly as $1 - \sum_{i=1}^{n-1}p_i$.