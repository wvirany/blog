---
title: "MLE vs. Bayesian inference for the Gaussian distribution"  
date: "2025-05-25"  
summary: ""  
description: ""  
draft: true
toc: true  
readTime: true  
autonumber: false  
math: true  
tags: []
showTags: false  
hideBackToTop: false
---


<!-- #### TO DO:
- Comment on how I'm trying to keep this self-contained?
- FIX RESULT FOR LINEAR GAUSSIAN SYSTEM: conditional mean is wrong
- Double-check results for Bayesian inference involving parameters of Gamma in normal-gamma distributions. See exercise 3.12 for procedure on how to derive this
- Fix formatting
- Coherence between appendix and main blog
- Links to examples and include them in TOC?
- Equation numbers for results from section on Bayes' rule for linear Gaussian systems
- Change TOC: maybe use same style as example boxes
- Fix equation numbers
- Examples
- Fill in appendix, proofs
- Discussion of Mahalanobis distance, Z-score, reconcile with example 1
- Discussion of conjugate priors, what they are and why we want to use them (because it makes finding an analytical form of the posterior easy)
- Comment about our frequent strategy of removing constants from sums in the exponent because they just become multiplicative scalars, which we account for add the end by normalizing
- Interpreting the moments!
- Double-check all results
- Tops of partials cut off (e.g., jacobian, matrix derivative rules) (and some fractions! e.g., see posteriors of univariate Gaussian)
- Conditioning: Add the other forms of the conditional params?
- Add all the results at the top for quick reference, or maybe just a nice TOC
- In "moments": show the formula for the second moment of a univariate Gaussian, and reference it properly (whether it's a footnote or appendix)
- Sum and product rule of probability in appendix + references to them
- Proper way to reference derivative rules for MLE section
 -->


In my [previous post](../gaussian1/), I developed several key results involving the Gaussian distribution. Now, I'll show two methods for estimating the parameters of a Gaussian distribution from data; maximum likelihood estimation (MLE) and Bayesian inference.



## Maximum likelihood estimation

In practice, we rarely know the true underlying distribution from which data is generated. The goal of maximum likelihood estimation (MLE) is to estimate the true parameters of a distribution from an observed set of data. Suppose we have $n$ i.i.d. samples from a Gaussian, $\X = (\x_1, \x_2 \ldots, \x_n)$, and we would like to estimate the mean and covariance. This is done by maximizing the likelihood function

$$
\begin{align*}
p(\X \mid \bmu, \Sigma) &= p(\x_1, \ldots, \x_n \mid \bmu, \Sigma) \\\\
&= \prod_{i=1}^n p(\x_i \mid \bmu, \Sigma),
\end{align*}
$$

where we can factorize the joint distribution as a product due to the assumption of independence. It can instead be easier to work with the log-likelihood, as it's easier to optimize a sum of functions as opposed to a product.[^fn6] Furthermore, we often choose to minimize the negative log-likelihood (NLL), as opposed to maximizing the log-likelihood, as this is often treated as a sort of loss function, and many modern optimization frameworks for machine learning are designed to minimize a loss objective. So, we express the NLL as

$$
-\log p(\X \mid \bmu, \Sigma) = - \sum_{i=1}^n \log p(\x_i \mid \bmu, \Sigma)
$$

Now, since each sample $\x_i$ takes a Gaussian, the expression for the likelihood of a single sample is given by

$$
p(\x_i \mid \bmu, \Sigma) = \frac{1}{(2\pi)^{d/2} \lvert \Sigma \rvert^{1/2}} \exp \left( -\frac{1}{2} (\x_i - \bmu)\T\Sigma\inv(\x_i - \bmu) \right),
$$

hence, the NLL is

$$
\begin{align*}
\nll(\bmu, \Sigma) &= - \sum_{i=1}^n \log \bigg[ \frac{1}{(2\pi)^{d/2}\lvert \Sigma \rvert^{1/2}} \exp \left(-\frac{1}{2} (\x_i - \bmu)\T\Sigma\inv(\x_i - \bmu) \right) \bigg] \\\\
&= - \sum_{i=1}^n \log \frac{1}{(2\pi)^{d/2}\lvert \Sigma \rvert^{1/2}} - \sum_{i=1}^n -\frac{1}{2} (\x_i - \bmu)\T\Sigma\inv(\x_i - \bmu) \\\\
&= \frac{nd}{2} \log 2\pi + \frac{n}{2} \lvert \Sigma \rvert + \frac{1}{2} \sum_{i=1}^n (\x_i - \bmu)\T \Sigma\inv (\x_i - \bmu).
\end{align*}
$$

Then, our goal is to minimize this with respect to the parameters, treating the data as a fixed quantity. First, let's minimize the NLL with respect to $\bmu$:

$$
\begin{align*}
\pbmu \nll(\bmu, \Sigma) &= \frac{1}{2} \sum_{i=1}^n \pbmu (\x_i - \bmu)\T \Sigma\inv (\x_i - \bmu) \\\\
&= - \sum_{i=1}^n \Sigma\inv(\x_i - \bmu),
\end{align*}
$$

where I've used the fact that

$$
\frac{\partial}{\partial\x} \bigg( \x\T\A\x \bigg) = 2\A\x.
$$

Then, setting this equal to zero and solving for $\bmu$:

$$
\begin{gather*}
\sum_{i=1}^n \Sigma\inv (\x_i - \bmu) = 0 \\\\
\Sigma\inv \sum_{i=1}^n \x_i = n\Sigma\inv\bmu \\\\
\bmu = \frac{1}{n} \sum_{i=1}^n \x_i.
\end{gather*}
$$

Thus, we see that the maximum likelihood estimator for $\bmu$ is just the average of the observed data:

$$
\bmu\ml = \frac{1}{n} \sum_{i=1}^n \x_i.
$$

Taking the derivative with respect to $\Sigma$ is a bit more involved. To do so, I'll make use of the following three rules, each of which I prove in the [appendix](#matrix-derivatives):

$$
\begin{gather}
\pbx \A\inv = - \A\inv \frac{\partial\A}{\partial\x} \A\inv \label{eq:deriv1} \\\\[2pt]
\frac{\partial}{\partial\A} \tr(\A\B) = \B\T \label{eq:deriv2} \\\\[5pt]
\frac{\partial}{\partial\A} \log \lvert \A \rvert = (\A\inv)\T. \label{eq:deriv3}
\end{gather}
$$

Now, we take the derivative of the NLL with respect to $\Sigma$:

$$
\begin{equation}
\pSigma \nll(\bmu, \Sigma) = \frac{n}{2} \pSigma \log \lvert \Sigma \rvert + \frac{1}{2} \pSigma \sum_{i=1}^n (\x_i - \bmu)\T \Sigma\inv (\x_i - \bmu).
\end{equation}
$$

Using the third derivative rule, the first term can be written as

$$
\begin{align*}
\frac{n}{2} \pSigma \log \lvert \Sigma \rvert &= \frac{n}{2} \left( \Sigma\inv \right)\T \\\\
&= \frac{n}{2} \Sigma\inv,
\end{align*}
$$

recalling that $\Sigma\inv$ is symmetric. Now, we will rewrite the second term using the following property of the trace:

$$
\tr \left( \a\b\T \right) = \b\T\a.
$$

That is, the trace of the outerproduct between two vectors $\a$ and $\b$ is just their inner product. Using this, we can write the sum in the second term as

$$
\begin{align*}
\sum_{i=1}^n (\x_i - \bmu)\T \Sigma\inv (\x_i - \bmu) &= \tr \left( \sum_{i=1}^n \Sigma\inv (\x_i - \bmu) (\x_i - \bmu)\T \right) \\\\
&= n \tr \left( \Sigma\inv \bS \right),
\end{align*}
$$

where I've introduced the *scatter matrix*:

$$
\bS = \frac{1}{n} \sum_{i=1}^n (\x_i - \bmu)(\x_i - \bmu)\T.
$$

Then, the second term in $(17)$ can be written as

$$
\begin{align*}
\frac{1}{2} \pSigma \sum_{i=1}^n (\x_i - \bmu)\T \Sigma\inv (\x_i - \bmu) &= \frac{n}{2} \pSigma \tr(\Sigma\inv \bS) \\\\
&= \frac{n}{2} \tr \left( \frac{\partial\Sigma\inv}{\partial\Sigma} \bS \right) \\\\
&= - \frac{n}{2} \tr \left( \Sigma\inv \frac{\partial\Sigma}{\partial\Sigma} \Sigma\inv \bS \right) \\\\
&= - \frac{n}{2} \tr \left( \frac{\partial\Sigma}{\partial\Sigma} \Sigma\inv \bS \Sigma\inv \right) \\\\
&= - \frac{n}{2} \left( \Sigma\inv\bS\Sigma\inv \right)\T \\\\
&= - \frac{n}{2} \Sigma\inv\bS\Sigma\inv,
\end{align*}
$$

where I've used the first derivative rule, along with the cyclic property of the trace and the fact that $\bS$ is symmetric.[^fn7] Then, substituting the derivatives on the right-hand side of $(17)$ and setting them equal to zero:

$$
\begin{gather*}
\frac{n}{2}\Sigma\inv - \frac{n}{2} \Sigma\inv\bS\Sigma\inv = 0 \\\\
\Sigma\inv = \Sigma\inv\bS\Sigma\inv \\\\
\I = \bS \Sigma\inv \\\\
\end{gather*}
$$

Hence, the maximum likelihood estimator for $\Sigma$ is given by

$$
\Sigma\ml = \frac{1}{n} \sum_{i=1}^n (\x_i - \bmu\ml)(\x_i - \bmu\ml)\T.
$$

It's important to note that the estimator we use for $\Sigma$ need be symmetric PSD. Otherwise, it would not be a valid covariance matrix. However, we see that $\bS$ is indeed symmetric PSD.

Often when estimating parameters from data, we would like to know whether or not our estimators are *biased*. That is, do the estimators equal the true parameters in expectation? Let's first examine this for the mean:

$$
\begin{align*}
\E[\bmu\ml] &= \E \left[ \frac{1}{n} \sum_{i=}^n \x_i \right] 
= \frac{1}{n} \sum_{i=1}^n \E [\x_i] 
= \frac{n\bmu}{n} 
= \bmu.
\end{align*}
$$

Thus, the maximum likelihood estimator for the mean is unbiased. Again, examining the expectation of $\Sigma\ml$ is more involved. To evaluate this, I'll use the fact that, for two observations $\x_i, \x_j$, we have

$$
\begin{equation}\label{eq:xixj}
\E[\x_i\x_j] = \bmu\bmu\T + \delta_{ij}\Sigma.
\end{equation}
$$

This follows from $(5)$, noting that the covariance between two independent samples will be zero. Then,

$$
\begin{align}
&\E[\Sigma\ml] = \frac{1}{n} \E \left[ \sum_{i=1}^n (\x_i - \bmu\ml)(\x_i - \bmu\ml)\T \right] \nonumber \\\\
&= \frac{1}{n} \E \left[ \sum_{i=1}^n \left( \x_i\x_i\T - \x_i\bmu\ml\T - \bmu\ml\x_i\T + \bmu\ml\bmu\ml\T \right) \right] \nonumber \\\\
&= \frac{1}{n} \bigg( \sum_{i=1}^n \E [\x_i\x_i\T] - \sum_{i=1}^n \E[\x_i \bmu\ml\T] - \sum_{i=1}^n \E [\bmu\ml\x_i\T] + \sum_{i=1}^n \E [\bmu\ml\bmu\ml\T]  \bigg). \nonumber \\\\
\end{align}
$$

The first term is given by

$$
\frac{1}{n} \sum_{i=1}^n \bmu\bmu\T + \Sigma,
$$

from $\eqref{eq:xixj}$.



## Bayesian inference

Using MLE to estimate the parameters of a distribution has a drawback: it gives *point estimates* of the parameters. Instead, suppose we want a range of values to choose from, each with a corresponding level of (un)certainty. This can be achieved using Bayesian inference, in which we find a probability distribution over the possible parameter values.

In this section, I'll show how to estimate the parameters of the Gaussian using Bayesian inference. I'll cover several scenarios:

1. Unknown mean, known variance
2. Known mean, unknown variance
3. Unknown mean, unknown variance

The goal in each scenario will be to infer the unknown parameter(s). First, I'll show the univariate case of each scenario, then I'll extend them to the multivariate case. 

### Unknown mean, known variance (univariate case)

Similar to the MLE framework, suppose we have some dataset $\X = \\{x_i\\}_{i=1}^n$, in which $x_i \sim \Norm(x_i \mid \mu, \sigma^2)$, where $\sigma^2$ is known, and we wish to infer $\mu$. However, as opposed to the MLE framework, we wish to find a probability distribution over $\mu$ given the observed data.

Using Bayes' theorem, we have

$$
p(\mu \mid \X) = \frac{p(\X \mid \mu) p(\mu)}{p(\X)},
$$

where $p(\X \mid \mu)$ is the likelihood (which we saw in the MLE framework), $p(\mu)$ is the prior, and $p(\mu \mid \X)$ is the posterior. $p(\X)$ is a constant with respect to $\mu$, and normalizes the product in the numerator to ensure that $p(\mu \mid \X)$ is a valid probability distribution. Instead of computing it explicitly, we will find the functional form of the posterior, then normalize at the end.[^fn8] Then, disregarding normalization, we have

$$
p(\mu \mid \X) \propto p(\X\mid\mu) p(\mu).
$$

The likelihood takes the form

$$
\begin{align}
p(\X \mid \mu) &= \prod_{i=1}^n p(x_i \mid \mu) \nonumber \\\\
&= \prod_{i=1}^n \frac{1}{(2\pi\sigma^2)^{1/2}} \exp \left( -\frac{(x_i - \mu)^2}{2\sigma^2} \right) \nonumber \\\\
&= \frac{1}{(2\pi\sigma^2)^{n/2}} \exp \left( -\frac{1}{2\sigma^2} \sum_{i=1}^n (x_i - \mu)^2  \right) \nonumber.
\end{align}
$$

Note that the likelihood function is not a valid probability distribution, since it is not normalized. However, we do see that it takes the form of an exponential of a quadratic function of $\mu$, hence it takes the form of a Gaussian distribution over $\mu$. Thus, if we choose the prior distribution to be a Gaussian as well, then the posterior will be the product of two Gaussians, which is itself a Gaussian. Thus, the *conjugate prior* of the Gaussian distribution is itself a Gaussian.[^fn9]

Then, let's choose the prior to be

$$
p(\mu) = \Norm(\mu \mid \mu_0, \sigma^2_0).
$$

The posterior then takes the form

$$
\begin{align}
p(\mu \mid \X) &\propto p(\X \mid \mu) p(\mu) \nonumber \\\\
&\propto \exp \left( -\frac{1}{2\sigma^2} \sum_{i=1}^n (x_i - \mu)^2 \right) \exp \left( -\frac{1}{2\sigma_0^2} (\mu - \mu_0)^2 \right) \nonumber \\\\
&\propto \exp\left(-\frac{1}{2} \sum_{i=1}^n \frac{(x_i - \mu)^2}{\sigma^2} + \frac{(\mu - \mu_0)^2}{n\sigma_0^2} \right). \nonumber
\end{align}
$$

The sum inside the exponential is then

$$
\begin{align*}
&\sum_{i=1}^n \frac{n\sigma_0^2 \left( x_i^2 - 2x_i\mu + \mu^2  \right) + \sigma^2 \left( \mu^2 - 2\mu\mu_0 + \mu_0^2 \right)}{n\sigma_0^2\sigma^2} \\\\
&= \frac{1}{n\sigma_0^2\sigma^2} \sum_{i=1}^n n\sigma_0^2x_i^2 - 2n\sigma_0^2x_i\mu + n\sigma_0^2\mu^2 + \sigma^2\mu^2 - 2\sigma^2\mu\mu_0 + \sigma^2\mu_0^2 \\\\
&= \frac{1}{n\sigma_0^2\sigma^2} \bigg( n^2\sigma_0^2\mu^2 + n\sigma^2\mu^2 - 2n\sigma^2\mu\mu_0 + n\sigma^2\mu_0^2 - 2n\sigma_0^2\mu \sum_{i=1}^n x_i  + n\sigma_0^2\sum_{i=1}^n x_i^2 \bigg) \\\\[3pt]
&= \frac{1}{n\sigma_0^2\sigma^2} \left[ (n^2\sigma_0^2 + n\sigma^2)\mu^2 - 2n\left(\sigma^2\mu_0 + \sigma_0^2\sum_{i=1}^n x_i\right) \mu + c  \right],
\end{align*}
$$

where $c$ denotes all the terms which are independent of $\mu$. Then, we can neglect these terms, since they can be factored out of the exponent and considered part of the normalization constant which we'll determine later. Thus, we are left with

$$
\frac{1}{n\sigma_0^2\sigma^2} \bigg[ (n^2\sigma_0^2 + n\sigma^2)\mu^2 - 2n\left(\sigma^2\mu_0 + n\sigma_0^2\mu\ml\right) \mu \bigg],
$$

recalling that $\mu\ml = \frac{1}{n}\sum_{i=1}^n x_i$. Then, completing the square and putting this expression back into the exponent, we have

$$
p(\mu \mid \X) \propto \exp \left( -\frac{1}{2} \frac{n\sigma_0^2 + \sigma^2}{\sigma_0^2\sigma^2} \left( \mu - \frac{n\sigma_0^2\mu\ml + \sigma^2\mu_0}{n\sigma_0^2 + \sigma^2} \right)^2 \right).
$$

As we predicted, this takes the form of a Gaussian:

$$
\mu \sim \Norm(\mu \mid \mu_n, \sigma^2_n),
$$

where

$$
\begin{align}
\mu_n &= \frac{\sigma^2}{n\sigma_0^2 + \sigma^2}\mu_0 + \frac{n\sigma_0^2}{n\sigma_0^2 + \sigma^2} \mu\ml = \frac{1}{\sigma_n^2} \left( \frac{1}{\sigma_0^2} \mu_0 + \frac{n}{\sigma^2} \mu\ml \right), \label{eq:poster-mean-uv-1} \\\\[10pt]
\frac{1}{\sigma_n^2} &= \frac{n\sigma_0^2 + \sigma^2}{\sigma_0^2\sigma^2} = \frac{n}{\sigma^2} + \frac{1}{\sigma_0^2}. \label{eq:poster-cov-uv-1}
\end{align}
$$

To recap, we started with some initial belief about the parameter $\mu$, which was represented by the prior $p(\mu)$. For example, the prior mean $\mu_0$ might reflect our belief of a reasonable value for $\mu$, and the prior variance $\sigma_0^2$ might reflect how certain we are in that belief. Then, after observing $n$ samples, we updated our expressions for these parameters using $\eqref{eq:poster-mean-uv-1}$ and $\eqref{eq:poster-cov-uv-1}$.

There are several interesting things to note about $\eqref{eq:poster-mean-uv-1}$ and $\eqref{eq:poster-cov-uv-1}$. First, we see that the posterior mean $\mu_n$ is a weighted average between the prior mean $\mu_0$ and the mle $\mu\ml$. When $n=0$, we simply have the prior mean $\mu_n = \mu_0$. As $n \to \infty$, the posterior mean approaches the mle $\mu_n \to \mu\ml$. 

Furthermore, we have expressed the posterior variance in terms of the precision:

$$
\lambda_n = \frac{1}{\sigma_n^2}.
$$

We see that, as $n$ gets big, the precision grows, and hence the variance gets small. That is, with more observations, we become more certain in our estimation of the parameters. Moreover, it's interesting to note that the precision gets updated additively. For each observation of the data, we increase prior precision $1/\sigma_0^2$ by one increment of the data precision $1/\sigma^2$.

Finally, it's interesting to note that as $n \to \infty$, not only does $\mu_n \to \mu\ml$, but $\sigma_n^2 \to 0$ as well. Thus, the posterior distribution in the limit of infinitely many observations is the delta function centered at $\mu\ml$. Thus, the maximum likelihood estimator is recovered in the Bayesian formulation.


### Unknown mean, known variance (multivariate case)

Here, I extend the previous results to the multivariate case. Now suppose we have a dataset $\X = \\{\x_i\\}_{i=1}^n$, where $\x_i \sim \Norm(\x_i \mid \bmu, \Sigma)$ with $\x_i \in \R^d$. Again, we'll assume that $\Sigma$ is known and we wish to find the posterior distribution $p(\bmu \mid \X)$.

Suppose the prior is given by $p(\bmu) = \Norm(\bmu \mid \bmu_0, \Sigma_0)$. Then, the posterior will take the form

$$
\begin{align*}
&p(\bmu \mid \X) \propto p(\X \mid \bmu) p(\bmu) \\\\
&= p(\bmu) \prod_{i=1}^n p(\x_i \mid \bmu) \\\\
&\propto \exp\left(-\frac{1}{2} (\bmu - \bmu_0)\T\Sigma_0\inv(\bmu - \bmu_0) \right) \exp \left( -\frac{1}{2}\sum_{i=1}^n (\x_i - \bmu)\T \Sigma\inv (\x_i - \bmu) \right) \\\\
&= \exp \left[ -\frac{1}{2} \left( (\bmu - \bmu_0)\T\Sigma_0\inv(\bmu-\bmu_0) + \sum_{i=1}^n (\x_i - \bmu)\T\Sigma\inv(\x_i - \bmu) \right) \right].
\end{align*}
$$

We see the expression in the exponent is quadratic in terms of $\bmu$, hence the posterior will again be a Gaussian. Examining the quadratic:

$$
\begin{align*}
&-\frac{1}{2} \left[ (\bmu - \bmu_0)\T\Sigma_0\inv(\bmu-\bmu_0) + \sum_{i=1}^n (\x_i - \bmu)\T\Sigma\inv(\x_i - \bmu) \right] \\\\
&\qquad= -\frac{1}{2} \bigg[ \bmu\T\Sigma_0\inv\bmu - 2\bmu\T\Sigma_0\inv\bmu_0 + \bmu_0\T\Sigma_0\inv\bmu_0 + n\bmu\T\Sigma\inv\bmu \\\ 
&\qquad\qquad + \sum_{i=1}^n \big( \x_i\T\Sigma\inv\x_i - 2\bmu\T\Sigma\inv\x_i \big) \bigg] \\\\
&\qquad= -\frac{1}{2} \bigg[ \bmu\T \left( \Sigma_0\inv + n\Sigma\inv \right) \bmu + -2 \bmu\T \left( \Sigma_0\inv\bmu_0 + \Sigma\inv \sum_{i=1}^n \x_i \right) \\\\
&\qquad\qquad + \bmu_0\T\Sigma\inv\bmu_0 + \sum_{i=1}^n \x_i\T\Sigma\inv\x_i \bigg] \\\\
&\qquad= -\frac{1}{2} \bmu\T \left( \Sigma_0\inv + n\Sigma\inv \right) + \bmu\T \left( \Sigma_0\inv\bmu_0 +n\Sigma\inv\bmu\ml \right) + c.
\end{align*}
$$

Recalling the general form of the Gaussian from $\eqref{eq:generalgaussian}$, we see that the precision of the posterior distribution is

$$
\begin{equation}\label{eq:posterior-precision-mv-1}
\Sigma_n\inv = n\Sigma\inv + \Sigma_0\inv.
\end{equation}
$$

Similarly, the linear term is

$$
\Sigma_n\inv\bmu_n = \Sigma_0\inv\bmu_0 + n\Sigma\inv\bmu\ml,
$$

hence the posterior mean is given by

$$
\begin{align}\label{eq:posterior-mean-mv-1}
\bmu_n &= \Sigma_n \Sigma_0\inv \bmu_0 + n\Sigma_n\Sigma\inv\bmu\ml.
\end{align}
$$

Comparing these expressions to $\eqref{eq:poster-mean-uv-1}$ and $\eqref{eq:poster-cov-uv-1}$, it's fairly straightforward to see the parallels.


### Known mean, unknown variance (univariate case)

Previously, we assumed that the variance of the data distribution was known, and we found the posterior distribution for the mean. Instead, let's now suppose that we know the mean of the data distribution, and we wish to infer the variance. We will instead work with the precision $\lambda = 1/\sigma^2$, as it simplifies the derivation.

The likelihood is now

$$
\begin{align*}
p(\X \mid \lambda) &= \prod_{i=1}^n \Norm (x_i \mid \mu, \lambda\inv) \\\\
&\propto \lambda^{n/2} \exp \left( -\frac{\lambda}{2} \sum_{i=1}^n (x_i - \mu)^2 \right) \\\\
&= \lambda^{n/2} \exp \left( -\frac{\sigma^2\ml}{2} \lambda \right),
\end{align*}
$$

recalling the expression for $\sigma^2\ml$.[^fn10] We see that the likelihood has the form of the product of a power of $\lambda$ and an exponential of a linear function of $\lambda$. This is the same form as the Gamma distribution:

$$
\gam(\lambda \mid a, b) = \frac{b^a}{\Gamma(a)} \lambda^{a-1} \exp(-b\lambda).
$$

So, the if we take the prior to be

$$
p(\lambda) = \gam(\lambda \mid a_0, b_0).
$$

Then, the posterior has the form

$$
\begin{align*}
p(\lambda \mid \X) &\propto p(\X \mid \lambda) p(\lambda) \\\\
&= \lambda^{a_0-1}\lambda^{n/2} \exp \left( -\frac{\sigma^2\ml}{2} \lambda \right)
\end{align*}
$$


### Known mean, unknown variance (multivariate case)


### Unknown mean, unknown variance (univariate case)


### Unknown mean, unknown variance (multivariate case)



Suppose we have some dataset $\X = \\{x_i\\}_{i=1}^n$, where each $x_i$ is sampled i.i.d. from a Gaussian distribution with mean $\mu$ and variance $\sigma^2$. Suppose $\mu$ is known, and we wish to infer the posterior distribution of $\sigma^2$. Instead, we'll work with the precision: $\lambda = 1/\sigma^2$. Then, by Bayes' theorem, we have

$$
\begin{align*}
p(\lambda \mid \X) \propto p(\X \mid \lambda) p(\lambda),
\end{align*}
$$

where $p(\X\mid\lambda)$ is the likelihood function, and $p(\lambda)$ is the prior distribution over $\lambda$. Then, the likelihood takes the form

$$
\begin{align*}
p(\X \mid \lambda) &= \prod_{i=1}^n p(x_i \mid \lambda) \\\\
&= \prod_{i=1}^n \Norm (x_i \mid \mu, \lambda\inv) \\\\
&\propto \lambda^{n/2} \exp \left( -\frac{\lambda}{2} \sum_{i=1}^n (x_i - \mu)^2 \right). \tag{1}
\end{align*}
$$

We see this takes the form of a Gamma distribution:

$$
\Gamma(\lambda \mid a, b) = \frac{b^a}{\Gamma(a)}\lambda^{a-1}\exp(-b\lambda).
$$

Thus, the corresponding conjugate prior will also take a Gamma distribution:

$$
p(\lambda) = \gam(\lambda \mid a_0, b_0).
$$

Then, the posterior takes the form

$$
\begin{align*}
p(\lambda \mid \X) &\propto \lambda^{n/2}\lambda^{a_0 - 1} \exp \left(-\frac{\lambda}{2} \sum_{i=1}^n (x_i - \mu)^2 \right) \exp(-b_0\lambda) \\\\
&= \lambda^{a_0 + n/2 -1} \exp \left( - \left( b_0 + \frac{1}{2} \sum_{i=1}^n (x_i - \mu)^2 \right) \lambda \right).
\end{align*}
$$

Thus,

$$
p(\lambda \mid \X) = \gam(\lambda \mid a_n, b_n),
$$

where

$$
\begin{align*}
a_n &= a_0 + \frac{n}{2} \tag{2} \\\\
b_n &= b_0 + \frac{1}{2} \sum_{i=1}^n (x_i - \mu)^2 = b_0 + \frac{n}{2}\sigma^2\ml, \tag{3}
\end{align*}
$$

where $\sigma^2\ml$ is the maximum likelihood estimator of the variance.


$$
\sigma^2\ml = \frac{1}{n}\sum_{i=1}^n (x_i - \mu\ml)^2
$$



## Appendix

### Matrix derivatives









## Bayesian linear regression

Here I'll show how to use our results to perform Bayesian linear regression. Consider the linear model

$$
\begin{align*}
f(\x; \w) = \w\T\x,
\end{align*}
$$

where $\x \in \R^d$ is some input vector with $d$ features and $\w \in \R^d$ is the vector of parameters which specify the model. Note that we can incorporate an intercept term by always letting one element of $\x$ be constant, say $x_0 = 1$:

$$
f(\x; \w) = w_0 + w_1x_1 + \cdots + w_{d-1}x_{d-1}.
$$

Then, we can model noisy predictions by adding a zero-mean Gaussian random variable to the functional outputs:

$$
y = f(\x; \w) + \epsilon,
$$

where $\epsilon \sim \Norm(0, \beta\inv)$, with precision $\beta$. This gives rise to a probability distribution over $y$:

$$
\begin{align*}
p(y \mid \x, \w, \beta) = \Norm(y \mid f(\x; \w), \beta\inv).
\end{align*}
$$

Then, suppose we observe a dataset $(\X, \y)$, where $\X \in \R^{d\times n}$ is the matrix whose columns are the $n$ observed input vectors, and $\y = (y_1, y_2 ,\ldots, y_n)\T$ contains the correspondng targets. In light of these observations, we would like to do the following:

1. Update our belief about the parameters $\w$ by modeling the posterior parameter distribution $p(\w \mid \X, \y)$
2. Given a new test point $(\x_\*, y_\*)$, use the posterior parameter distribution to model the distribution over $y_\*$: $p(y_\* \mid \x_\*, \X, \y).$

First, we'll compute the posterior parameter distribution. To do so, recall Bayes' rule:

$$
\begin{align*}
p(\w \mid \cD) = \frac{p(\cD \mid \w)}{p(\w)}{p()}
\end{align*}
$$


<details>
  <summary>Matrix derivatives</summary>
  <p>
  Here I prove the three matrix derivative rules $\eqref{eq:dervi1}, \eqref{eq:deriv2}$, and $\eqref{eq:deriv3}$, which are used in deriving the maximum likelihood estimator for the covariance matrix.

  **Proof of $\eqref{eq:deriv1}$:**

  We wish to show that 

  $$
  \pbx\A\inv = -\A\inv\frac{\partial\A}{\partial\x}\A\inv.
  $$

  From the product rule, we can write

  $$
  \pbx \left( \A\B \right) = \frac{\partial\A}{\partial\x}\B + \A\frac{\partial\B}{\partial\x}.
  $$

  Then, consider the equation

  $$
  \A\A\inv = \I.
  $$

  Differentiating both sides gives

  $$
  \begin{align*}
  \pbx (\A \A \inv) &= \pbx \I \\\\
  &= \zero.
  \end{align*}
  $$

  Then, applying the product rule, we have

  $$
  \begin{align*}
  \pbx (\A \A\inv) = \pbx(\A\inv)\A + \A\inv \frac{\partial\A}{\partial\x},
  \end{align*}
  $$

  hence

  $$
  \pbx (\A\inv)\A = - \A\inv \frac{\partial\A}{\partial\x},
  $$

  or

  $$
  \pbx (\A\inv) = - \A\inv \frac{\partial\A}{\partial\x} \A\inv,
  $$

  giving our result.

  **Proof of $\eqref{eq:deriv2}$:**

  We wish to show that

  $$
  \pbA \tr(\A\B) = \B\T.
  $$

  Consider the product $\A\B$:

  $$
  \begin{align*}
  \A\B &= \begin{bmatrix}
  a_{11} & a_{12} & \cdots & a_{1n} \\\\
  a_{21} & a_{22} & \cdots & a_{2n} \\\\
  \vdots & \vdots & \ddots & \vdots \\\\
  a_{n1} & a_{n2} & \cdots & a_{nn}
  \end{bmatrix}
  \begin{bmatrix}
  b_{11} & b_{12} & \cdots & b_{1n} \\\\
  b_{21} & b_{22} & \cdots & b_{2n} \\\\
  \vdots & \vdots & \ddots & \vdots \\\\
  b_{n1} & b_{n2} & \cdots & b_{nn}
  \end{bmatrix} \\\\
  &= \begin{bmatrix}
  \sum_{i=1}^na_{1i}b_{i1} & \sum_{i=1}^na_{1i}b_{i2} & \cdots & \sum_{i=1}^na_{1i}b_{in} \\\\
  \sum_{i=1}^na_{2i}b_{i1} & \sum_{i=1}^na_{2i}b_{i2} & \cdots & \sum_{i=1}^na_{2i}b_{in} \\\\
  \vdots & \vdots & \ddots & \vdots \\\\
  \sum_{i=1}^na_{ni}b_{i1} & \sum_{i=1}^na_{ni}b_{i2} & \cdots & \sum_{i=1}^na_{ni}b_{in}
  \end{bmatrix}.
  \end{align*}
  $$

  Each diagonal element is given by $\sum_{i=1}^n a_{ji}b_{ij}$ for $j \in [n]$, hence the trace is the sum over all the diagonal elements:

  $$
  \tr(\A\B) = \sum_{i=1}^n \sum_{j=1}^n a_{ji}b_{ij},
  $$

  Then, consider the derivative of this quantity with repect to a single element of $\A$:

  $$
  \begin{align*}
  \frac{\partial}{\partial a_{ij}} \tr(\A\B) &= \frac{\partial}{\partial a_{ij}} \sum_{k=1}^n \sum_{l=1}^n a_{kl}b_{lk},
  \end{align*}
  $$

  where I have replaced the indices of the sums to avoid confusion with the fixed matrix element $a_{ij}$. This derivative will be $0$ when $a_{kl} \neq a_{ij}$, hence

  $$
  \frac{\partial}{\partial a_{ij}} \sum_{k=1}^n \sum_{l=1}^n a_{kl}b_{lk} = \frac{\partial}{\partial a_{ij}} \left( a_{ij} b_{ji} \right) = b_{ji}.
  $$

  Thus, for a single matrix element, we have

  $$
  \frac{\partial}{\partial a_{ij}} \tr(\A\B) = b_{ji}.
  $$

  Generalizing this to the entire matrix $\A$, we have

  $$
  \pbA \tr(\A\B) = \B\T.
  $$


  **Proof of $\eqref{eq:deriv3}$:**

  


  </p>
</details>


Here, I'll show that

$$
\pbA \log \lvert \A \rvert = \left( \A\inv \right)\T.
$$

In order to prove this, I'll first show that

$$
\begin{equation}\label{A.2}\tag{A.2}
\px \log \lvert \A \rvert = \tr \left( \A\inv \frac{\partial\A}{\partial x} \right).
\end{equation}
$$

To see this, we can start by expressing the determinant of $\A$ as the product of its eigenvalues, hence

$$
\px \log \lvert \A \rvert = \px \log \prod_{i=1}^n \lambda_i = \px \sum_{i=1}^n \log \lambda_i = \sum_{i=1}^n \px \log \lambda_i.
$$

In general, each $\lambda_i$ could be some function of $x$. Using the chain rule, we have

$$
\px \log \lvert \A \rvert = \sum_{i=1}^n \frac{1}{\lambda_i} \frac{\partial\lambda_i}{\partial x}.
$$

Now, let's consider the right-hand side of $\eqref{A.2}$. Using the eigenvalue decomposition of $\A$, we can write this as

$$
\begin{align*}
\tr\left( \A\inv \frac{\partial\A}{\partial x} \right) &= \tr \left [\left( \sum_{i=1}^n \frac{\u_i\u_i\T}{\lambda_i} \right) \px \left( \sum_{i=1}^n \lambda_i\u\u_i\T \right) \right] \\\\
&= \tr\left[ \sum_{i, j} \frac{\u_i\u_i\T}{\lambda_i} \px \left( \lambda_j \u_j\u_j\T \right) \right] \\\\
\text{\scriptsize (via product rule)} \qquad &= \tr \left[ \sum_{i,j} \frac{\u_i\u_i\T}{\lambda_i} \left( \u_j\u_j\T \frac{\partial\lambda_j}{\partial x} + \lambda_j \px\u_j\u\T \right) \right] \\\\
&= \tr \left[ \sum_{i,j} \frac{1}{\lambda_i} \frac{\partial\lambda_j}{\partial x} \u_i\u_i\T\u_j\u_j\T + \sum_{i,j} \frac{\lambda_j}{\lambda_i} \u_i\u_i\T \px \u_j\u_j\T \right]
\end{align*}
$$

Noting that $\u_i\T\u_j = \delta_{ij}$, the first term inside the trace becomes

$$
\begin{equation}\label{A.3}\tag{A.3}
\sum_{i} \frac{1}{\lambda_i} \frac{\partial\lambda_j}{\partial x} \u_i\\u_i\T.
\end{equation}
$$

Then, examing the second term:

$$
\begin{align}
\sum_{i,j} \frac{\lambda_j}{\lambda_i} \u_i\u_i\T \px \u_j\u_j\T &= \sum_{i,j} \frac{\lambda_j}{\lambda_i} \u_i\u_i\T \left( \u_j \frac{\partial\u_j\}{\partial x}\T + \frac{\partial\u_j}{\partial x}\u_j\T \right) \nonumber \\\\
&= \sum_{i,j} \left( \frac{\lambda_j}{\lambda_i} \u_i \u_i\T\u_j \frac{\partial \u_j}{\partial x}\T + \frac{\lambda_j}{\lambda_i} \u_i\u_i\T \frac{\partial\u_j}{\partial x} \u_j\T\right) \nonumber \\\\
&= \sum_i \u_i\frac{\partial\u_j}{\partial x}\T + \sum_{i,j} \frac{\lambda_j}{\lambda_i} \u_i\u_i\T \frac{\partial\u_j}{\partial x} \u_j\T. \label{A.4}\tag{A.4}
\end{align}
$$

Putting together $\eqref{A.3}$ and $\eqref{A.4}$, we have

$$
\tr\left( \A\inv \frac{\partial\A}{\partial x} \right) = \tr \left[ \sum_i \frac{1}{\lambda_i} \frac{\partial\lambda_j}{\partial x} \u_i\u_i\T + \sum_i \u_i \frac{\partial\u_j}{\partial x}\T + \sum_{i,j} \frac{\lambda_j}{\lambda_i} \u_i\u_i\T \frac{\partial\u_j}{\partial x}\u_j\T \right].
$$

Next, I'll use the fact that $\tr(\A + \B) = \tr(\A) + \tr(\B)$. Note that this also implies that the trace and sum are interchangeable. Thus, we can write

$$
\begin{align}
\tr\left( \A\inv \frac{\partial\A}{\partial x} \right) &= \tr \left[ \sum_i \frac{1}{\lambda_i} \frac{\partial\lambda_j}{\partial x} \u_i\u_i\T \right] \tag{1} \\\\
&+ \tr \left[ \sum_i \u_i \frac{\partial\u_j}{\partial x}\T \right] \tag{2} \\\\
&+ \tr \left[ \sum_{i,j} \frac{\lambda_j}{\lambda_i} \u_i\u_i\T \frac{\partial\u_j}{\partial x}\u_j\T \right]. \tag{3}
\end{align}
$$

Then, I'll use two more properties of the trace; first, scalar multiplication: $\tr(c\A) = c\tr(\A)$, and second: the trace of an outer product of two vectors $\a$ and $\b$ is equal to their inner product: $\tr(\a\b\T) = \b\T\a$. Thus,

$$
(1) = \sum_i \lambda_i \frac{\partial\lambda_i}{\partial x} \tr(\u_i\u_i\T) = \sum_i \lambda_i \frac{\partial\lambda_i}{\partial x} \u_i\T\u_i = \sum_i \lambda_i \frac{\partial \lambda_i}{\partial x},
$$

where the dot product vanishes, since $\u_i\T\u_i = 1$. Similarly,

$$
(2) = \sum_i \frac{\lambda_i}{\lambda_j} \tr \left( \u_i \frac{\partial\u_j}{\partial x}\T \right) = \sum_i \frac{\lambda_i}{\lambda_j} \frac{\partial\u_j}{\partial x}\T \u_i
$$


Furthermore, again using the cyclic property of the trace, we have

$$
\tr \left( \u_i\u_i\T \frac{\partial \u_j}{\partial x}\u_j\T \right) = \tr \left( \u_j\T\u_i\u_i\T \frac{\partial\u_j}{\partial x} \right),
$$

hence

$$
(3) = \sum_{i,j} \frac{\lambda_j}{\lambda_i} \tr \left( \u_j\T\u_i\u_i\T \frac{\partial\u_j}{\partial x} \right) = \sum_i \tr \left( \u_i\T \frac{\partial \u_j}{\partial x} \right).
$$

Then, the expression inside the trace is an inner product between two vectors, resulting in a scalar. Since the trace of a scalar is itself a scalar, we have

$$
(3) = \sum_i \u_i\T \frac{\partial\u_j}{\partial x}.
$$

Putting these terms together,

$$
\tr\left( \A\inv \frac{\partial\A}{\partial x} \right) = \sum_i \lambda_i
$$



[^fn5]: This rule is particularly useful for Bayesian approaches to machine learning in which we model observations of some underlying distribution with additive Gaussian noise. For example, consider the case of linear regression, where observations are related via the function $f(\x) = \w\T\x.$ We can model noisy predictions by adding a zero-mean Gaussian random variable: $y = f(\x) + \epsilon$, where $\epsilon \sim \Norm(0, \sigma^2).$ Then, the observations are themselves Gaussian with mean $f(\x)$ and variance $\sigma^2$: $p(y \mid \x, \w) = \Norm(y \mid \w\T\x, \sigma^2).$ Thus, we see there is a linear relationship between 


[^fn7]: Note that since the logarithm is monotonic, the maximizer of $\log f(x)$ is equivalent to that of $f(x)$, so finding the parameters which maximize the log-likelihood is equivalent to find those which maximize the likelihood. Furthermore, we often choose to *minimize* the NLL as opposed to *maximizing* the log-likelihood, as this is often treated as a sort of loss function, and many modern optimization frameworks for machine learning are designed to minimize loss objectives.

[^fn7]: $\bS$ is symmetric because it is formed by taking a sum of symmetric matrices. Each matrix in the sum is symmetric because it is the outer product of a vector with itself, which always forms a symmetric PSD matrix as long as the vectors are nonzero.

[^fn8]: This is a common strategy which we've already seen in many of the previous derivations. Instead of carrying normalization constants throughout the procedure, we will find the form of the posterior distribution. Then, we can compute the normalization constant at the end. This will be easy if we find that the posterior takes the form of a known distribution, in which the normalization constant can be found by inspection.

[^fn9]: Given a parameter $\theta$ and a dataset $\cD$, we call the prior distribution $p(\theta)$ a *conjugate prior* to the likelihood $p(\cD \mid \theta)$ if the posterior $p(\theta \mid \cD)$ takes the same functional form as the prior. Since it is up to us to choose the prior distribution, it is often desirable to choose a conjugate prior, since it simplifies the computation of the posterior.

[^fn10]: Note that here we use $\sigma^2\ml = \frac{1}{n}\sum_{i=1} (x_i - \mu)^2$. There is a subtle point here: we actually defined the maximum likelihood estimator for the variance to be $$



<!-- ## Bayes' rule for linear Gaussian systems

Another useful result is Bayes' rule for linear Gaussian systems.[^fn5] In the previous sections, we had a random vector which specified a joint Gaussian distribution over $d$ random variables, and we wanted to find expressions for the marginal and conditional distributions of subsets of these random variables. Instead, suppose we are given a marginal distribution $p(\x)$ and a conditional distribution $p(\y \mid \x)$ and we wish to know the marginal distribution $p(\y)$ and the conditional $p(\x \mid \y).$ This can be seen as an application of Bayes' theorem. 

Specifically, consider the two random vectors $\x$ and $\y$ given by

$$
\begin{align*}
p(\x) &= \Norm(\x \mid \bmu, \bLambda\inv), \\\\[2pt]
p(\y \mid \x) &= \Norm(\y \mid \A\x+\b, \L\inv),
\end{align*}
$$

and suppose we want to know $p(\x \mid \y).$ First let's express $\x$ and $\y$ in terms of their joint distribution. As before, let

$$
\z = \begin{pmatrix}
\x \\\\
\y
\end{pmatrix}.
$$

Then, by the product rule of probability, we have

$$
p(\z) = p(\y \mid \x) \\, p(\x).
$$

It's nice to express this in terms of the log of the joint distribution:

$$
\begin{align}
&\log p(\z) = \log p(\y \mid \x) + \log p(\x) \nonumber \\\\
&= -\frac{1}{2} \bigg[ \left( \y - \A\x-\b \right)\T\L\left( \y - \A\x - \b \right) + (\x - \bmu)\T\bLambda(\x - \bmu) \bigg] + c \label{eq:16}
\end{align}
$$

We see that the exponent of the joint distribution is quadratic in terms of $\x$ and $\y,$ so $p(\z)$ will be a Gaussian. Then, let's again collect the quadratic and linear terms to find the parameters of the joint distribution. First, expanding $\eqref{eq:16},$ we have

$$
\begin{align*}
& -\frac{1}{2} \y\T\L\y + \y\T\L\A\x - \frac{1}{2}\x\T\A\T\L\A\x + \y\T\L\b - \x\T\A\T\L\b \\\\
&- \frac{1}{2}\x\T\bLambda\x + \x\T\bLambda\bmu\ + c.
\end{align*}
$$

We see that the quadratic terms are given by

$$
\begin{align*}
& -\frac{1}{2}\x\T\bLambda\x - \frac{1}{2}\y\T\L\y - \frac{1}{2}\x\T\A\T\L\A\x + \y\T\L\A\x \\\\[3pt]
&= -\frac{1}{2}\x\T \left( \bLambda + \A\T\L\A \right) \x + \y\T\L\A\x - \frac{1}{2}\y\T\L\y \\\\[3pt]
&= -\frac{1}{2}
\begin{pmatrix}
\x \\\\
\y
\end{pmatrix}\T
\begin{pmatrix}
\bLambda + \A\T\L\A & -\A\T\L \\\\
-\L\A & \L
\end{pmatrix}
\begin{pmatrix}
\x \\\\
\y
\end{pmatrix} \\\\[3pt]
&= -\frac{1}{2} \z\T\mathbf{R}\z,
\end{align*}
$$

where

$$
\mathbf{R} = \begin{pmatrix}
\bLambda + \A\T\L\A & -\A\T\L \\\\
-\L\A & \L
\end{pmatrix}.
$$

Thus, $\mathbf{R}$ is the precision matrix of the joint distribution $p(\z),$ and hence the covariance matrix $\bSigma_z$ can be found by the matrix inversion formula:

$$
\bSigma_z = \mathbf{R}\inv = \begin{pmatrix}
\bLambda\inv & \bLambda\inv\A\T \\\\
\A\bLambda\inv & \L\inv + \A\bLambda\inv\A\T
\end{pmatrix}.
$$

Similarly, we can collect the linear terms to identify the mean:

$$
\begin{align*}
\x\T\bLambda\bmu + \y\T\L\b - \x\T\A\T\L\b &= \x\T \left( \bLambda\bmu - \A\T\L \right) \b + \y\T\L\b \\\\[3pt]
&= \begin{pmatrix}
\x \\\\
\y
\end{pmatrix}\T
\begin{pmatrix}
\bLambda\bmu\ - \A\T\L\b \\\\
\L\b
\end{pmatrix}.
\end{align*}
$$

Again comparing this to $(7)$ we have that

$$
\bSigma_z\inv\bmu_z = 
\begin{pmatrix}
\bLambda\bmu\ - \A\T\L\b \\\\
\L\b
\end{pmatrix}.
$$

Then, the mean is given by

$$
\begin{align*}
\bmu_z &= \bSigma_z \begin{pmatrix}
\bLambda\bmu\ - \A\T\L\b \\\\
\L\b
\end{pmatrix} \\\\[3pt]
&= \begin{pmatrix}
\bLambda\inv & \bLambda\inv\A\T \\\\
\A\bLambda\inv & \L\inv + \A\bLambda\inv\A\T
\end{pmatrix}
\begin{pmatrix}
\bLambda\bmu\ - \A\T\L\b \\\\
\L\b
\end{pmatrix} \\\\[3pt]
&= \begin{pmatrix}
\bmu\ \\\\
\A\bmu\ + \b
\end{pmatrix}
\end{align*}
$$

Now, we can obtain the parameters of the marginal distribution $p(\y)$ using $(14)$ and $(15):$

$$
\begin{align*}
\bmu_y = \A\bmu + \b, \\\\
\bSigma_y = \L\inv + \A\bLambda\inv\A\T.
\end{align*}
$$

Similarly, we can use our previously derived rules to get an expression for the conditional distribution $p(\x \mid \y).$ In this case, it will be easier to use our expressions for the conditional parameters in terms of the precision matrix. From $(8),$ the conditional covariance is

$$
\begin{align*}
\bSigma_{x\mid y} &= \bR_{xx}\inv \\\\
&= \left( \bLambda + \A\T\L\A \right)\inv,
\end{align*}
$$

and, using $(9),$ the conditional mean is given by

$$
\begin{align*}
\bmu_{x\mid y} &= \bmu - \bR_{xx} \inv \bR_{xy}(\y - \bmu_y) \\\\
&= \bmu + (\bLambda + \A\T\L\A)\inv \A\T\L (\y - \A\bmu - \b) \\\\
&= \bmu + \bSigma_{x\mid y} \bigg[ \A\T\L (\y - \b) + \bLambda\bmu \bigg].
\end{align*}
$$ -->
