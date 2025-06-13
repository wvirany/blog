---
title: "A spelled-out introduction to Gaussian processes"  
date: "2025-06-13"  
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


Gaussian processes (GPs) have confounded me since I was first introduced to them. Many introductions talk about the beauty of implicitly defining infinitely many basis functions, or performing Bayesian inference directly in the space of functions. These explanations can seem daunting at first, but in this blog I aim to build up to GPs from a basic linear model,

spelled-out introduction.


## Bayesian linear regression

### The linear model

To build the foundation for GPs, I'll start by considering a Bayesian treatment of linear regression. We'll see that this is in fact a basic example of a GP.

Consider the linear model

$$
\begin{align*}
f(\x; \w) = \x\T\w,
\end{align*}
$$

where $\x \in \R^d$ is some input vector with $d$ features and $\w \in \R^d$ is the vector of parameters which specify the model. Note that we can incorporate an intercept term by always letting one element of $\x$ be constant, say $x_0 = 1$: [^fn1]

$$
f(\x; \w) = w_0 + w_1x_1 + \cdots + w_{d-1}x_{d-1}.
$$

Moreover, we can define a feature transformation $\phi: \R^d \to \R^m$. This transforms our feature vectors as follows:

$$
\phi(\x) = 
\begin{bmatrix}
\phi_0(\x) \\\\
\phi_1(\x) \\\\
\vdots \\\\
\phi_{m-1}(\x)
\end{bmatrix}.
$$

Again, by defining $\phi_0(\x) = 1$, we can implicitly incorporate a bias term. Now, we can redefine our model in terms of these basis functions:

$$
f(\x; \w) = \bphi\T\w,
$$

where $\bphi = \phi(\x)$, and now $\w \in \R^m$. If the basis functions $\{\phi_i\}$ are nonlinear in terms of $\x$, we can model nonlinear relationships between the features and targets while stil enjoying the benefits of a linear model, since $f$ is linear in terms of $w$.

As a simple example, suppose we have a one-dimensional input $x$, and we wish to model the class of polynomials up to degree $m-1$. Then, we simply define $\phi_j(x) = x^j$, which gives the following model:

$$
\begin{align*}
f(\x; \w) &= \bphi\T\w \\\\
&= w_0\phi_0(x) + w_1\phi_1(x) + \dots + w_{m-1}\phi_{m-1}(x) \\\\
&= w_0 + w_1x + w_2x^2 + \dots + w_{m-1}x^{m-1}
\end{align*}
$$

Now, we usually assume that a given observation $(\x, y)$ is corrupted by some noise, which we can model by adding a zero-mean Gaussian random variable to the functional outputs:

$$
y = f(\x; \w) + \epsilon,
$$

where $\epsilon \sim \Norm(0, \sigma^2)$. This gives rise to a probability distribution over $y$: [^fn2]

$$
\begin{align*}
p(y \mid \x, \w, \sigma^2) = \Norm(y \mid f(\x; \w), \sigma^2).
\end{align*}
$$

Before moving forward, a quick note on notation: when referring to a general probability distribution $p$, we list the dependent variables on the LHS of the conditional and the given variables on the RHS, including hyperparameters like $\sigma^2$, in no particular order. For example, $p(y \mid \x, \w, \sigma^2) = p(y \mid \w, \sigma^2, \x)$. It's completely arbitrary what order the variables are in, so long as they fall on the correct side of the conditional, and often certain variables are omitted from the notation and are implicitly assumed. From this point, I will omit hyperparameters from the general expressions for distributions.

However, when we refer to a specific distribution like the Gaussian, the positions of given variables have a specific meaning: the first position after the conditional is reserved for the mean, and the second position is reserved for the variance, hence $p(y \mid \x, \w) = \Norm(y \mid f(\x; \w), \sigma^2)$.

### Computing the parameters

Now, suppose we observe some iid dataset $\cD = (\X, \y)$, where $\X \in \R^{d\times n}$ is the matrix whose columns are the $n$ observed input vectors, and $\y = (y_1, y_2 ,\ldots, y_n)\T$ contains the correspondng target variables. Moreover, we can write the matrix containing our feature vectors as $\bPhi \in \R^{m\times n}$. Then, we have the following matrix equation:

$$
\y = \bPhi\T\w + \bepsilon,
$$

where $\bepsilon \sim \Norm(0, \sigma^2\I)$. As is often the case in supervised learning, we seek to find reasonable values for the parameters $\w$ in light of this observed data. In the frequentist approach to linear regression, we might model the likelihood function:

$$
\begin{align*}
p(\cD \mid \w) &= p(\y \mid \X, \w) \\\\
&= p(y_1, \dots, y_n \mid \x_1, \dots, \x_n, \w) \\\\
\text{\scriptsize (from iid assumption)} \qquad &= \prod_{i=1}^n p(y_i \mid \x, \w) \\\\
&= \prod_{i=1}^n \Norm(y_i \mid f(\x_i; \w), \sigma^2).
\end{align*}
$$

Then, we would maximize this expression w.r.t $\w$, which would give a point estimate for the parameters.

Instead, we will take a Bayesian treatment, which will allow us to compute a probability distribution over all possible values of of the parameters. To do so, we start by defining a prior on $\w$:

$$
p(\w) = \Norm \left( \w \mid 0, \bSigma \right).
$$

With no previous information about $\w$, it's reasonable to assume that all values of $\w$ are equally likely --- this corresponds to a zero-mean Gaussian. Furthermore, we often assume the parameters are independent, so $\bSigma = \alpha\I$, for some constant $\alpha$. However, I'll continue with the general form for the prior covariance.

Now, we'd like to infer the values of $\w$ from the observed data by computing the posterior distribution $p(\w \mid \cD)$. To do so, we can model the joint distribution of $\y$ and $\w$, then use the [rules for conditioning](../gaussian/#conditioning) on multivariate Guassian distributions. [^fn3]

First, we note that $\y$ is the [sum of two Gaussians](../gaussian/#sum-of-gaussians); the transformed $\bPhi\T\w$, and $\bepsilon$. Thus, $\y$ will be Gaussian-distributed as follows:

$$
p(\y \mid \X) = \Norm \left( \y \mid 0, \bPhi\T\bSigma\bPhi + \sigma^2\I \right).
$$

Finally, we compute the covariance between $\y$ and $\w$:

$$
\cov(\y, \w) = \cov(\bPhi\T\w + \bepsilon, \w) = \bPhi\T\cov(\w, \w) = \bPhi\T\bSigma.
$$

Thus, the joint distribution is given by

$$
p(\y, \w \mid \X) = \Norm \left( \left. \begin{bmatrix}
\y \\\\
\w
\end{bmatrix}\right\vert
0, \begin{bmatrix}
\bPhi\T\bSigma\bPhi + \sigma^2\I & \bPhi\T\bSigma \\\\
\bSigma\bPhi & \bSigma
\end{bmatrix}
\right).
$$

Now, the conditional distribution $p(\w \mid \y, \X)$ is Gaussian with the following parameters:

$$
\begin{align*}
\bmu_{\w\mid\cD} &= \bSigma\bPhi \left( \bPhi\T\bSigma\bPhi + \sigma^2\I \right)\inv\y, \\\\
\bSigma_{\w\mid\cD} &= \bSigma - \bSigma\bPhi\left( \bPhi\T\bSigma\bPhi + \sigma^2\I \right)\inv \bPhi\T\bSigma.
\end{align*}
$$


### Making predictions

Using our posterior parameter distribution $p(\w \mid \cD)$, we'd like to now make predictions at new test points $\X_\ast$, i.e., we'd like to compute the posterior predictive distribution $p(\y_\ast \mid \X_\ast, \cD)$. One way to do this is to average over all values of $\w$:

$$
p(\y_\ast \mid \X_\ast, \D) = \int p(\y_\ast \mid \X_\ast, \w) p(\w \mid \cD) d\w,
$$

where $p(\y_\ast \mid \X_\ast, \w)$ is just the likelihood and $p(\w\mid\cD)$ is the previously computed posterior parameter distribution. This integral is tractable, but takes a bit of work.[^fn4] An easier way to compute the predictive distribution is to note that, under our model,

$$
\y_\ast = \bPhi_\ast\T\w + \bepsilon.
$$

Then, if we use the posterior distribution over $\w$ and once again use the rules for transforming Gaussians, we have the following result:

$$
p(\y_\ast \mid \X_\ast, \D) = \Norm \left( \y_\ast \mid \bPhi_\ast\T\bmu_{\w\mid\cD}, \bPhi_\ast\T\bSigma_{\w\mid\cD}\bPhi_\ast + \sigma^2\I \right).
$$


### Bayesian linear regression in Python


## The kernel trick

If we write out the mean and covariance for the posterior predictive distribution, we have

$$
\begin{align*}
\bmu_{y_\ast \mid \cD} &= \bPhi_\ast\T\bmu_{\w\mid\cD} \\\\
&= \bPhi_\ast\T\bSigma\bPhi \left( \bPhi\T\bSigma\bPhi + \sigma^2\I \right)\inv\y,
\end{align*}
$$

and

$$
\begin{align*}
\bSigma_{\y_\ast \mid \cD} &= \bPhi_\ast\T\bSigma_{\w\mid\cD}\bPhi_\ast + \sigma^2\I \\\\
&= \bPhi_\ast\T \left[ \bSigma - \bSigma\bPhi \left( \bPhi\T\bSigma\bPhi + \sigma^2\I \right)\inv \bPhi\T\bSigma \right] \bPhi_\ast + \sigma^2\I \\\\
&= \bPhi_\ast\T\bSigma\bPhi_\ast - \bPhi_\ast\T\bSigma\bPhi \left( \bPhi\T\bSigma\bPhi + \sigma^2\I \right)\inv \bPhi\T\bSigma\bPhi_\ast + \sigma^2\I.
\end{align*}
$$

Thus, we see that all the dependence of the posterior predictive distribution on the features $\bPhi$ and $\bPhi_\ast$ is in the form of one of the following inner products:

$$
\bPhi\T\bSigma\bPhi, \quad \bPhi\T\bSigma\bPhi_\ast, \quad \bPhi_\ast\T\bSigma\bPhi, \quad \bPhi_\ast\T\bSigma\bPhi_\ast.
$$

In other words, all of the dependence on the features depends on an expression of the form

$$
k(\x, \x\p) = \phi(\x)\T\bSigma\phi(\x\p),
$$

where we've defined a "kernel function" $k$. By noting that $\bSigma$ is positive definite, and hence has a matrix square root[^fn5], we can rewrite this expression as

$$
k(\x, \x\p) = \psi(\x)\T\psi(\x\p),
$$

where $\psi(\x) = \Sigma^{1/2}\phi(\x)$. Then, the above expressions involving our features can be rewritten as the following Gram matrices: [^fn6]

$$
\begin{align*}
\K &= \bPhi\T\bSigma\bPhi, \quad \K_\ast = \bPhi\T\bSigma\bPhi_\ast, \quad \K_{\ast\ast} = \bPhi_\ast\T\bSigma\bPhi_\ast.
\end{align*}
$$

Note that $K_\ast = (\bPhi_\ast\T\bSigma\bPhi)\T$, so we can represent each of the expressions in terms of these three Gram matrices. We can now rewrite the parameters of the predictive distribution as

$$
\begin{align*}
\bmu_{\y_\ast \mid \cD} &= \K_\ast\T \left( \K + \sigma^2\I \right)\inv\y, \\\\
\bSigma_{\y_\ast \mid \cD} &= \K_{\ast\ast} - \K_\ast\T \left( \K + \sigma^2\I \right)\inv \K_\ast + \sigma^2\I.
\end{align*}
$$

To reiterate, we showed that, for *any* choice of feature map, we could express our result in terms of an inner product. Thus, we could achieve the same result by choosing a kernel function which can be represented by an inner product - then we never have to explicitly compute our feature vectors!


The advantage of this is that perhaps the feature vectors we'd like to work with are very high dimensional, and it might be cheaper to work in terms of the kernel function. As an example, suppose we have some



As a concrete example[^fn7], consider some vector $\x = (x_1, x_2, \dots, x_d) \in R^d$, and suppose we wish to express all the second-order polynomials in terms of the features of $\x$:

$$
\phi(\x) = \begin{bmatrix}
x_1 \\\\
x_2 \\\\
\vdots \\\\
x_d \\\\
x_1^2 \\\\
x_1x_2 \\\\
\vdots \\\\
x_d^2
\end{bmatrix}.
$$

Noting that there are $d$ linear terms, ${d\choose2}$ cross terms, and $d$ squared terms, computing this requires $\mathcal{O}(d^2)$ operations:

$$
d + {d\choose 2} + d = 2d + \frac{d(d-1)}{2} \sim \mathcal{O}(d^2).
$$

Alternatively, we can write the inner product as

$$
\begin{align*}
\phi(\x)\T\phi(\x\p) &= \sum_{i=1}^dx_ix_i\p + \sum_{i=1}^dx_i^2x_i^{\prime\\,2} + 2 \sum_{i=1}^d \sum_{j \neq i}^d x_ix_i\p x_jx_j\p \\\\
&= \sum_{i=1}^d x_ix_i\p + \left( \sum_{i=1}^d x_ix_i\p \right)^2 \\\\
&= \x\T\x\p + (\x\T\x\p)^2.
\end{align*}
$$

Thus, we can define the kernel function $k(\x, \x\p) = \x\T\x\p + (\x\T\x\p)^2$ --- we never have to compute $\phi(\x)$ directly, we can just plug $\x$ into the kernel function and make our predictions based on these values.




### Comments on valid kernel functions

A caveat to the approach descirbed above is that we must use a *valid* kernel. However, there are some fairly straightforward methods of obtaining these.

There are several ways to check if a function is a valid kernel; one way is to show that, for any set of vectors $S$, the Gram matrix whose elements are given by $\K_{ij} = k(\x_i, \x_j)$ for each $\x_i, \x_j \in S$ is always positive-semidefinite. Another way is to show that $k$ can be represented as $k(\x, \x\p) = \psi(\x)\T\psi(\x\p)$, for some explicit feature map $\psi$, as we saw before.

Moreover, once we have some valid kernel functions, we can use these as building blocks to construct new ones. For example, sums and products of valid kernels yield still valid kernels --- we can use these properties to build rich classes of kernel functions. However, I will not focus my dicussion on these properties. For further discussion I like chapter 6 in [Bishop, 2006](#references).

## Gaussian process regression

Now, let's return to the context of Bayesian linear regression. We previously derived the results for the predictive distribution


### Exact observations


### Noisy observations


### Gaussian process regression in Python


## References

1. C. M. Bishop, [*Pattern Recognition and Machine Learning*](https://www.microsoft.com/en-us/research/wp-content/uploads/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf), 2006.

2. C. E. Rasmussen & C. K. I. Williams, [*Gaussian Processes for Machine Learning*](https://gaussianprocess.org/gpml/chapters/RW.pdf), 2006.

3. Henry Chai's course, [*Bayesian Methods in Machine Learning*](https://www.cs.cmu.edu/~hchai2/courses/10624/), 2025.



[^fn1]: For a given variable, I use bold-faced symbols to refer to vectors; e.g., $\x \in \R^d$, and I use regular symbols to denote scalar values; e.g., the components of $\x$ are $(x_0, x_1, \dots, x_{d-1})$.

[^fn2]: This can be computed by noting that $y = f + \epsilon$ is an [affine transformation](../gaussian/#affine-transformation) of $\epsilon$. In general, given a Gaussian random variable $\x \sim \Norm(, \bSigma)$, the affine transformation $\y = \A\x + \b$ will also be Gaussian-dsitributed with $\y \sim \Norm(\A\bmu + \b, \A\bSigma\A\T)$.

[^fn3]: Alternatively, we could compute the posterior directly via Bayes' rule:
$$
p(\w \mid \cD) = \frac{p(\cD\mid\w)p(\w)}{p(\cD)}
$$

[^fn4]: This integral can be done explicitly by writing out the integrand as a product of Gaussians, then completing the square in the exponent in terms of $\w$. The integrand takes the form of a Gaussian distribution over $\w$, which can be easily computed by identifying it's normalization constant, and the resulting predictive distribution takes the form of a Gaussian in terms of the variables which are left over, i.e., those which were factored out of the integral.

[^fn5]: To see this, note that any $n\times n$ positive definite (PD) matrix has a valid eigenvalue decomposition (SVD). Thus, we can write $\bSigma = \U\Lambda\U\T$, where $\U\U\T = \I$, $\Lambda = \diag(\lambda_1, \dots, \lambda_n)$, and $\{\lambda_i\}$ are the eigenvalues of $\bSigma$ (also note that, since $\bSigma$ is PD, its eigenvalues are positive). Then, we define the matrix square root as $\bSigma^{1/2} = \U\Lambda^{1/2}\U\T$, where $\Lambda^{1/2}$ is, unsurprisingly, the matrix whose diagonal elements are given by the square roots of the eigenvalues.

[^fn6]: A Gram matrix is one whose elements are formed by the pairwise inner products for a set of vectors. In our case, the sets of vectors for our Gram matrices are $\\{\psi(\x)\\}$, for the input vectors in the train / test sets.

[^fn7]: I got this example from Prof. Henry Chai's [lecture slides](https://www.cs.cmu.edu/~hchai2/courses/10624/lectures/Lecture7_Slides.pdf) on the kernel trick.