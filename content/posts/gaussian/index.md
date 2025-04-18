---
title: "Some math behind the Gaussian distribution"  
date: "2025-03-03"  
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
- Fix formatting
- Fix equation numbers
- Examples
- Fill in appendix, proofs
- Discussion of Mahalanobis distance, Z-score, reconcile with example 1
- Discussion of conjugate priors, what they are and why we want to use them (because it makes finding an analytical form of the posterior easy)
- Comment about our frequent strategy of removing constants from sums in the exponent because they just become multiplicative scalars, which we account for add the end by normalizing
- Interpreting the moments!
- Double-check all results
- Tops of partials cut off (e.g., jacobian, matrix derivative rules)
- Conditioning: Add the other forms of the conditional params?
- Add all the results at the top for quick reference
- In "moments": show the formula for the second moment of a univariate Gaussian, and reference it properly (whether it's a footnote or appendix)
- Sum and product rule of probability in appendix + references to them
- Proper way to reference derivative rules for MLE section
 -->

<style>
  details {
    border: 1px solid black;
    border-radius: 8px;
    padding: 0.5em 0.5em 0em;
    margin-bottom: 2em;
    margin-top: 2em;
  }
  
  summary {
    font-weight: bold;
    margin: -0.5em -0.5em 0;
    padding: 0.5em;
    cursor: pointer;
    border-bottom: 1px solid #aaa;
    border-radius: 8px 8px 0 0;
  }
  
  details[open] summary {
    border-bottom: 1px solid #aaa;
    margin-bottom: 0;
  }
  
  details p {
    padding-top: 0.1em;
  }
</style>

Despite its ubiquity, I have frequently found myself in a state somewhere between *discomfort* and *panic* each time I am faced with the task of manipulating the Gaussian distribution, particularly in multiple dimensions. So, I've taken the opportunity to spend some time with the Gaussian in an attempt to overcome this aversion.

In this blog, I work through detailed derivations of key results involving the Gaussian which are important foundations for many topics in machine learning. I'll focus on the multivariate Gaussian distribution, beginning by reasoning about its shape and properties in high dimensions. Then, I derive some useful formulas, such as conditioning, marginalization, and Bayes' rule. Finally, I show two methods for estimating the parameters of a Gaussian from data; maximum likelihood estimation and Bayesian inference. I also include several examples along the way; some in math, some in code, and some in both!


## Properties of the Gaussian

In this section, I'll start by considering some basic properties of the Gaussian distribution. First, I'll show that surfaces on which the likelihood is constant form ellipsoids. Then, we'll see that the multivariate Gaussian distribution is indeed normalized, making it a valid probability distribution. Finally, we'll consider the first- and second-order moments of the multivariate Gaussian distribution in order to provide an interpretation of its parameters.

We write the Gaussian distribution for a random vector $\x \in \R^d$ as

$$
\begin{equation*}
\Norm(\x \mid \bmu, \Sigma) = \frac{1}{(2\pi)^{d/2}\lvert \Sigma \rvert^{1/2}}\exp\left( -\frac{1}{2} (\x - \bmu)\T\Sigma\inv(\x - \bmu)\right),
\end{equation*}
$$

where $\bmu \in \R^d$ is the mean vector and $\Sigma \in \R^{d\times d}$ is the covariance matrix. It's often nicer to work with the quadratic form in the exponent, which we define

$$
\begin{equation}\label{eq:mahalanobis}
\Delta^2 = (\x - \bmu)\T\Sigma\inv(\x-\bmu).
\end{equation}
$$

$\Delta$ is called the *Mahalanobis distance*, and is analagous to the "z-score" of a univariate Gaussian random variable $X$:

$$
\begin{equation*}
Z = \frac{X - \mu}{\sigma}.
\end{equation*}
$$


The z-score measures the number of standard deviations a random variable X is from the mean. It's also interesting to note that  and reduces to the Euclidean distance when $\Sigma$ is the identity matrix.

Since the covariance matrix $\Sigma$ is real and symmetric, we can perform [eigenvalue decomposition](#eigenvalue-decomposition) to write it in the form

$$
\begin{align*}
\Sigma &=  \U\Lambda\U\T \\\\
&=\sum_{i=1}^d\lambda_i\u_i\u_i\T,
\end{align*}
$$

Here, $\U$ is the matrix whose rows are given by $\u_i\T$, the eigenvectors of $\Sigma$, and $\Lambda = \diag(\lambda_1, \lambda_2, \ldots, \lambda_d)$ contains the corresponding eigenvalues. Note that we can choose the eigenvectors to be orthonormal (see ), i.e., [^fn1]

$$
\begin{equation*}
\u_i\T\u_j = \delta_{ij}.
\end{equation*}
$$

Thus, $\U$ is an orthogonal matrix, so $\U\U\T = \I$, and thus $\U\T = \U\inv.$ Moreover, we can easily write the inverse of the covariance matrix as

$$
\begin{equation*}
\Sigma\inv = \sum_{i=1}^d\frac{1}{\lambda_i}\u_i\u_i\T.
\end{equation*}
$$

Substituting this into $\eqref{eq:mahalanobis}$, we get

$$
\begin{align*}
\Delta^2 &= \sum_{i=1}^d \frac{1}{\lambda_i}(\x - \bmu)\T\u_i\u_i\T(\x - \bmu) \\\\[1pt]
&= \sum_{i=1}^d\frac{y_i^2}{\lambda_i},
\end{align*}
$$

where I've introduced

$$
\begin{equation*}
y_i = \u_i\T(\x - \bmu),
\end{equation*}
$$

The set $\\{y_i\\}$ can then be seen as a transformed coordinate system, shifted by $\bmu$ and rotated by $\u_i.$[^fn2] Alternatively, we can write this as a vector:

$$
\begin{equation}\label{eq:ytransform}
\y = \U(\x - \bmu),
\end{equation}
$$

Now, all of the dependence of the Gaussian on $\x$ is determined by $\Delta^2.$ Thus, the Gaussian is constant on surfaces for which $\Delta^2$ is constant. Then, let

$$
\begin{equation*}
\Delta^2 = \sum_{i=1}^d\frac{y_i^2}{\lambda_i} = r
\end{equation*}
$$

for some constant $r.$ This defines the equation of an ellipsoid in $d$ dimensions.  

<details>
  <summary>Example: Level sets of the Gaussian</summary>
  <p>
  In the case of the Gaussian distribution, we often like to talk about the probability that an observation will fall within some range of values. For example, we might like to use the fact that the probability a random variable falls within one standard deviation from the mean is approximately $0.683.$

  As an example, I'll consider the analagous case for a bivariate Gaussian, in which we would like to find the ellipses corresponding to the probabilities that a point falls within one, two, or three standard deviations from the mean.

  First consider a univariate Gaussian random variable $X \sim \Norm(\mu, \sigma^2).$ The probability that $\X$ is within one standard deviation from the mean is given by

  $$
  \begin{align*}
  P(\lvert X - \mu \rvert \leq \sigma) &= P(-\sigma \leq X - \mu \leq \sigma) \\\\
  &= P(-1 \leq \frac{X - \mu}{\sigma} \leq 1) \\\\
  &= P(-1 \leq Z \leq 1),
  \end{align*}
  $$

  where $Z = \frac{X - \mu}{\sigma}$ is a standard Normal random variable. Then, this probability is given by

  $$
  \begin{align*}
  P(-1 \leq Z \leq 1) &= P(Z \leq 1) - P(Z \leq -1) \\\\
  &= \Phi(1) - \Phi(-1)
  \end{align*}
  $$

  where $\Phi(\cdot)$ is the cumulative distribution function of $Z$, for which the functional values are usually determined from a [table](https://engineering.purdue.edu/ChanGroup/ECE302/files/normal_cdf.pdf), or when in doubt we can make the computers think for us:

  ```python
  from scipy.stats import norm

  print(f"{norm.cdf(1) - norm.cdf(-1):.3f}")
  ```
  ```
  0.683
  ```
  
  We can do this in a similar fashion to find the probabilities that $X$ falls within $2\sigma$ or $3\sigma$ from the mean, for which the values are approximately $0.954$ and $0.997$, respectively.

  In the univariate case, the quantity $\lvert X - \mu \rvert / \sigma$ measures the number of standard deviations $X$ is from the mean. The Mahalanobis is analgous to this in the multivariate case. However, it's important to note that, in multiple dimensions, the number of standard deviations a random vector $\x$ is from the mean depends on the *direction* $\x$ is with respect to the mean. For example, a Gaussian will in general have different variances along different dimensions, and thus, depending on the direction, $\x$ 
  

  Thus, we seek to find some constant $k$ for which

  $$
  \begin{equation*}
  P(\Delta^2 \leq k^2) = 0.683.
  \end{equation*}
  $$

  To do so, we note that $\Delta^2$ follows a chi-squared distribution. To see this, recall the expression for $\Delta^2$:

  $$
  \begin{align*}
  \Delta^2 &= (\x - \bmu)\T\U\T\Lambda\inv\U(\x - \bmu) \\\\
  &= \y\T\Lambda\inv\y.
  \end{align*}
  $$

  Then, $\y$ is a random vector with zero mean and diagonal covariance $\Lambda.$ Since it has diagonal covariance, the elements of $\y$ are uncorrelated. In general, uncorrelated does not imply independence; however, [in the case of the Gaussian, it does](#1). Then, consider yet another transformation:
  
  $$
  \begin{equation*}
  \z = \Lambda^{-1/2}\y,
  \end{equation*}
  $$

  where $\Lambda^{-1/2} = \diag(\lambda_1^{-1/2}, \ldots, \lambda_d^{-1/2}).$ Now, the elements of $\z$ have been standardized, so $\z$ is a vector of standard Normal random variables. Then, we have

  $$
  \begin{align*}
  \Delta^2 &= \z\T\z \\\\
  &= z_1^2 + z_2^2 + \cdots + z_d^2.
  \end{align*}
  $$

  Since each $z_i$ is an independent standard Normal, the sum $\Delta^2$ takes a chi-squared distribution with $d$ degrees of freedom. Then, consider the cumulative distribution function of a chi-squared random variable with $d=2$:

  $$
  \begin{equation*}
  F_{\chi^2_2} (x) = P(\chi^2 \leq x).
  \end{equatioon*}
  $$

  In this case, we know $F_{\chi^2_2} (x) = 0.683$, and we wish to find x. To do so, we can make use of the inverse cumulative distribution function, otherwise known as the *quantile function*:

  $$
  \begin{equation*}
  k = F_{\chi^2_2}\inv (p) = Q(p).
  \end{equation*}
  $$

  We can evaluate this in several ways, but the cumulative distribution function of $\chi^2_2$ takes a nice form, so we'll do it by hand:

  $$
  \begin{equation*}
  F_{\chi^2_2}(x) = 1 - e^{-x/2} = p
  \end{equation*}
  $$

  Hence,

  $$
  \begin{align*}
  Q(p) &= \log \frac{1}{(1 - p)^2} \\\\
  Q(0.683) &\approx 2.30.
  \end{align*}
  $$

  Thus, the value for $k$ for which

  $$
  \begin{equation*}
  P(\Delta^2 \leq k) = 0.683
  \end{equation*}
  $$

  is approximately 2.30. Thus, the equation for the corresponding ellipse will be given by

  $$
  \begin{equation*}
  \frac{y_1^2}{\lambda_1} + \frac{y_2^2}{\lambda_2} = 2.30
  \end{equation*}
  $$

  </p>
</details>



### Normalization

Now, our goal is to show that the multivariate Gaussian distribution is normalized. Let's consider the Gaussian in the new coordinate system $\\{y_i\\}.$ Rearranging $\eqref{eq:ytransform}$, we can write the transformation as

$$
\begin{equation*}
\x = g(\y) = \U\T\y + \bmu.
\end{equation*}
$$

Then, to transform from $\x$-space to $\y$-space, we use the [change of variables formula](#change-of-variables), given by

$$
\begin{align*}
p_y(\y) &= p_x(\x)\lvert \J \rvert \\\\[2pt]
&= p_x(g(\y))\lvert \J \rvert.
\end{align*}
$$

Here, $\J$ is the Jacobian whose elements are given by

$$
\begin{equation*}
J_{ij} = \frac{\partial x_i}{\partial y_j}.
\end{equation*}
$$

The derivative of $\x$ with respect to $\y$ is $\U\T$, hence the elements of $\J$ are

$$
J_{ij} = U_{ji}.
$$

Then, to find the determinant of the Jacobian, we have

$$
\lvert \J \rvert ^2 = \lvert \U\T \rvert ^2 = \lvert \U\T \rvert \lvert \U \rvert = \lvert \U\T\U \rvert = \lvert \I \rvert = \mathbf{1}.
$$

Thus, $\lvert \J \rvert = \mathbf{1}$, making our transformation

$$
p_y(\y) = p_x(g(\y)).
$$

Then, we can write the Gaussian in terms of $\y$ as

$$
\begin{align*}
p_y(\y) &= \frac{1}{(2\pi)^{d/2}\lvert\Sigma\rvert^{1/2}}\exp\left( -\frac{1}{2} (\x - \bmu)\T\Sigma\inv(\x - \bmu) \right).
\end{align*}
$$

Examining the term in the exponent, we have

$$
\begin{align*}
(\x - \bmu)\T\Sigma\inv(\x - \bmu)  &= (\U\T\y)\T \Sigma\inv(\U\T\y) \\\\[1pt]
&= \y\T\U (\U\T\Lambda\U)\inv \U\T\y \\\\[1pt]
&= \y\T\U \U\inv\Lambda (\U\T)\inv \U\T\y \\\\[1pt]
&= \y\T\Lambda\y.
\end{align*}
$$

So,

$$
\begin{align*}
p_y(\y) &= \frac{1}{(2\pi)^{d/2}\lvert \Sigma \rvert ^{1/2}} \exp \left( -\frac{1}{2} \y\T\Lambda\y \right) \nonumber \\\\[1pt]
&= \frac{1}{(2\pi)^{d/2}\lvert \Sigma \rvert ^{1/2}} \exp \left( -\frac{1}{2} \sum_{i=1}^d \frac{y_i^2}{\lambda_i} \right).
\end{align*}
$$

Then, it's useful to show that

$$
\begin{align*}
\lvert \Sigma \rvert &= \lvert \U\Lambda\U\T \rvert = \lvert \U \rvert \lvert \Lambda \rvert \lvert \U\T \rvert = \lvert \Lambda \rvert = \prod_{i=1}^d \lambda_i,
\end{align*}
$$

hence

$$
\frac{1}{\lvert \Sigma \rvert^{1/2}} = \prod_{i=1}^d \frac{1}{\sqrt{\lambda_i}}.
$$


Thus, noting that the exponent of a sum becomes a product of exponents, we have

$$
p_y(\y) = \prod_{i=1}^d \frac{1}{\sqrt{2\pi\lambda_i}} \exp \left( -\frac{y_i^2}{2\lambda_i} \right).
$$

Then,

$$
\begin{align*}
\int_\y p_y(\y) d\y &= \prod_{i=1}^d \int_{y_i}  \frac{1}{\sqrt{2\pi\lambda_i}} \exp \left( -\frac{y_i^2}{2\lambda_i} \right) dy_i.
\end{align*}
$$

We see that each element of the product is just a univariate Gaussian over $y_i$ with mean $0$ and variance $\lambda_i$, each of which integrates to 1. This shows that $p_y(\y)$ and thus $p_x(\x)$ is indeed normalized.


### Moments of the Gaussian

Finally, we will examine the first and second moments of the Gaussian. The first moment is given by

$$
\begin{align*}
\E[\x] &= \frac{1}{(2\pi)^{d/2}\lvert\Sigma\rvert^{1/2}}\int\exp\left( -\frac{1}{2} (\x - \bmu)\T\Sigma\inv(\x - \bmu) \right) \x \\, d\x \\\\[3pt]
&= \frac{1}{(2\pi)^{d/2}\lvert\Sigma\rvert^{1/2}}\int\exp\left( -\frac{1}{2} \z\T\Sigma\inv\z \right) (\z + \bmu ) \\, d\z,
\end{align*}
$$

where I've introduced the change of variables $\z = \x - \bmu.$ We can split this up as

$$
\frac{1}{(2\pi)^{d/2}\lvert\Sigma\rvert^{1/2}} \left[ \int\exp\left( -\frac{1}{2} \z\T\Sigma\inv\z \right) \z \\, d\z + \int\exp\left( -\frac{1}{2} \z\T\Sigma\inv\z \right) \bmu \\, d\z\right].
$$

Inspecting the first term, we see that $\exp(-\frac{1}{2}\z\T\Sigma\inv\z)$ is an even function in $\z$, and $\z$ is odd. Then, the product is an odd function, so the integral over a symmetric domain (in this case all of $\R^d$) is zero. The second term is just $\bmu$ times a Gaussian, which will integrate to 1 when multiplied by the normalization constant. Thus, we have the (perhaps unsurprising) result:

Now, in the univariate case, the second moment is given by $\E[x^2].$ In the multivariate case, there are $d^2$ second moments, each given by $\E[x_i, x_j]$ for $i, j \in [d].$ We can group these together to form the matrix $\E[\x\x\T].$ We write this as

$$
\begin{align}
\E[\x\x\T] &= \frac{1}{(2\pi)^{d/2}\lvert\Sigma\rvert^{1/2}}\int\exp\left( -\frac{1}{2}(\x - \bmu)\T\Sigma\inv (\x-\bmu) \right) \x\x\T d\x \nonumber \\\\[3pt]
&= \frac{1}{(2\pi)^{d/2}\lvert\Sigma\rvert^{1/2}}\int\exp\left( -\frac{1}{2}\z\T\Sigma\inv \z \right) (\z + \bmu)(\z + \bmu)\T d\z \nonumber \\\\[3pt]
&= \frac{1}{(2\pi)^{d/2}\lvert\Sigma\rvert^{1/2}}\int\exp\left( -\frac{1}{2}\z\T\Sigma\inv \z \right) (\z\z\T + 2\z\T\bmu + \bmu\bmu\T)  d\z.
\end{align}
$$

By the same arguments as before, the term involving $\z\T\bmu$ will vanish due to symmetry, and the term involving $\bmu\bmu\T$ will integrate to $\bmu\bmu\T$ due to normalization. Then, we are left with the term involving $\z\z\T.$ Using the eigenvalue decomposition of $\Sigma$, we can write

<p align=center>
$\y = \U\z, \quad$ or $\quad \z = \U\T\y.$
</p>

Recall that $\U$ is the matrix whose rows are given by the eigenvectors of $\Sigma.$ So, $\U\T$ is the matrix whose *columns* are given by the eigenvectors. Thus,

$$
\begin{align*}
\z &= \begin{bmatrix}
\u_1 & \u_2 & \cdots & \u_d
\end{bmatrix}
\begin{bmatrix}
y_1 \\\\
y_2 \\\\
\vdots \\\\
y_d
\end{bmatrix} \\\\[3pt]
&= \begin{bmatrix}
u_{11} & u_{21} & \cdots & u_{d1} \\\\
u_{12} & u_{22} & \cdots & u_{d2} \\\\
\vdots & \vdots & \ddots & \vdots \\\\
u_{1d} & u_{2d} & \cdots & u_{dd} \\\\
\end{bmatrix}
\begin{bmatrix}
y_1 \\\\
y_2 \\\\
\vdots \\\\
y_d
\end{bmatrix} \\\\[3pt]
&= \begin{bmatrix}
u_{11}y_1 + u_{21}y_2 + \cdots + u_{d1}y_d \\\\
u_{12}y_1 + u_{22}y_2 + \cdots + u_{d2}y_d \\\\
\vdots \\\\
u_{1d}y_1 + u_{2d}y_2 + \cdots + u_{dd}y_d \\\\
\end{bmatrix} = \sum_{i=1}^d y_i\u_i,
\end{align*}
$$

where $u_{ij}$ is the $j$th element of $\u_i.$ Then, using this expression for $\z$, and recalling the form for $p_y(\y)$ in $(3)$, we can write the first term of $(4)$ as

$$
\begin{align}
&\quad\\,\\, \frac{1}{(2\pi)^{d/2}\lvert\Sigma\rvert^{1/2}} \int \exp \left( - \sum_{k=1}^d \frac{y_k^2}{2\lambda_k} \right) \sum_{i=1}^d\sum_{j=1}^d y_i y_j \u_i\u_j\T d\y \nonumber \\\\[2pt]
&= \frac{1}{(2\pi)^{d/2}\lvert\Sigma\rvert^{1/2}} \sum_{i=1}^d\sum_{j=1}^d \u_i\u_j\T  \int \exp \left( - \sum_{k=1}^d \frac{y_k^2}{2\lambda_k} \right) y_i y_j d\y.
\end{align}
$$

Now, the integral takes the form

$$
\begin{align*}
\int \exp \left( - \sum_{k=1}^d y_k^2\right) y_i y_j d\y.
\end{align*}
$$

When $i\neq j$, we can expand this as the product

$$
\begin{gather*}
\prod_{k=1}^d \int \exp(-y_k^2) y_i y_j d\y \\\\[2pt] = \int \\! \exp(-y_1^2) dy_1 \cdots \\! \int \exp(-y_i^2) y_i dy_i \cdots \int \\! \exp(-y_j^2) y_j dy_j \cdots \int \\! \exp(-y_d^2) dy_d.
\end{gather*}
$$

In this case, due to our symmetry arguments, the terms involving $y_i$ and $y_j$ vanish, and hence the integral vanishes when $i\neq j.$ If $i=j$, then the second term in $(5)$ can be written as

$$
\sum_{i=1}^d \u_i\u_i\T \prod_{k=1}^d \int \frac{1}{\sqrt{2\pi\lambda_k}} \exp \left( - \frac{y_k^2}{2\lambda_k} \right) y_i^2 dy_k.
$$

where we brought the normalization constant inside the product. The terms in the product for which $i \neq k$ are just univariate Gaussian, and hence normalize to $1.$ Thus, the only term left in the product is

$$
\int \frac{1}{\sqrt{2\pi\lambda_k}} \exp \left( - \frac{y_i^2}{2\lambda_i} \right) y_i^2 dy_i,
$$

which is just the expression for the second moment of a univariate Gaussian with mean $0$ and variance $\lambda_i.$ In general, the second moment of a univariate Gaussian $\Norm(x \mid \mu, \sigma^2)$ is $\mu\^2 + \sigma^2.$ Thus, we are left with

$$
\begin{align}
\E[\x\x\T] &= \bmu\bmu\T + \sum_{i=1}^d \u_i\u_i\T \lambda_i \nonumber \\\\[1pt]
&= \bmu\bmu\T + \Sigma.
\end{align}
$$

So, we have that the first and second moments of the Gaussian are given by $\E[\x] = \bmu$ and $\E[\x\x\T] = \bmu\bmu\T + \Sigma$, respectively.

## Conditioning

Now, suppose we have some random vector $\z \in \R^d$, specified by a Gaussian distribution:

$$
\z \sim \Norm(\z \mid \bmu, \Sigma).
$$

Then, suppose we partition $\z$ into two constituent vectors $\x \in \R^m$ and $\y \in \R^{d-m}$:

$$
\z = \begin{pmatrix}
\x \\\\
\y
\end{pmatrix},
$$

and our goal is to find an expression for the conditional distribution $p(\x \mid \y).$ The parameters specifying the joint distribution can likewise be partitioned as follows:

$$
\bmu\ = \begin{pmatrix}
\bmu_x \\\\
\bmu_y
\end{pmatrix}, \quad
\Sigma = \begin{pmatrix}
\Sigma_{xx} & \Sigma_{xy} \\\\
\Sigma_{yx} & \Sigma_{yy}
\end{pmatrix}.
$$

Note that since $\Sigma$ is symmetric, we have $\Sigma_{xx} = \Sigma_{xx}\T, \Sigma_{yy} = \Sigma_{yy}\T$, and $\Sigma_{xy} = \Sigma_{yx}\T.$ It's also useful to define the precision matrix $\Lambda = \Sigma\inv$, and its partitioned form: [^fn3]

$$
\Lambda = \begin{pmatrix}
\Lambda_{xx} & \Lambda_{xy} \\\\
\Lambda_{yx} & \Lambda_{yy}
\end{pmatrix}.
$$

Since the inverse of a symmetric matrix is itself symmetric, we have that $\Lambda = \Lambda\T$, hence the same properties hold as the covariance matrix regarding the symmetry between constituent parts of the partitioned matrix. However, it's important to note that the partitioned matrices of the precision matrix are not simply the inverses of the corresponding elements of the covariance matrix. Instead, we'll shortly see how to take the inverse of a partitioned matrix.

Now, one way to find an expression for the conditional $p(\x \mid \y)$ would be to simply use the [product rule of probability](#sum-and-product-rules-of-probability):

$$
\begin{align*}
p(\x, \y) &= p(\x \mid \y) \\, p(\y) \\\\[3pt]
\Rightarrow \quad p(\x \mid \y) &= \frac{p(\x, \y)}{p(\y)}.
\end{align*}
$$

However, normalizing the resulting expression can be cumbersome. Instead, let's consider the quadratic form in the exponent of the joint distribution:

$$
\begin{align}
& -\frac{1}{2}(\z - \bmu\)\T\Sigma\inv (\z - \bmu\) \nonumber \\\\[10pt]
&= -\frac{1}{2} \begin{pmatrix}
\x - \bmu_x \\\\
\y - \bmu_y
\end{pmatrix}\T
\begin{pmatrix}
\Lambda_{xx} & \Lambda_{xy} \\\\
\Lambda_{yx} & \Lambda_{yy}
\end{pmatrix}
\begin{pmatrix}
\x - \bmu_x \\\\
\y - \bmu_y
\end{pmatrix} \nonumber \\\\[15pt]
&= -\frac{1}{2} (\x - \bmu_x)\T\Lambda_{xx} (\x - \bmu_x) - \frac{1}{2}(\x - \bmu_x)\T \Lambda_{xy} (\y - \bmu_y) \nonumber \\\\
&\\qquad - \frac{1}{2}(\y-\bmu_y)\T\Lambda_{yx} (\x - \bmu_x) - \frac{1}{2} (\y - \bmu_y)\T \Lambda_{yy} (\y - \bmu_y) \nonumber \\\\[10pt]
&= -\frac{1}{2} (\x - \bmu_x)\T\Lambda_{xx} (\x - \bmu_x) - (\x-\bmu_x)\T\Lambda_{xy} (\y - \bmu_y) \nonumber \\\\
&\\qquad - \frac{1}{2} (\y - \bmu_y)\T \Lambda_{yy} (\y - \bmu_y).
\end{align}
$$

In the last line, I use the fact that [^fn4]

$$
(\x - \bmu_x)\T\Lambda_{xy}  (y - \bmu_y) = (\y - \bmu_y)\T\Lambda_{yx}  (\x - \bmu_x).
$$

I'll repeatedly use this fact in the following calculations to combine cross terms.

Evaluating the conditional $p(\x \mid \y)$ involves fixing $\y$ and treating this as a function of $\x.$ Then, since the expression in $(6)$ is a quadratic function of $\x$, the resulting distribution $p(\x \mid \y)$ will also take the form of a Gaussian. So, our goal is to find the mean $\bmu_{x\mid y}$ and covariance $\Sigma_{x\mid y}$ which specify this distribution. To do so, note that in general, we can write the exponent of a Gaussian as

$$
\begin{equation}
-\frac{1}{2}(\z - \bmu\)\T \Sigma\inv (\z - \bmu\) = -\frac{1}{2}\z\T\Sigma\inv\z + \z\T\Sigma\inv\bmu\ + c,
\end{equation}
$$

where $c$ denotes all the terms independent of $\z.$ Thus, if we can rewrite $(6)$ in this form, we can identify the coefficients of the quadratic and linear terms in $\x$ as the mean and covariance of $p(\x \mid \y).$ This may not seem clear at first, but I think going through the process will illuminate things.

Expanding $(6)$ gives

$$
-\frac{1}{2} \x\T\Lambda_{xx} \x + \x\T\Lambda_{xx} \bmu_x - \x\T\Lambda_{xy} \y + \x\T\Lambda_{xy} \bmu_y + c,
$$

where $c$ again denotes all terms which do not depend on $\x.$ Equating this to the general form as in the right-hand side of $(7)$, we have

$$
-\frac{1}{2} \x\T\Lambda_{xx} \x + \x\T\Lambda_{xx} \bmu_x - \x\T\Lambda_{xy} \y + \x\T\Lambda_{xy} \bmu_y = -\frac{1}{2}\x\T\Sigma_{x\mid y}\inv \x + \x\T\Sigma_{x\mid y}\inv \bmu_{x \mid y}.
$$

Immediately, we can equate the quadratic terms to see that

$$
\begin{equation}
\Sigma_{x\mid y}\inv = \Lambda_{xx}.
\end{equation}
$$

Then, collecting the linear terms, we have

$$
\x\T\Lambda_{xx} \bmu_x - \x\T\Lambda_{xy} \y + \x\T \Lambda_{xy} \bmu_y = \x\T\left( \Lambda_{xx} \bmu_x - \Lambda_{xy}(\y - \bmu_y) \right).
$$

Thus, we have

$$
\Sigma_{x\mid y}\inv \bmu_{x\mid y} = \Lambda_{xx} \bmu_x - \Lambda_{xy} (\y - \bmu_y),
$$

or, using $(8)$:

$$
\begin{equation}
\bmu_{x\mid y} = \bmu_x - \Lambda_{xx}\inv\Lambda_{xy} (\y - \bmu_y).
\end{equation}
$$

Here, we've expressed the quantities $\bmu_{x\mid y}$ and $\Sigma_{x\mid y}$ in terms of $\Lambda.$ Instead, we can express them in terms of $\Sigma.$ To do so, we'll use the matrix inversion identity:

$$
\begin{pmatrix}
\A & \B \\\\
\bC & \D
\end{pmatrix}\inv = \begin{pmatrix}
\M & -\M\B\D\inv \\\\
-\D\inv\bC\M & \D\inv + \D\inv\bC\M\B\D
\end{pmatrix},
$$

where $\M$ is the [Schur complement](#the-schur-complement), defined

$$
\M = (\A - \B\D\inv\bC)\inv.
$$

Then, since

$$
\begin{pmatrix}
\Lambda_{xx} & \Lambda_{xy} \\\\
\Lambda_{yx} & \Lambda_{yy}
\end{pmatrix}\inv = 
\begin{pmatrix}
\Sigma_{xx} & \Sigma_{xy} \\\\
\Sigma_{yx} & \Sigma_{yy}
\end{pmatrix},
$$

we have

$$
\begin{align*}
\Lambda_{xx} &= (\Sigma_{xx} - \Sigma_{xy}\Sigma_{yy}\inv\Sigma_{yx})\inv, \\\\[4pt]
\Lambda_{xy} &= - (\Sigma_{xx} - \Sigma_{xy}\Sigma_{yy}\inv\Sigma_{yx})\inv \Sigma_{xy} \Sigma_{yy}\inv.
\end{align*}
$$

Plugging these expressions into $(8)$ and $(9)$ gives

$$
\Sigma_{x\mid y} = \Sigma_{xx} - \Sigma_{xy}\Sigma_{yy}\inv\Sigma_{yx}
$$

and

$$
\begin{align*}
\bmu_{x\mid y} &= \bmu_x + (\Sigma_{xx} - \Sigma_{xy}\Sigma_{yy}\inv\Sigma_{yx}) (\Sigma_{xx} - \Sigma_{xy}\Sigma_{yy}\inv\Sigma_{yx})\inv \Sigma_{xy} \Sigma_{yy}\inv (\y - \bmu_y) \\\\[2pt]
&= \bmu_x - \Sigma_{xy}\Sigma_{yy}\inv (\y - \bmu_y).
\end{align*}
$$

Thus, $p(\x \mid \y)$ is the Gaussian distribution given by the following parameters:

$$
\begin{align}
\quad \bmu_{x\mid y} &= \bmu_x - \Sigma_{xy}\Sigma_{yy}\inv(\y - \bmu_y) \quad \\\\[2pt] 
\quad \Sigma_{x\mid y} &= \Sigma_{xx} - \Sigma_{xy}\Sigma_{yy}\inv\Sigma_{yx}. \quad 
\end{align}
$$

## Marginalization

Now, given the joint distribution $p(\x, \y)$ as above, suppose we wish to find the marginal distribution

$$
\begin{equation}
p(\x) = \int p(\x, \y) d\y.
\end{equation}
$$

Our goal, then, is to integrate out $\y$ to obtain a function of $\x.$ Then, we can normalize the resulting function of $\x$ to obtain a valid probability distribution. To do so, let's again consider the quadratic form in the exponent given by $(6).$ First, we collect all terms which depend on $\y$:

$$
\begin{align}
&\quad \\,\\, - (\x - \bmu_x)\T\Lambda_{xy} (\y - \bmu_y) - \frac{1}{2} (\y - \bmu_y)\T\Lambda_{yy} (\y - \bmu_y) \nonumber \\\\[2pt]
&= -\frac{1}{2} \y\T \Lambda_{yy} \y + \y\T\Lambda_{yy} \bmu_y - \y\T \Lambda_{yx} (\x - \bmu_x) \nonumber \\\\[2pt]
&= -\frac{1}{2}\y\T \Lambda_{yy} \y + \y\T \m,
\end{align}
$$

where I've introduced

$$
\m = \Lambda_{yy} \bmu_y - \Lambda_{yx} (\x - \bmu_x).
$$

By [completing the square](#completing-the-square), we can write $(13)$ as

$$
-\frac{1}{2} (\y - \Lambda_{yy}\inv\m)\T \Lambda_{yy} (\y - \Lambda_{yy}\inv\m) + \frac{1}{2}\m\T\Lambda_{yy}\inv \m.
$$

Note that $\m$ does not depend on $\y$; however, it does depend on $\x.$ Now, we're able to factor the integral in $(11)$ as

$$
\exp\big( g(\x) \big)\int \exp \left\\{ -\frac{1}{2} (\y - \Lambda_{yy}\inv\m)\T \Lambda_{yy} (\y - \Lambda_{yy}\inv\m) \right\\} d\y,
$$

where $g(\x)$ contains all the remaining terms which do not depend on $\y.$ This integral is now easy to compute, since it is just an unnormalized Gaussian and will evaluate to the reciprocal of the corresponding normalization factor. Thus, the marginal distribution $p(\x)$ will have the exponential form given by $g(\x)$, and we can perform the same analysis by inspection to retrieve the values for the corresponding parameters $\bmu_x$ and $\Sigma_x.$

To acquire an expression for $g(\x)$, we consider all the remaining terms:

$$
\begin{align*}
g(\x) &= -\frac{1}{2} (\x - \bmu_x)\T\Lambda_{xx} (\x - \bmu_x) + \x\T\Lambda_{xy} \bmu_y + \frac{1}{2}\m\T \Lambda_{yy}\inv \m \\\\[3pt]
&= -\frac{1}{2} \x\T\Lambda_{xx} \x + \x\T \left( \Lambda_{xx} \bmu_x + \Lambda_{xy} \bmu_y \right) \\\\ 
&\\quad + \frac{1}{2} \bigg[ \big( \Lambda_{yy} \bmu_y - \Lambda_{yx}(\x-\bmu_x) \big)\T \Lambda_{yy}\inv \big( \Lambda_{yy} \bmu_y - \Lambda_{yx}(\x-\bmu_x) \big) \bigg].
\end{align*}
$$

Then, expanding this and dropping all constant terms with respect to $\x$, we have

$$
\begin{align*}
g(\x) &= -\frac{1}{2}\x\T \left( \Lambda_{xx} + \Lambda_{xy}\Lambda_{yy}\inv\Lambda_{yx} \right) \x + \x\T\left( \Lambda_{xx} + \Lambda_{xy}\Lambda_{yy}\inv\Lambda_{yx} \right) \bmu_x \\\\[2pt]
&= -\frac{1}{2}\x\T \Sigma_{xx}\inv \x + \x\T\Sigma_{xx}\inv\bmu_x,
\end{align*}
$$

where we have

$$
\Sigma_{xx} = \left( \Lambda_{xx} + \Lambda_{xy}\Lambda_{yy}\inv\Lambda_{yx} \right)\inv
$$

from the matrix inversion formula. Comparing this to our general form in $(7)$, we have the following expressions for the mean and covariance of the marginal $p(\x)$:

$$
\begin{align}
\E[\x] &= \bmu_x \\\\
\cov[\x] &= \Sigma_{xx}.
\end{align}
$$

That is, the mean and covariance of the marginal distribution are found by simply taking the "slices" of the partitioned matrices from the joint distribution which correspond to the marginal variable.



<details>
  <summary>Example: Conditioning vs. marginalization</summary>
  <p>
  Let's consider a bivariate Gaussian distribution and see what happens when we condition vs. marginalize.
  </p>
</details>


## Bayes' rule for linear Gaussian systems

Another useful result is Bayes' rule for linear Gaussian systems.[^fn5] In the previous sections, we had a random vector which specified a joint Gaussian distribution over $d$ random variables, and we wanted to find expressions for the marginal and conditional distributions of subsets of these random variables. Instead, suppose we are given a marginal distribution $p(\x)$ and a conditional distribution $p(\y \mid \x)$ and we wish to know the marginal distribution $p(\y)$ and the conditional $p(\x \mid \y).$ This can be seen as an application of Bayes' theorem. 

Specifically, consider the two random vectors $\x$ and $\y$ given by

$$
\begin{align*}
p(\x) &= \Norm(\x \mid \bmu, \Lambda\inv), \\\\[2pt]
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
\log p(\z) &= \log p(\y \mid \x) + \log p(\x) \nonumber \\\\
&= -\frac{1}{2} \bigg[ \left( \y - \A\x-\b \right)\T\L\left( \y - \A\x - \b \right) \nonumber \\\\
& \quad\qquad + (\x - \bmu)\T\Lambda(\x - \bmu) \bigg] + c 
\end{align}
$$

We see that the exponent of the joint distribution is quadratic in terms of $\x$ and $\y$, so $p(\z)$ will be a Gaussian. Then, let's again collect the quadratic and linear terms to find the parameters of the joint distribution. First, expanding $(16)$, we have

$$
\begin{align*}
& -\frac{1}{2} \y\T\L\y + \y\T\L\A\x - \frac{1}{2}\x\T\A\T\L\A\x + \y\T\L\b - \x\T\A\T\L\b \\\\
&- \frac{1}{2}\x\T\Lambda\x + \x\T\Lambda\bmu\ + c.
\end{align*}
$$

We see that the quadratic terms are given by

$$
\begin{align*}
& -\frac{1}{2}\x\T\Lambda\x - \frac{1}{2}\y\T\L\y - \frac{1}{2}\x\T\A\T\L\A\x + \y\T\L\A\x \\\\[3pt]
&= -\frac{1}{2}\x\T \left( \Lambda + \A\T\L\A \right) \x + \y\T\L\A\x - \frac{1}{2}\y\T\L\y \\\\[3pt]
&= -\frac{1}{2}
\begin{pmatrix}
\x \\\\
\y
\end{pmatrix}\T
\begin{pmatrix}
\Lambda + \A\T\L\A & -\A\T\L \\\\
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
\Lambda + \A\T\L\A & -\A\T\L \\\\
-\L\A & \L
\end{pmatrix}.
$$

Thus, $\mathbf{R}$ is the precision matrix of the joint distribution $p(\z)$, and hence the covariance matrix $\Sigma_z$ can be found by the matrix inversion formula:

$$
\Sigma_z = \mathbf{R}\inv = \begin{pmatrix}
\Lambda\inv & \Lambda\inv\A\T \\\\
\A\Lambda\inv & \L\inv + \A\Lambda\inv\A\T
\end{pmatrix}.
$$

Similarly, we can collect the linear terms to identify the mean:

$$
\begin{align*}
\x\T\Lambda\bmu + \y\T\L\b - \x\T\A\T\L\b &= \x\T \left( \Lambda\bmu - \A\T\L \right) \b + \y\T\L\b \\\\[3pt]
&= \begin{pmatrix}
\x \\\\
\y
\end{pmatrix}\T
\begin{pmatrix}
\Lambda\bmu\ - \A\T\L\b \\\\
\L\b
\end{pmatrix}.
\end{align*}
$$

Again comparing this to $(7)$ we have that

$$
\Sigma_z\inv\bmu_z = 
\begin{pmatrix}
\Lambda\bmu\ - \A\T\L\b \\\\
\L\b
\end{pmatrix}.
$$

Then, the mean is given by

$$
\begin{align*}
\bmu_z &= \Sigma_z \begin{pmatrix}
\Lambda\bmu\ - \A\T\L\b \\\\
\L\b
\end{pmatrix} \\\\[3pt]
&= \begin{pmatrix}
\Lambda\inv & \Lambda\inv\A\T \\\\
\A\Lambda\inv & \L\inv + \A\Lambda\inv\A\T
\end{pmatrix}
\begin{pmatrix}
\Lambda\bmu\ - \A\T\L\b \\\\
\L\b
\end{pmatrix} \\\\[3pt]
&= \begin{pmatrix}
\bmu\ \\\\
\A\bmu\ + \b
\end{pmatrix}
\end{align*}
$$

Now, we can obtain the parameters of the marginal distribution $p(\y)$ using $(14)$ and $(15)$:

$$
\begin{align*}
\bmu_y = \A\bmu + \b, \\\\
\Sigma_y = \L\inv + \A\Lambda\inv\A\T.
\end{align*}
$$

Similarly, we can use our previously derived rules to get an expression for the conditional distribution $p(\x \mid \y)$. In this case, it will be easier to use our expressions for the conditional parameters in terms of the precision matrix. From $(8)$, the conditional covariance is

$$
\begin{align*}
\Sigma_{x\mid y} &= \bR_{xx}\inv \\\\
&= \left( \Lambda + \A\T\L\A \right)\inv,
\end{align*}
$$

and, using $(9)$, the conditional mean is given by

$$
\begin{align*}
\bmu_{x\mid y} &= \bmu - \bR_{xx} \inv \bR_{xy}(\y - \bmu_y) \\\\
&= \bmu + (\Lambda + \A\T\L\A)\inv \A\T\L (\y - \A\bmu - \b) \\\\
&= \bmu + (\Lambda + \A\T\L\A)\inv \bigg[ \A\T\L (\y - \b) + \Lambda\bmu \bigg].
\end{align*}
$$


## Maximum likelihood estimation

In practice, we rarely know the true underlying distribution from which data is generated. The goal of maximum likelihood estimation (MLE) is to estimate the true values of the parameters of a distribution from an observed set of data. Suppose we have $n$ i.i.d. samples from a Gaussian distribution $\X = (\x_1, \x_2 \ldots, \x_n)$, and we would like to estimate the mean and covariance of the underlying distribution. This is done by maximizing the likelihood function

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
\begin{gather*}
\pbx \A\inv = - \A\inv \frac{\partial\A}{\partial\x} \A\inv \\\\[2pt]
\frac{\partial}{\partial\A} \tr(\A\B) = \B\T \\\\[5pt]
\frac{\partial}{\partial\A} \log \lvert \A \rvert = (\A\inv)\T.
\end{gather*}
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
\E[\Sigma\ml] &= \frac{1}{n} \E \left[ \sum_{i=1}^n (\x_i - \bmu\ml)(\x_i - \bmu\ml)\T \right] \nonumber \\\\
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
&\propto \exp\left(-\frac{1}{2} \sum_{i=1}^n \frac{(x_i - \mu)^2}{\sigma^2} + \frac{(\mu - \mu_0)^2}{n\sigma_0^2} \right) \nonumber
\end{align}
$$

The sum inside the exponential is then

$$
\begin{align*}
&\quad \sum_{i=1}^n \frac{n\sigma_0^2 \left( x_i^2 - 2x_i\mu + \mu^2  \right) + \sigma^2 \left( \mu^2 - 2\mu\mu_0 + \mu_0^2 \right)}{n\sigma_0^2\sigma^2} \\\\
&= \frac{1}{n\sigma_0^2\sigma^2} \sum_{i=1}^n n\sigma_0^2x_i^2 - 2n\sigma_0^2x_i\mu + n\sigma_0^2\mu^2 + \sigma^2\mu^2 - 2\sigma^2\mu\mu_o + \sigma^2\mu_0^2 \\\\
&= \frac{1}{n\sigma_0^2\sigma^2} \bigg( n^2\sigma_0^2\mu^2 + n\sigma^2\mu^2 - 2n\sigma^2\mu\mu_0 + n\sigma^2\mu_0^2 - 2n\sigma_0^2\mu \sum_{i=1}^n x_i \\\\
&\qquad\qquad\qquad  + n\sigma_0^2\sum_{i=1}^n x_i^2 \bigg) \\\\[3pt]
&= \frac{1}{n\sigma_0^2\sigma^2} \left[ (n^2\sigma_0^2 + n\sigma^2)\mu^2 - 2n\left(\sigma^2\mu_0 + \sigma_0^2\sum_{i=1}^n x_i\right) \mu + c  \right]
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
\mu_n &= \frac{\sigma^2}{n\sigma_0^2 + \sigma^2}\mu_0 + \frac{n\sigma_0^2}{n\sigma_0^2 + \sigma^2} \mu\ml \nonumber \\\\[8pt]
&= \frac{1}{\sigma_n^2} \left( \frac{1}{\sigma_0^2} \mu_0 + \frac{n}{\sigma^2} \mu\ml \right), \label{eq:mun} \\\\[10pt]
\frac{1}{\sigma_n^2} &= \frac{n\sigma_0^2 + \sigma^2}{\sigma_o^2\sigma^2} \nonumber \\\\[4pt]
&= \frac{n}{\sigma^2} + \frac{1}{\sigma_0^2}. \label{eq:sigman}
\end{align}
$$

To recap, we started with some initial belief about the parameter $\mu$, which was represented by the prior $p(\mu)$. For example, the prior mean $\mu_0$ might reflect our belief of a reasonable value for $\mu$, and the prior variance $\sigma_0^2$ might reflect how certain we are in that belief. Then, after observing $n$ samples, we updated our expressions for these parameters using $\eqref{eq:mun}$ and $\eqref{eq:sigman}$.

There are several interesting things to note about $\eqref{eq:mun}$ and $\eqref{eq:sigman}$. First, we see that the posterior mean $\mu_n$ is a weighted average between the prior mean $\mu_0$ and the mle $\mu\ml$. When $n=0$, we simply have the prior mean $\mu_n = \mu_0$. As $n \to \infty$, the posterior mean approaches the mle $\mu_n \to \mu\ml$. 

Furthermore, we have expressed the posterior variance in terms of the precision:

$$
\lambda_n = \frac{1}{\sigma_n^2}.
$$

We see that, as $n$ gets big, the precision grows, and hence the variance gets small. That is, with more observations, we become more certain in our estimation of the parameters. Moreover, it's interesting to note that the precision gets updated additively. For each observation of the data, we increase prior precision $1/\sigma_0^2$ by one increment of the data precision $1/\sigma^2$.

Finally, it's interesting to note that as $n \to \infty$, not only does $\mu_n \to \mu\ml$, but $\sigma_n^2 \to 0$ as well. Thus, the posterior distribution in the limit of infinitely many observations is the delta function centered at $\mu\ml$. Thus, the maximum likelihood estimator is recovered in the Bayesian formulation.


### Unknown mean, known variance (multivariate case)

Here, I extend the previous results to the multivariate case.


### Known mean, unknown variance (univariate case)


### Known mean, unknown variance (multivariate case)


### Unknown mean, unknown variance (univariate case)


### Unknown mean, unknown variance (multivariate case)




## References and further reading

The content of this post largely follows section 2.3 of Bishop's *Pattern Recognition and Machine Learning*. However, the aim of this blog was to supplement this with detailed derivations, provide my own insights, and extend the discussion on certain points.

The other resource I used was Murphy's *Probabilistic Machine Learning: An Introduction*, which I primarily referenced for discussions on many of the mathematical concepts which can be found detailed in the appendix below. Again, any material from this book which I've included, I aimed to supplement and extend.

Besides that, I made use of many helpful Math Stack Exchange posts, reddit discussions, and Wikipedia articles.

## Appendix

### Eigenvalue decomposition

### Change of variables

### Sum and product rules of probability

### The Schur complement

### Completing the square

### Matrix derivatives

1. 

### Short proofs


#### 1.

Show that uncorrelated does not imply independent. However, in the case of a Gaussian, it does.

#### 2.

Show that if a matrix $\A$ is real and symmetric, then its eigenvalues are real, and the eigenvectors can be chosen to form an orthonormal set.

#### 3.

Show that the inverse of a symmetric matrix is symmetric. We'll use this to argue that the precision matrix $\Lambda$ is symmetric.



[^fn1]: Here I've used the Kronecker delta for notational simplicity:
$$
\delta_{ij} = \begin{cases}
1 &\ i=j, \\\\
0 &\ i\neq j.
\end{cases}
$$

[^fn2]: Note that since $\U$ is an orthogonal matrix, a linear transformation defined by $\U$ preserves the length of the vector which it transforms, and thus is either a rotation, reflection, or a combination of both.

[^fn3]: Most literature notation uses $\Lambda$ for the precision matrix. Not that we have overdefined $\Lambda$, since it also refers to the matrix containing the eigenvalues of $\Sigma$ in our definition for the eigenvalue decomposition. However, for the rest of this blog, $\Lambda$ will refer to the precision matrix.

[^fn4]: To see this, consider the product between two vectors $\a \in \R^m$ and $\b \in \R^n$ defined by $\a\T\A\b$ for some matrix $\A \in \R^{m \times n}.$ Then, since the resulting product is a scalar, we have $\a\T\A\b  = \b\T\A\T\a.$

[^fn5]: This rule is particularly useful for Bayesian approaches to machine learning in which we model observations of some underlying distribution with additive Gaussian noise. For example, consider the case of linear regression, where observations are related via the function $f(\x) = \w\T\x.$ We can model noisy predictions by adding a zero-mean Gaussian random variable: $y = f(\x) + \epsilon$, where $\epsilon \sim \Norm(0, \sigma^2).$ Then, the observations are themselves Gaussian with mean $f(\x)$ and variance $\sigma^2$: $p(y \mid \x, \w) = \Norm(y \mid \w\T\x, \sigma^2).$

[^fn6]: Note that since the logarithm is monotonic, the maximizer of $\log f(x)$ is equivalent to that of $f(x)$, so finding the parameters which maximize the log-likelihood is equivalent to find those which maximize the likelihood. Furthermore, we often choose to *minimize* the NLL as opposed to *maximizing* the log-likelihood, as this is often treated as a sort of loss function, and many modern optimization frameworks for machine learning are designed to minimize loss objectives.

[^fn7]: $\bS$ is symmetric because it is formed by taking a sum of symmetric matrices. Each matrix in the sum is symmetric because it is the outer product of a vector with itself, which always forms a symmetric PSD matrix as long as the vectors are nonzero.

[^fn8]: This is a common strategy which we've already seen in many of the previous derivations. Instead of carrying normalization constants throughout the procedure, we will find the form of the posterior distribution. Then, we can compute the normalization constant at the end. This will be easy if we find that the posterior takes the form of a known distribution, in which the normalization constant can be found by inspection.

[^fn9]: Given a parameter $\theta$ and a dataset $\mathcal{D}$, we call the prior distribution $p(\theta)$ a *conjugate prior* to the likelihood $p(\mathcal{D} \mid \theta)$ if the posterior $p(\theta \mid \mathcal{D})$ takes the same functional form as the prior. Since it is up to us to choose the prior distribution, it is often desirable to choose a conjugate prior, since it simplifies the computation of the posterior.