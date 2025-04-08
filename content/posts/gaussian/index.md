---
title: "Some math behind the Gaussian distribution"  
date: "2025-03-03"  
summary: ""  
description: ""  
draft: true  
toc: false  
readTime: true  
autonumber: false  
math: true  
tags: []
showTags: false  
hideBackToTop: false
---



<style>
  details {
    border: 1px solid black;
    border-radius: 8px;
    padding: 0.5em 0.5em 0;
    margin-bottom: 1em;
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
  }
  
  details[open] {
    padding: 0.5em;
  }
  
  .example-content {
    margin-top: 1em;
  }
</style>

The Gaussian distribution is 

but, despite its ubiquity in ML, I have frequently found myself in a state somewhere between *discomfort* and *panic* each time I am faced with the task of manipulating it.

In this blog, I begin by reasoning about the shape and behavior of the Gaussian distribution in multiple dimensions. I then derive some useful formulas, such as conditioning, marginalization, and Bayes' rule with Gaussians. I aim to provide thoroughness in the math, taking particular care to clearly articulate the concepts which gave me trouble upon a first, second, or sometimes third reading. Along the way, I also provide some examples; some in math, some in code.


## Properties of the Gaussian distribution

In this section we'll start by considering some basic properties of the Gaussian distribution. First, we'll show that surfaces on which the likelihood is constant form ellipsoids. Then, we'll conclude that the multivariate Gaussian distribution is indeed normalized, making it a valid probability distribution. Finally, we'll consider the first- and second-order moments of the multivariate Gaussian distribution in order to provide an interpretation of its parameters.

We write the Gaussian distribution for a random vector $\x \in \R^d$ as

$$
\normal(\x \mid \mu, \Sigma) = \frac{1}{(2\pi)^{d/2} \lvert \Sigma \rvert^{1/2}}\exp\left( -\frac{1}{2} (\x - \mu)\T\Sigma^{-1}(\x - \mu)\right),
$$

where $\bmu \in \R^d$ is the mean vector and $\Sigma \in \R^{d\times d}$ is the covariance matrix. The quadratic form in the exponent is known as the *Mahalanobis distance*, defined

$$
\begin{equation}
\Delta^2 = (\x - \mu)\T\Sigma^{-1}(\x-\bmu).
\end{equation}
$$

This is often nicer to work with when manipulating the Gaussian, as opposed to the exponential function. 

Since the covariance matrix $\Sigma$ is real and symmetric, we can perform [eigenvalue decomposition](#eigenvalue-decomposition) to write it in the form

$$
\Sigma = \sum_{i=1}^d\lambda_i\u_i\u_i\T,
$$


where $\\{\u_i\\}\_{i=1}^d$ are eigenvectors of $\Sigma$, and $\\{\lambda_i\\}\_{i=1}^d$ are the corresponding eigenvalues. Note that we can choose the eigenvectors to be orthonormal (see ), i.e., [^fn1]
$$
\u_i\T\u_j = \delta_{ij}.
$$
Moreover, we can easily write the inverse of the covariance matrix as
$$
\Sigma\inv = \sum_{i=1}^d\frac{1}{\lambda_i}\u_i\u_i\T,
$$

Substituting this into $(1)$, we get

$$
\begin{align*}
\Delta^2 &= \sum_{i=1}^d \frac{1}{\lambda_i}(\x - \bmu)\T\u_i\u_i\T(\x - \bmu) \\\\
&= \sum_{i=1}^d\frac{y_i^2}{\lambda_i},
\end{align*}
$$

where we've introduced
$$
y_i = \u_i\T(\x - \bmu),
$$
or
$$
\y = \U(\x - \bmu),
$$
where $\U$ is the matrix whose rows are given by $\u_i\T$. Note that $\U$ is an orthogonal matrix, so $\U\U\T = \I,$ and thus $\U\T = \U\inv$. The set $\\{y_i\\}$ can then be seen as a transformed coordinate system, shifted by $\bmu$ and rotated by $\U$.[^fn2]

All of the dependence of the Gaussian on $\x$ is determined by $\Delta^2$. Thus, it is constant on surfaces for which $\Delta^2$ is constant. Then, let
$$
\Delta^2 = \sum_{i=1}^d\frac{y_i^2}{\lambda_i} = r
$$
for some constant $r$. This defines the equation of an ellipsoid in $d$ dimensions. For example, if $d=2,$ we have the equation
$$
\frac{y_1^2}{\lambda_1} + \frac{y_2^2}{\lambda_2} = r,
$$
which gives a 2d ellipse centered at the origin with semi-major and semi-minor axes given by $\sqrt{\lambda_1r}$ and $\sqrt{\lambda_2r}$.


<details>

  <summary>Example</summary>


</details>

Next, our goal is to show that the multivariate Gaussian distribution is normalized. Let's consider the Gaussian in the new coordinate system $\{y_i\}$. To transform from $\x$-space to $\y$-space, we use the [change of variables formula](#change-of-variables), given by
$$
\begin{align*}
p_y(\y) &= p_x(\x)\lvert \J \rvert \\
&= p_x(g(\y))\lvert \J \rvert,
\end{align*}
$$
where $\x = g(\y)$ defines the transformation. Here, we have the Jacobian $\J$, whose elements are given by
$$
J_{ij} = \frac{\partial x_i}{\partial y_j}.
$$
The relationship between $\x$ and $\y$ is given by
$$
\y = \U(\x - \bmu),
$$
or
$$
\x = \U\T\y + \bmu.
$$
Thus, the derivative of $\x$ with respect to $\y$ is given by $\U\T$, hence the Jacobian is given by
$$
\J = \U\T,
$$
or
$$
J_{ij} = U_{ji}.
$$Then, to find the determinant of the Jacobian, we have
$$
\begin{align*}
\lvert \J \rvert ^2&= \lvert \U\T \rvert ^2 \\\\
&= \lvert \U\T \rvert \lvert \U \rvert \\\\
&= \lvert \U\T\U \rvert \\\\
&= \lvert \I \rvert \\\\
&= 1.
\end{align*}
$$
Thus, $\lvert \J \rvert = 1$, making our transformation
$$
p_y(\y) = p_x(g(\y)).
$$
Then, the Gaussian in terms of $\y$ is given by
$$
\begin{align*}
p_y(\y) &= \frac{1}{(2\pi)^{d/2}\lvert\Sigma\rvert^{1/2}}\exp\left( -\frac{1}{2} (\x - \bmu)\T\Sigma\inv(\x - \bmu) \right).
\end{align*}
$$
First, it's useful to show that
$$
\begin{align*}
\lvert \Sigma \rvert &= \lvert \U\T\Lambda\U \rvert \\\\
&= \lvert \U\T \rvert \lvert \Lambda \rvert \lvert \U \rvert \\\\
&= \lvert \Lambda \rvert \\\\
&= \prod_{i=1}^d \lambda_i,
\end{align*}
$$
hence
$$
\frac{1}{\lvert \Sigma \rvert^{1/2}} = \prod_{i=1}^d \frac{1}{\sqrt{\lambda_i}}.
$$
Examining at the term in the exponent, we have
$$
\begin{align*}
(\x - \bmu)\T\Sigma\inv(\x - \bmu)  &= (\U\T\y)\T \Sigma\inv(\U\T\y) \\\\
&= \y\T\U (\U\T\Lambda\U)\inv \U\T\y \\\\
&= \y\T\U \U\inv\Lambda (\U\T)\inv \U\T\y \\\\
&= \y\T\Lambda\y.
\end{align*}
$$
Then,
$$
\begin{align*}
p_y(\y) &= \frac{1}{(2\pi)^{d/2}\lvert \Sigma \rvert ^{1/2}} \exp \left( -\frac{1}{2} \y\T\Lambda\y \right) \\\\
&= \frac{1}{(2\pi)^{d/2}\lvert \Sigma \rvert ^{1/2}} \exp \left( -\frac{1}{2} \sum_{i=1}^d \frac{y_i^2}{\lambda_i} \right).
\end{align*}
$$
Using our expression for the determinant of the covariance matrix, and noting exponent of a sum becomes a product of exponents, we have
$$
p_y(\y) = \prod_{i=1}^d \frac{1}{\sqrt{2\pi\lambda_i}} \exp \left( -\frac{y_i^2}{2\lambda_i} \right).
$$
Then,
$$
\begin{align*}
\int_\y p_y(\y) d\y &= \prod_{i=1}^d \int_{y_i}  \frac{1}{\sqrt{2\pi\lambda_i}} \exp \left( -\frac{y_i^2}{2\lambda_i} \right) dy_i.
\end{align*}
$$
We see that each element of the product is just a univariate Gaussian over $y_i$ with variance $\lambda_i$, each of which integrates to 1, showing that $p_y(\y)$, and thus $p_x(\x)$ is indeed normalized.

Finally, we will examine the first and second moments of the Gaussian.


## Appendix

### Eigenvalue decomposition

### Change of variables

### The Schur complement

### Completing the square

### Short proofs

1. Show that if a matrix $\A$ is real and symmetric, then its eigenvalues are real, and the eigenvectors can be chosen to form an orthonormal set.
2. Show that the inverse of a symmetric matrix is symmetric. We'll use this to argue that the precision matrix $\Lambda$ is symmetric.



[^fn1]: Here I've used the Kronecker delta for notational simplicity:
$$
\delta_{ij} = \begin{cases}
1 &\ i=j, \\\\
0 &\ i\neq j.
\end{cases}
$$

[^fn2]: Note that, since $\U$ is an orthogonal matrix, a linear transformation defined by $\U$ preserves the length of the vector which it transforms, and thus is either a rotation, reflection, or a combination of both.