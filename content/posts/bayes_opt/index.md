---
title: "Foundations of Bayesian Optimization"  
date: ""  
summary: ""  
description: ""  
draft: true  
toc: false  
readTime: true  
autonumber: false  
math: true  
tags: ["code, drug discovery"]
showTags: false  
hideBackToTop: false
---


## Background

* Goal is to maximize some function $f$, i.e., find

$$
\begin{align*}
x^* \in \arg \max_{x \in \mathcal{X}}
\end{align*}
$$

* $f$ need not be analytic (i.e., can't write it down), nor computable - we just need some access to information at specified points