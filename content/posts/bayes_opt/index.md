---
title: "Bayesian Optimization for Molecular Design"  
date: "2025-02-01"  
summary: ""  
description: ""  
draft: true  
toc: true  
readTime: true  
autonumber: false  
math: true  
tags:
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



