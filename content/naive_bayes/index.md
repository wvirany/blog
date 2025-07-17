---
title: "Naive Bayes"
draft: false
math: true
comments: false
---

First, let's establish the Naive Bayes classification problem.

Supppose we are given a dataset of $n$ iid observations $\cD = \\{\(\x^{(i)}, y_i)\\}_{i=1}^n$, where each $\x^{(i)}$ is some $d$-dimensional input vector and $y_i \in \\{1, 2, \ldots, K\\}$ is the corresponding class label. Given a new input $\x$, the goal of classification is to correctly assign $\x$ to one of $K$ classes.

Let's distinguish between two approaches to classification: *discriminative* vs. *generative* modeling.

In the **discriminative modeling** approach, we wish to model the distribution over class labels $p(y \mid \x)$. An example of this would be using a generalized linear model (such as logistic regression) to predict class labels given the features. Thus, given an input $\x$, we should be able to assign probabilities to each of the $K$ classes, and choose the $y$ which maximizes this probabilitiy.

In the **generative modeling** approach, we wish to model the data generation process. That is, using Bayes' rule, we can write

$$
\begin{equation}\label{1}
p(y \mid \x) \propto p(\x \mid y)p(y).
\end{equation}
$$

Instead of skipping straight to the step of modeling the class probabilities, we aim to model the class-conditional probability densities $p(\x \mid y)$ and the prior class probabilities $p(y)$ for each class $K$. One might ask: why bother? Well, in doing so, we gain the ability to generate new data points by first sampling $y \sim p(y)$, then $x \sim p(\x \mid y)$.

Naive Bayes takes a generative modeling approach to classification.

There are many ways to estimate the densities $p(\x \mid y)$ and $p(y)$. One such method is maximum likelihood estimation (MLE), where we assume a particular parametric form of each distribution---for example, $\x$ might be drawn from a Gaussian distribution, with parameters $\bmu_k$ and $\bSigma_k$, and $y$ might be drawn from a multinomial distribution with $K$ classes, with probabilities $p(y = k) = \pi_k$. Then, we could find the parameters which maximize the likelihood of the observed dataset and use these to estimate the probability densities. However, for the sake of understanding Naive Bayes, I won't focus on density-estimation techniques here.

**Why Naive Bayes?**

Let's assume that the features of $\x$ are binary. Then, we can write the joint distribution over the features of $\x$, conditioned on $y$ as

$$
\begin{equation}\label{2}
p(\x \mid y) = p(x_1, x_2, \ldots, x_d \mid y),
\end{equation}
$$

where each $x_i \in \\{0, 1\\}$. If we want to estimate the probability of $\x$ being in a particular class $y$, we would need to assign a probability to each possible configuration of $\x$ --- there are $2^d$ possible such configurations! This requires estimating $2^d - 1$ parameters for each class.[^fn1]  In order to gain an accurate estimation for such a distribution, we would need a lot of data ---  this is an example of the **curse of dimensionality**. Thus, we'd like to reduce the dimensionality of the feature space.


### Conditional independence

The "naive" assumption in Naive Bayes is that the features of our inputs are conditionally independent. Under this assumption, we can factor the joint distribution in $\eqref{2}$ as

$$
p(x_1, \ldots, x_d \mid y) = p(x_1 \mid y) \cdots p(x_2 \mid y).
$$

One way to say this is "$x_1, x_2, \ldots, x_d$ are independent, conditioned on $y$." Now, we can model each individual distribution $p(x_i \mid y)$ independently. In the case of binary features, this only requires $d$ parameters for each class instead of $2^d-1$.

### When does Naive Bayes work?

Let's consider two classification problems:

**Example: Sentiment analysis**

Suppose we are classifying restaurant reviews as either positive $P$ or negative $N$. To do so, we'll use the following words as features:

* $D$ - "delicious"
* $T$ - "terrible"
* $F$ - "fast"
* $S$ - "slow"

where each variable is binary indicating whether or not the corresponding word is present in a review. Then, suppose we have a training dataset containing 10 positive and 10 negative reviews, i.e., $P(P) = P(N) = 0.5$. Moreover, suppose we observe the above features with the following probabilities:

$$
\begin{align*}
&P(D \mid P) = 0.6, \quad P(D \mid N) = 0 \\\\
&P(T \mid P) = 0, \quad \\; \\;\\; P(T \mid N) = 0.5 \\\\
&P(F \mid P) = 0.5, \quad P(F \mid N) = 0.2 \\\\
&P(S \mid P) = 0.1, \quad P(S \mid N) = 0.6
\end{align*}
$$

This corresponds to observing "delicious" in 6/10 of the positive reviews, and none of the negative reviews, and so on.

Now consider a new review: "delicious food and fast service". From Bayes' rule, we have

$$
\begin{align*}
P(P \mid D, S) &= \frac{P(D, S \mid P) \\, P(P)}{P(D, S)} \qquad\qquad\qquad\qquad \\\\[10pt]
\text{\scriptsize (naive Bayes assumption)} \qquad\qquad\qquad &= \frac{P(D \mid P) \\, P(S \mid P) \\, P(P)}{P(D, S)}
\end{align*}
$$

We can evaluate the denominator using the law of total probability:

$$
\begin{align*}
P(D, S) &= P(D, S \mid P) \\, P(P) + P(D, S \mid N) \\, P(N) \\\\[6pt]
&= P(D \mid P) \\, P(S \mid P) \\, P(P) + P(D \mid N) \\, P(S \mid N) \\, P(N) \\\\[6pt]
&= 0.6 * 0.1 * 0.5 = 0.03
\end{align*}
$$


Then, the Naive Bayes model predicts the probability of the review being positive as

$$
P(P \mid D, S) = \frac{0.6 * 0.1 * 0.5}{0.03} = 1.
$$

This seems reasonable! Although, I will point out a caveat: since we never saw a negative review with the word "delicious", then we will always predict a review with this word as positive, noting that

$$
P(N \mid D, S) \propto P(D \mid N) \\, P(S \mid N) \\, P(N)
$$

and $P(D \mid N) = 0$. To overcome this, a common strategy is to add a small value to the frequency of each word --- for example, by adding 1 to the number of times we observed each word, and recomputing the probabilities to maintain normalization. Thus, if we see a new sample that was not in our training set, we won't assign it a probability of exactly 0.

**Example: Medical diagnosis**

Consider the task of diagnosing heart disease based on the following symptoms:

* $C$ - chest pain
* $F$ - fatigue
* $S$ - shortness of breath

Furthermore, let $H$ indicate whether or not the patient has heart disease. Given a patient with any combination of these symptoms, we wish to diagnose their condition.

Now, suppose we know $P(H) = 0.05$ and $P(\neg H) = 0.95$. That is, 5% of the population suffers from heart disease. Moreover, suppose we know that for people with heart disease:

$$
\begin{gather*}
&P(C \mid H) = 0.8 \\\\
&P(F \mid H) = 0.6 \\\\
&P(S \mid H) = 0.7
\end{gather*}
$$

and that for people without heart disease:

$$
\begin{gather*}
&P(C \mid \neg H) = 0.05 \\\\
&P(F \mid \neg H) = 0.2 \\\\
&P(S \mid \neg H) = 0.1
\end{gather*}
$$

If a patient has heart disease, the probability that they suffer from a given symptom is high. On the other hand, the probability that a healthy patient suffers from a given symptom is low.

Now, suppose we have a patient who is suffering from all three symptoms. Using the same process as the previous example, Naive Bayes predicts

$$
\begin{align*}
P(H \mid C, F, S) &= \frac{P(C, F, S \mid H) \\, P(H)}{P(C, F, S)} \\\\[10pt]
&= \frac{0.0168}{0.0168 + 0.00095} \\\\[10pt]
&\approx 0.95
\end{align*}
$$

In other words, there is approximately a 95% chance that the patient has heart disease!

However, there is a problem with this model. In reality, if a patient is suffering from any one of the symptoms, they are probably also suffering from the others. For example, if we know that someone has chest pain, learning that they also have shortness of breath is less surprising --- however, the model assumes this is independent evidence!

This shows an example of where the Naive Bayes assumption breaks down due to **highly correlated features**. The model essentially "double counts" features and overestimates the probability that the patient has heart disease.


### Conclusion

We saw that Naive Bayes makes a strong simplifying assumption --- conditional independence --- which helps to reduce the complexity of the problem. While this assumption might not be entirely accurate, there are some scenarios in which it's reasonable.

In the sentiment analysis problem, words like "delicious" and "fast" describe different aspects of the quality of the restaurant - it's reasonable to assume these are independent. On the other hand, in the medical diagnosis problem, different symptoms were strongly correlated, so the independence assumption was invalid.

Another way to think about this is in terms of **information gain** - how much new evidence each feature provides. In sentiment analysis, learning that a review contains 'fast' gives you roughly the same amount of information about positivity regardless of whether you already know it contains 'delicious'. Each word contributes independent evidence.

However, in the medical diagnosis problem, the information gain from each symptom depends heavily on what you already know. If you observe chest pain first, then learning about shortness of breath adds relatively little new information - you're already suspecting heart problems. But if shortness of breath were your first observation, it would provide substantial evidence. The problem is that Naive Bayes treats both scenarios identically, always giving shortness of breath the same 'weight' regardless of context.

**Key takeaway:** Naive Bayes works well when the features provide roughly the same amount of information regardless of what features you've already observed. This is the conditional independence assumption.


[^fn1]: Estimating a discrete probability distribution with $n$ possible inputs only requires $n-1$ parameters due to the summation constraint:
$$
\sum_{i=1}^n p_i = 1,
$$
where $p_i$ is the probability of the $i^{\text{th}}$ event. The $n^{\text{th}}$ probability is defined implicitly as $1 - \sum_{i=1}^{n-1}p_i$.