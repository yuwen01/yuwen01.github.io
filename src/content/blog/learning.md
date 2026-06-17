---
title: "Things I'm learning"
date: 2026-06-16
hidden: true
---

One of my goals is to improve a little bit every day. The exact method doesn't really matter. Something something exponential, compounding, etc. 

To improve my retention, I will write down a couple things I learn every week. Lots of these will be transcriptions from wikipedia, with maybe a couple small comments.

## 6/15/2026 - 6/19/2026

### QR Decomposition

For a matrix $A$, there exists a decomposition $A=QR$ such that $Q$ is orthonormal and $R$ is upper triangular. I'm just going to assume $A \in \Reals^{n \times n}$ for simplicity.

#### Gram Schmidt

The simplest way to construct this is with Gram Schmidt. 

1. Express $A$ as a bunch of columns $[a_1, a_2, ... a_n]$.
2. Let $u_1 := a_1$ and $e_1$ be $u_1$ normalized.
3. Let $u_2 := a_2 - proj_{u_1}(a_2)$. $u_2$ is definitely orthogonal to $u_1$, since we got rid of the component that is not orthogonal. Then define $e_2$ also the same way. 
4. Continue to get $u_n, e_n$. 

So now $[e_1, e_2, ... e_n]$ is an orthonormal basis. 

Also, $a_1 = ||u_1|| * e_1$, and $a_2 = ||u_2|| * e_2 + \langle a_2, u_1 \rangle * e_1$, etc. Like $a_i$ is a linear combination of $e_j$ with $j \leq i$, so then R is upper triangular. 

When $a_n$ is very close to $\sum{proj({\text{whatever}})}$, that is a problem. 

#### Numerical instability

TLDR whenever you have a float or whatever, you can think of the float itself representing a potentially very small range. For example, in fp64, you get 52 (53?) bits of mantissa. So you only get like 16 ish digits of significant figures. Everything that isn't captured by those significant figures is called "noise floor".

The problem in the gram schmidt case is that the $\sum{proj({\text{whatever}})}$ term amplifies a lot of existing noise through multiplications and divisions by small numbers. There's this thing called sterbenz' lemma, which says that if you subtract two values that are within a factor of two from each other, you lose no precision from the subtraction. The left hand side of the subtraction is fine. But the right hand side has all of this amplified error, which is carried into the final result. So that is a problem in this gram-schmidt process.

There are some other common failure modes for floating point arithmetic. For example, if you try to add a small number to a big number, the small number's sig figs disappear.

#### Householder reflections

todo

#### making it go on gpu

i feel like my last job was mostly about making not-super-gpu-friendly operations go fast on gpu so i want to try on qr decomposition.

### Rejection sampling in speculative decoding

Paper link that someone sent in slack: [link](https://arxiv.org/pdf/2606.12370).

So I tried to read this and got hung up in the preliminaties because I'm stupid. For context we have two models: the slow target $p$ and the fast draft $q$. One thing we want is to get lots of draft tokens accepted per draft. Another thing we want is to propose draft tokens really frequently. You shove state into the model, and you get out logprobs for each token. But kind of counterintuitively, we are going to say $p(x)$ is the probability we output that token

You have two options: Target Only or Rejection sampling. They are supposed to have the same behavior, but not necessarily the same efficiency. 

Target-Only: Greedily choose whatever token the draft model likes the most. Accept it with probability p(c). If you get it wrong, sample from p, excluding the greedy token.

What is the probability we emit a token $t$? Either we guess it right the first time, or we sample from the rejection thingy the second time. Concretely, if we draft token $c$, 

$$ P(\text{get t}) = p(c) 1[t = c] + (1 - p(c)) * p(t) / (1 - p(c)) * 1[t != c] $$

so if t = c then you get p(t), and if t != c then you also get p(t).

Rejection sampling: 

Sample a token $\hat{y} \sim q$, and accept it with probability $min(1, p(\hat{y})/q(\hat{y}))$. 

So if the target likes the first sampled token more than the draft, then it is definitely accepted. Otherwise, it is accepted proportionally to how much less the target likes it than the draft. 

So that means if it's rejected, our sampling needs to again weigh how much more the target likes each token than the drat. Concretely, this means we sample again from $p_{resid}(y) \propto max(0, p(y)-q(y))$. 

That's annoying because now we need to materialize logprobs for the entire draft and the entire target even in the happy path.

That's as far as I got before I had to do real work.


