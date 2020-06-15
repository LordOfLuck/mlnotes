# Probabilistic modeling of sequences

## Markov chain (transition model)

![https://bookdown.org/probability/beta/MC1.png](https://bookdown.org/probability/beta/MC1.png)
A first-order markov chain (MC) consists of a set of states $\mathcal{S}$ and a square proability transition matrix 
$$\mathbb{P_n} = \{p_{i, j}\}_{i, j=1}^{|\mathcal{S}|}\text{ such that }\\p_{i, j} = \Pr(S_n = s_i | S_{n-1}, \ldots, S_0) = \Pr(S_n = s_i | S_{n-1}=s_j),\\ \sum_jp_{i,j} = 1.$$ The conditional independence on states in earlier timesteps is the **first-order markovian independence condition**. Higher order MC has conditionally independent transitions given more previous states. But a higher-order MC(k) can be translated to a first-order MC(1) by merging each configuration of past $m$ states into a new state, but the size of the new MC will be exponential in the original size.  A MC can be represented as a directed graph.

The markov chain is **time-homogeneous** if the matrix $\mathbb{P}$ does not change with time. Let $\pi_0$ be a row vector of initial transition probabilities over all states and $\pi_n = \Pr(S_n)$ be the distribution over all states at time $n$. Then
$$\pi_n= \pi_0 \cdot \mathbb{P}^n$$ because we have
$$(\mathbb{P}^2)_{i, j} = \vec{p}_{i, \cdot} \vec{p}_{\cdot, j} = \sum_k p_{i, k}p_{k, j} = \Pr \left(\bigcup_k (i \rightarrow k, k\rightarrow j) \right)$$ so $(\mathbb{P}^2)_{i, j}$ tells us the probability of transitioning from $i$ to $j$ in two steps by calculating the probability of the union of all options. By induction, $(\mathbb{P}^n)_{i, j}$ is the sum of probabilities of all possible paths of length $n$ between states $i, j$.  If we multiply the matrix by $\pi_0$, we also include the state prior.

**Stationary distribution $\pi$** of a (time-homogeneous) MC satisfies $$\pi\mathbb{P} = \pi.$$  If the Markov chain is irreducible and aperiodic, then there is a unique stationary distribution and the following holds. $$\lim_{k \rightarrow \infty} \mathbb{P^k} = (\pi, \ldots, \pi)^T.$$ A Markov chain is said to be **irreducible** if it is possible to get to any state from any state. A state $i$ has **period**  $k$ if any return to state  $i$  must occur in multiples of  $k$ time steps. If $k=1$, the state is **aperiodic**.

### Training
We have a random variable $X$ attaining values in $\{1, \ldots k\}$ and we have sequences of data $\mathbf{x^1}, \dots, \mathbf{x}^k$. For first-order MC, we can estimate transition probabilities by maximum likelihood: $$p_{i, j} = \frac{\sum_m(\#_n(\mathbf{x}^m_n = i, \mathbf{x}^m_{n+1} = j))}{\sum_m(\#_n(\mathbf{x}^m_n = i))}$$ where $\#_n(.)$ is the count of true events inside the bracket. The sequences of random variables could be e.g. sequences of DNA symbols. 

### Evaluation
We can evaluate the probability of a sequence of states $\mathbf{x}$ by $$\Pr(\mathbf{x}) = \Pi_i p_{x_i, x_{i+1}}.$$ We can also answer questions such as *What is the probability that next 7 days are going to be sunny if it rained in last three days? What is the probability, that every second day it will rain?*

## Hidden markov models
Hidden markov model (HMM) consists of two stochastic processes  $\mathcal{S}$ and $\mathcal{O}$. 
The second process - **sensor model** - is observed and is dependent on the first process - **transition model** - that is not observed. The transition model is typically discrete and can be modeled with a markov chain. Formally, a homogeneous first order HMM can be described by a triplet  $\lambda=(\mathbb{A}, \mathbb{B}, \pi)$, where $\mathbb{A}= \{a_{ij}\}$ is a MC(1) transition matrix, row $i$ in $\mathbb{B}=\{b_{i, j}\}$ is a distribution over observation given hidden state $i$ and $\pi$ is an initial hidden state distribution **column** vector.  The Figure below contains a first-order linear hidden markov chain.
![](https://www.mdpi.com/jrfm/jrfm-12-00168/article_deploy/html/images/jrfm-12-00168-g001.png)The HMM has the advantage that it conditionally separates the observations from each other across time. It means that the observations are independent conditional on the hidden state $O_n \perp O_{:n-1} | S_n$ i.e. they are **d-separated** by $S_n$. Marginal distributions can be much easier to model. The observations can have complicated indirect dependencies. By introducing structure, we can disentangle the dependencies by directly modelling relations among observations and hidden states and direct interactions among hidden states alone. The hidden states themselves may not have any real world interpretation. 
See more on that in [this question](https://stats.stackexchange.com/questions/428902/what-is-the-benefit-of-latent-variables/432740#432740).

**Example** consider discrete observation $O$ attaining values in $\{1, \ldots n\}$ and probability at time $n$ depends on $k-1$ previous observations so $\Pr(O_n|O_{0:n-1}) = \Pr(O_n|O_{n-k:n-1})$. We need space exponential in $k$ to store such distribution $O(n^k)$ - exponentially many sequences of possible predecessors. On the other hand, if we introduce a latent variable $Z$ with $k$ possible states and assume $O_n \perp O_{:n-1} | Z_n$, we only store $O(nk)$ values for observables and $O(k^2)$ values for latents. So in the second case, we have fewer values to estimate which will give us lower variance estimates and will require fewer data to fit. However, we may lose some expressiveness depending on how entangled are the variables.

**A first-order sensor model** can be described as $$\Pr(O_n | S_{0:n}, O_{0:n-1}) = \Pr(O_n | S_n).$$ We are interested in the following problems.
1. Compute probability of observed seqence $O$ given a HMM $\lambda$
2. Calculate distribution of hidden state sequence $S$ given observed sequence $O$. All subproblems can be solved based on problem 1.
	- Filtering - $\Pr(S_n|O_{1:n})$
	- Prediction - $\Pr(S_{n+k}|O_{1:n})$
	- Smoothing - $\Pr(S_k|O_{1:n})$
	- Find a hidden state sequence $S$ that best explains observed sequence $O$
4. Train the model $\lambda$ to maximize likelihood of given data

### Problem 1.
We need to calculate the following efficiently. 
$$P(O|\lambda) = \sum_{\text{all sequences S}} P(O,S|\lambda) = \sum_{\text{all sequences S}} P(O|S, \lambda)P(S|\lambda)$$ This can be calculated with the **forward** algorithm (or by combining forward, backward).

**Forward algorithm:** $O(n|S|^2)$
Define $\alpha_n(i) = \Pr(O_{1:n}, S_n=i | \lambda)$ and $\alpha_n=(\alpha_n(1), \ldots, \alpha_n(|\mathcal{S}|))^T$. We can calculate dynamically as follows.
1. Set $\alpha_1 = \Pr(O_{1}=o_1, S_1 | \lambda) = \Pr(S_1 | \lambda) \Pr(O_{1}=o_1| S_1, \lambda)  = \pi \odot \mathbb{B}_{:, O_1}$ 
  where $\mathbb{B}_{i, O_1}$ is the probability of observing $O_1$ in state $i$.
2. Induction $\alpha_{k+1} = (\mathbb{A}^T\alpha_k) \odot \mathbb{B}_{:, O_{k+1}}$
3. $\Pr(O_{1:n}| \lambda) = \sum_{i=1}^{|S|} \alpha_n[i]$

### Problem 2.

**Backward algorithm:** $O(n|S|^2)$
Define $\beta_m(i) = \Pr(O_{m+1:n} | S_m=i, \lambda)$ and $\beta_m=(\beta_m(1), \ldots, \beta_m(|\mathcal{S}|))^T$. We can calculate dynamically as follows.
1. Set $\beta_n = \vec{1}^T$
2. Induction $\beta_{k} = \mathbb{A} \cdot( \mathbb{B}_{:, o_{k+1}} \odot \beta_{k+1})$

**Forward-backward (Smoothing)**:
$$\gamma_k = \Pr(S_k|O_{1:n}, \lambda) = \frac{\Pr(S_k, O_{1:n}|\lambda)}{\Pr(O|\lambda)} =\frac{\Pr(O_{k+1:n}|S_k, O_{1:k}, \lambda) \odot \Pr(S_k, O_{1:k}|\lambda)}{\sum_{S_k} \Pr(O, S_k| \lambda)}= \frac{\alpha_k \odot \beta_k}{\alpha_k \cdot \beta_k}$$

(Pozn.: Tahle verze se lisi od Bartakovy - neni potreba bayes rule ani normalizace na konci vypoctu. Vysledek by mel byt identicky. Akorat se na zacatku nepodminuje na pozorovanich ale na stavech)

**Filtering**: $$\Pr(S_n | O_{1:n}, \lambda)=\frac{\alpha_n}{\sum_i\alpha_n(i)}$$
**Prediction**: $$\Pr(S_{n+1} | O_{1:n}, \lambda) = \frac{\sum_i\Pr(S_{n+1}| S_n=i, \lambda)\Pr(O_{1:n}, S_n|\lambda)}{\Pr(O|\lambda)} = \frac{\mathbb{A}^T\alpha_n}{\sum_i\alpha_n(i)}$$

**Most likely state sequence**
Define $$\delta_k(j) = \max_{s_1, \ldots, s_{k-1}} \Pr(s_1, \ldots ,s_{k-1}, s_k=j)$$ and we have 
1. $\delta_{k+1}(j) = \mathbb{B}_{j, O_{k+1}}\max_i\delta_k(i)\mathbb{A_{i, j}}$. 
We keep track from where we came and traca back to extract the optimal hidden state sequence. TODO: add more detailed description

### Problem 3 (Baum-Welch).

$$\xi_{i j}(k)=\Pr\left(S_{k}=i, S_{k+1}=j | O, \lambda\right)=\frac{\Pr\left(S_{k}=i, S_{k+1}=j, O | \lambda\right)}{\Pr(O | \lambda)}=\frac{\alpha_{k}(i) a_{i j} \beta_{k+1}(j) b_{j, O_{k+1}}}{\sum_{l} \sum_m \alpha_{k}(l) a_{lm} \beta_{k+1}(m) b_{m, O_{k+1}}}$$ and we can estimate $$a^*_{ij} = \frac{\Pr(S_k=i, S_{k+1}=j | O)}{\Pr(S_k=i | O)} = \frac{\sum_{k=1}^T\xi_{ij}(k)}{\sum_{k=1}^T\gamma_k(i)}$$ so the transition matrix probabilities are estimated like relative frequencies of transitions $i \rightarrow j$ given all visits to $i$. TODO: write in matrix form, add equation for sensor model estimation and initial probas.

TODO: describe kalman filter and GMM 
