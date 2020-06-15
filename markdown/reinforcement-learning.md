# Reinforcement learning
Based on [Reinforcement learning: An Introduction](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf) and slides from Roman Barták. The notation has been adjusted to comply with the slides.

**Supervised learning** agent is explicitly told what to do. Receives input and required output.
**Reinforcement learning** agent only receives a numeric **reward** for his actions without being told what actions to take. The agent interacts with its environment, moves from state to state and based on itsa actions receives rewards from the environment.
<div style="display: flex; justify-content: center;">
<img src="https://miro.medium.com/max/1000/0*2KA8W5SAb0Zc3JwA.png" width="600"/>
</div>

Formally a reinforcement agent is placed in a **markov decision process**(MDP).
A MDP consists of the follwing:
1. A set of states $\mathcal{S}$
2. A set of available actions in each state $\mathcal{A}_s$ (all states can have identical action set)
3. A transition function $P_a(s, s') = Pr(s_{t+1} = s' | s_t = s, a_t = a)$
4. A reward function $R_a{s, s'}$ - immediate reward for doing action a in state $s$ and moving to state $s'$

The transition probabilities between states are independent of previously visited states (markov property)
Reward function can depend only on state (Bartákovy slidy, můžeme přechazet mezi definicemi $R(s) = \mathbb{E}_a\mathbb{E}_{s'}[R_a{s, s'}]$, takže immediate reward stavu je průměrný immediate reward stavu. Rozdíl je také v tom, že případ $R(s)$ počítá s tím, že reward je obdržen při příchodu do stavu s, zatímco $R_a(s, s)$ je přidělen až během přechodu do dalšího stavu a dal by se tak počítat jako reward v dalším časovém kroku, což má vliv na formulaci následujících rovnic)

The agent can have a
1. deterministic policy $\pi : \mathcal{S}  \rightarrow \mathcal{A_s}$
2. stochastic policy $A_s \sim \pi (.|s)$, where $\pi(a|s) = Pr(a|s)$ is probability of taking action $a$ in state $s$.

<div style="display: flex; justify-content: center;">  
<img src="https://upload.wikimedia.org/wikipedia/commons/2/21/Markov_Decision_Process_example.png" width="600"/>  
</div>

The agent tries to maximize his **utility** - the expected discounted return (present value of future rewards).

$$U_{\pi}(s)  = \mathbb{E}_{\pi}[\sum_{k=0}^\infty \gamma^k R_{t+k},  |S_t = s], 0\leq \gamma \leq 1$$

The utility function (value function in other literature) can be rewriten recursively according to the bellman's equation.

$$U_{\pi}(s) = R(s) + \gamma \sum_a\pi(a|s)\sum_{s'} P_a(s, s') U_{\pi}(s')$$

If we know the transition probabilities and rewards, it is possible to calculate $U$ for the given policy
1. **dynamically** by backwards induction (at least if there is a terminal state - with infinite process I am not very sure, but it should be possible - it is a system of linear equations).
2. Iteratively by **policy evaluation**. 
	 - Input $\pi,$ the policy to be evaluated 
	 - Initialize an array $U(s)=0,$ for all $s \in \mathcal{S}^{+}$ 
	 - Repeat
		 * $\Delta \leftarrow 0$
		 * For each $s \in \mathcal{S}$
			 + $u \leftarrow U(s)$
			 + $U(s) \leftarrow R(s) + \gamma \sum_a\pi(a|s)\sum_{s'} P_a(s, s') U_{\pi}(s')$
			+ $\Delta \leftarrow \max (\Delta,|u-U(s)|)$
	 - until $\Delta<\theta$ (a small positive number) Output $U \approx u_{\pi}$
3. With **temporal difference updates**
	 - Input $\pi,$ the policy to be evaluated, $\alpha$
	 - Initialize an array $U(s)=0,$ for all $s \in \mathcal{S}^{+}$ 
	 - Repeat
		 - init $s$
		 * repeat
			  + $a \leftarrow \pi(a|s)$, observe $R, S'$
			 + $U(s) \leftarrow U(s) + \alpha (R + \gamma U(s')  - U_{\pi}(s))$
			+ $s \leftarrow s'$
		 - until $s$ terminal
	 -  until not happy

The optimal policy can be formulated as $\pi^*=\argmax_{\pi}(U_{\pi}(s))$ for all $s\in \mathcal{S}$ so an optimal policy must be at least as good as other policies in all states. The optimal policy may not be unique, but all optimal policies share the same optimal utility function $U^*(s)=\max_{\pi}(U_{\pi}(s))$ for all $s$.

We can divide agents into the following groups:
1. **Utility-based** agent learns the values of $U_{\pi}$ and selects a policy that is an improvement over the current one. 
2. **Q-learning agent** - learns action-value function - expected utility of taking a given action in a given state (this update is only for evaluation of policy!)
	$$Q_{\pi}(a, s)=R(s) + \gamma \sum_{s'} P_a(s, s') U_{\pi}(s') = R(s) + \gamma (\sum_{s'} P_a(s, s')[R(s') + \gamma \sum_a' \pi(a'|s') Q_{\pi}(a', s')])$$
3. **Reflex agent** learns a policy that maps directly from states to actions

We can divide types of learning: 

1.  **Passive** - learn values $U$ for fixed policy $\pi$
	* **Direct utility estimation** - collect experience by applying $\pi$, keep running average return $\hat{U}(s)$ for each state. Supervised learning with (state, utility) pairs . Inefficient - it assumes independence between states, but we know the bellman equation which could help us shrink the hypothesis space.
	* **Adaptive dynamic programming**(ADP) - Instead of $U$, learn the transition probabilities (as frequencies) and rewards. Eg. repeatedly iterate over all states and in each state sample a reward and a random action and observe result state. Calculate $U_{\pi}$ dynamically by backwards induction based on the estimates and chosen policy $\pi$ or use  *policy evaluation*. 
	* **temporal difference** - Estimate $U_{\pi}$ directly, but enforce consistency according to bellman equation - using the temporal difference update (smarter than direct utility estimation)
2.  **Active** - learn how to act optimally in an environment - must iteratively estimate the value function for current policy and update his policy to improve his expected gains. It involves **exploration**.
	- policy iteration, SARSA, Q-learning 

**Policy iteration**  - Iteratively evaluate $U_{\pi}$ and improve $\pi$  by selecting greedy actions w.r.t. current estimate of $U$. Thanks to *policy improvement theorem* this algorithm is guaranteed to converge, but  we **need accurate estimate of the environment transitions**, which can easily be intractable. 
- def PolicyImprovement(...):
	-  for $s$ in $\mathcal{S}$:
		- $a \leftarrow \pi(s)$
		- $\pi(s) = \argmax_{\mathcal{A}_s} \sum_{s'}P_a(s,s') U_{\pi}(s')$
	- return $\pi$
- Use  ADP to estimate transition probabilities $P$ and the reward
- Select initial $\pi$
- while True:
	-   $U_\pi \leftarrow PolicyEvaluation(\mathcal{S}, \mathcal{A}, R, P,  \pi)$
	-  $\pi \leftarrow PolicyImprovement(\mathcal{S}, \mathcal{A}, R, P,  \pi, U_{\pi})$
	-  if $\pi$ has not changed from last iteration, we have found optimal policy; break
<div style="display: flex; justify-content: center;">
<img src="https://programmingbeenet.files.wordpress.com/2019/03/policy_iteration.png" width="600"/>
</div>

The following do not need to know the environment dynamics.

 **SARSA**
- for each epoch:
	- init $s$ 
	- $a \leftarrow EpsilonGreedy(Q, \epsilon)$
	- repeat:
		- take action $a$, observe $R, s'$
		- - $a' \leftarrow EpsilonGreedy(Q, \epsilon)$
		- $Q(a, s) \leftarrow Q(a, s) + \alpha(R(s) + \gamma Q(s', a') - Q(a, s)])$
		- $s \leftarrow s'$ 
	- until $s$ terminal
The update could be rewritten: $Q(a, s) \leftarrow (1-\alpha)Q(a, s) + \alpha(R(s) + \gamma Q(s', a')])$ which is like a linear combination of current estimate and an $\epsilon-greedy$ sample of estimate of future returns.
The algorithm is **on-policy** meaning that the agent behaves according to some policy (eg. $\epsilon$-greedy) and updates its estimates to improve the same policy.

**Q-learning** - explore to search  new states - with small proba $\epsilon$ , select any action with equal proba. Otherwise exploit (greedy (best) action in given state according to current estimate of $Q$ in state $s$)
- for each epoch:
	- init $s$ 
	- repeat:
		- $a \leftarrow EpsilonGreedy(Q, \epsilon)$, observe $R, s'$
		- $Q(a, s) \leftarrow Q(a, s) + \alpha(R(s) + \gamma \max_{a} Q(s', a) - Q(a, s)])$
		- $s \leftarrow s'$ 
	- until $s$ terminal
The update rule equality holds if the $Q$ values are optimal (bellman)
The algorithm is **off-policy** - the agent behaves according to $\epsilon$-greedy policy, but makes updates to improve fully greedy policy (hard max in the update) so it improves different policy than the one that was used to collect the experience.
Q-learning can learn an optimal policy even when a random policy is followed during training - more data efficient, easy to reuse experience, but may converge slower.

**Exploration**
Choosing optimal actions(exploitation) w.r.t. the current estimate of the environment may not be optimal w.r.t the true environment. Exploration means trying immediately suboptimal actions in order to discover states that are potentially more beneficial in the long-term. Exploration improves the model. The options are:
1. $\epsilon$-greedy - be greedy with proba $(1-\epsilon)$, else explore uniformly
2. Give large weight to actions with large variance estimate and avoid low variance low utility actions(actions that we know are bad)
	- eg. initialize $U$ high for all states - optimistically biased (curious) agent. The agent will be mostly disappointed , but will have a motivation to discover.
