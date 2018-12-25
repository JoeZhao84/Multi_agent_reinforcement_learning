[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/43851646-d899bf20-9b00-11e8-858c-29b5c2c94ccc.png "Crawler"
[image3]: case_episode_vs_score.png
[image4]: DDPG2.png
[image5]: MAAC.PNG



# Collaboration and Competition - Multi-Agent Reinforcement Learning

### By Deepak Trivedi, December 25, 2018

![Trained Agent][image1]
  
## Introduction

A multi-agent system is a group of autonomous, interacting entities sharing a common environment, which they perceive with sensors and upon which they act with actuators. Emergent behavior and complexity arise in a multi-agent system from agents co-evolving together. Adapting reinforcement learning to multi-agent systems is crucial to building artificially intelligent systems that can perform complex tasks that are outside the capability of single agents. Multi-Agent Reinforcement Learning (MARL) is therefore a rich area of ongoing research that has implications for game theory, evolutionary computation and optimization theory. Multi-agent systems can be used to address problems in a variety of domains, including robotics, distributed control, telecommunications, and economics [[1]](http://www.dcsc.tudelft.nl/~bdeschutter/pub/rep/10_003.pdf).


Traditional reinforcement learning approaches such as Q-Learning or policy gradient are poorly suited to multi-agent environments [[2]](https://papers.nips.cc/paper/7217-multi-agent-actor-critic-for-mixed-cooperative-competitive-environments.pdf). There is a major challenge with learning stability since each agent’s policy is changing as training progresses, the environment is non-stationary from the perspective of any individual agent.
 
One solution is a multi-agent policy gradient algorithm introduced by OpenAI, where agents learn a centralized critic based on the observations and actions of all agents. Although learning is centralized in this way, execution is decentralized. In this way, the algorithm allow agents to learn from their own actions as well as the actions of other agents in the environment [[3]](https://towardsdatascience.com/the-coopetition-dilemma-building-reinforcement-learning-agents-that-learn-to-collaborate-and-84f2b2acb186)


### Multi-Agent Deep Deterministic Policy Gradients (MADDPG)

First we explain the DDPG algorithm, on which MADDPG is based. is the Deep Deterministic Policy Gradients is an Actor-Critic based Reinforcement learning algorithm for continuous action [[4]](https://arxiv.org/abs/1509.02971). This is an adaptation of the ideas underlying the success of Deep Q-Learning to the continuous action domain.  


The following figure ([[5]](https://www.renom.jp/notebooks/tutorial/reinforcement_learning/DDPG/notebook.html)) illustrates the DDPG algorithm.
 
![DDPG algorithm][image4]  

The steps of the DDPG algorithm are listed below. ([[5]](https://www.renom.jp/notebooks/tutorial/reinforcement_learning/DDPG/notebook.html)): 

1. Two Neural Networks , Actor ***μ*** and Critic ***Q*** are initialized.
2. Two other neural networks target actor ***μ′*** and target critic ***Q′***
3. Get initial state ***s***
4. Get action ***a*** = ***μ***(***s***)
5. Take action ***a*** and get reward ***r*** with next state ***s′***
6. Get the value of present state, **value** ( ***s*** , ***a*** )= ***Q*** ( ***s*** , ***a*** )
7. Get target action ***a′*** = ***μ′*** ( ***s′*** ) and target critic value of next state ***Q′*** ( ***s′*** , ***a′*** )
8. Get value of current state from Bellman equation, **value** _ **target** (***s*** , ***a*** ) = ***r*** + ***Q′*** ( ***s′*** , ***a′*** )
9. Get Loss 1/m ∑( **value** _ **target** ( ***s′*** , ***a′*** ) − **value** ( ***s*** , ***a*** ))^2
10. Update Critic Network according to the loss
11. Get Gradient of ***Q*** ( ***s*** , ***a***) with respect to actor network ***μ*** ( ***s*** ), ∂ ***Q*** /∂ ***μ*** ( ***s*** )
12. Get Gradient of ***μ*** ( ***s***) with respect to weights of actor network ***θ*** ***μ***, ∂*** μ*** (***s*** )∂ ***θ*** _ ***μ***
13. Update Critic Network by maximizing the value of ∂ ***Q*** /∂ ***μ*** ( ***s*** ) ∂ ***μ*** ( ***s*** )/∂ ***θ_μ***
14. Update target critic using the equation ***Q′*** ← ***τQ*** + (1− ***τ*** ) ***Q′***
15. Update target actor using the equation ***μ′*** ← ***τμ*** + (1− ***τ*** ) ***μ′***


In the MADDPG model [[3]](https://towardsdatascience.com/the-coopetition-dilemma-building-reinforcement-learning-agents-that-learn-to-collaborate-and-84f2b2acb186), each agent is treated as an “actor” which gets advice from a “critic” about the what action to take. The critic is a model of the world that predicts the future reward of an action in a particular state, which is used by the agent — the actor — to update its policy. To make it feasible to train multiple agents that can act in a globally-coordinated way, MADDPG allow critics to access the observations and actions of all the agents. However, at the actual execution time,  the agents do not need to access the central critic and act solely based on their observations and ***their predictions of other agents' behaviors***. Since a centralized critic is learned independently for each agent, this approach can also be used to model different kinds of reward structures, such as zero-sum games, cooperative games, and mixed games. 

![MADDPG algorithm][image5]  

### Hyperparameters

A brief description of the hyperparameters is provided below. The results section lists out the various combinations of hyperparameters that were tried for building the model. 

- `seed`: Random seed used for evolving the game.
- `max_score`: The target average score. This was fixed at 30.   
- `BUFFER_SIZE`: Memory size for experience replay, a random sample of prior actions instead of the most recent action to proceed. 
- `BATCH_SIZE`: Batch size for optimization. Smaller batch size affected convergence stability. Larger batch size made the process very slow.   
- `TAU` : parameter for soft update of target parameters        
- `LR_ACTOR`, `LR_CRITIC`: Learning rate for the above algorithm, for the actor and the critic respectively. Large values are detrimental to getting a good final accuracy.          
- `WEIGHT_DECAY`: Decay rate per iteration of the weights used for learning. 
- `fc1_units`: Number of neurons in the first hidden layer of the neural network. 
- `fc2_units`: Number of neurons in the seocnd hidden layer of the neural network. 




## Tennis Environment
In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

![Trained Agent][image1]


### Installation Instructions

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

2. Place the file in the DRLND GitHub repository, in the `p3_collab-compet/` folder, and unzip (or decompress) the file. 


## Results

The completed code was run for a number of hyperparameters to train the model. The table below shows the various hyperparameters that were tested in different models. The parameter `seed` was added to hyperparameters to understand the repeatability of the training algorithm. The target score for considering the environment solved is +0.5 averaged over 100 episodes.  

It can be seen that for the same set of hyperparameters, the environment gets solved for one value of the random seed, but not for the other, implying that there are no guarantees of successfull solution for any of these set of hyperparameters - in this sense, the success of the algorithm will depend on *luck*.   

In this table, a Convergence value of 3000 indicates that the algorithm exhausted the maximum number of episodes, fixed at 3000, and yet did not achieve an average score of 0.5 averaged over 100 iterations.

The cases shown in boldface are cases that led to a successful solution of the environment. 

|Case| seed | BUFFER_SIZE | BATCH_SIZE | TAU  | LR_ACTOR | LR_CRITIC | WEIGHT_DECAY | fc1_units | fc2_units           | Convergence |
|---|---|--------------|-------------|-------|-----------|------------|---------------|------------|---------------------|------|
|**1**  | 1 | 100000 |  64 |  0.001 |  0.0002 |  0.0001 |  0 |  400 |  300 |  **1424** | 
|2  | 1 | 100000 |  64 |  0.001 |  0.0001 |  0.0001 |  0 |  400 |  300 |  3000 | 
|3  | 1 | 100000 |  128 |  0.001 |  0.0003 |  0.0001 |  0 |  400 |  300 |  3000 | 
|**4**  | 1 | 100000 |  128 |  0.001 |  0.0002 |  5e-05 |  0 |  450 |  300 |  **2637** | 
|5  | 1 | 100000 |  128 |  0.001 |  0.0001 |  3e-05 |  0 |  300 |  200 |  3000 | 
|6  | 1 | 100000 |  64 |  0.002 |  0.001 |  0.0001 |  0 |  300 |  200 |  3000 | 
|7  | 1 |  100000 |  64 |  0.002 |  0.001 |  0.0001 |  0 |  200 |  300 |  3000 | 
|8 | 1 |  100000 |  32 |  0.002 |  0.001 |  0.0001 |  0 |  200 |  300 |  3000 | 
|9  | 2 |  100000 |  64 |  0.001 |  0.0002 |  0.0001 |  0 |  400 |  300 |  3000 | 
|**10**  | 2 |  100000 |  64 |  0.001 |  0.0001 |  0.0001 |  0 |  400 |  300 |  **2899** | 
|**11**  | 2 |  100000 |  128 |  0.001 |  0.0003 |  0.0001 |  0 |  400 |  300 |  **2309** | 
|**12**  | 2 |  100000 |  128 |  0.001 |  0.0002 |  5e-05 |  0 |  450 |  300 |  **2335** | 
|13  | 2 |  100000 |  128 |  0.001 |  0.0001 |  3e-05 |  0 |  300 |  200 |  3000 | 
|14 | 2 |  100000 |  64 |  0.002 |  0.001 |  0.0001 |  0 |  300 |  200 |  3000 | 
|15 | 2 |  100000 |  64 |  0.002 |  0.001 |  0.0001 |  0 |  200 |  300 |  3000 | 
|16  | 2 |  100000 |  32 |  0.002 |  0.001 |  0.0001 |  0 |  200 |  300 |  3000 | 
|17  | 3 |  100000 |  64 |  0.001 |  0.0002 |  0.0001 |  0 |  400 |  300 |  3000 | 
|18 | 3 |  100000 |  64 |  0.001 |  0.0001 |  0.0001 |  0 |  400 |  300 |  3000 | 
|19 | 3 |  100000 |  128 |  0.001 |  0.0003 |  0.0001 |  0 |  400 |  300 |  3000 | 
|20 | 3 |  100000 |  128 |  0.001 |  0.0002 |  5e-05 |  0 |  450 |  300 |  3000 | 
|21 | 3 |  100000 |  128 |  0.001 |  0.0001 |  3e-05 |  0 |  300 |  200 |  3000 | 
|**22** | 3 |  100000 |  64 |  0.002 |  0.001 |  0.0001 |  0 |  300 |  200 |  **1086** | 
|23 | 3 |  100000 |  64 |  0.002 |  0.001 |  0.0001 |  0 |  200 |  300 |  3000 | 
|24  | 3 |  100000 |  32 |  0.002 |  0.001 |  0.0001 |  0 |  200 |  300 |  3000 | 

The figure below shows the average score (averaged over 100 episodes.) It can also be seen that scores do not monotonously increase. In fact, the score could start possibly stop dropping, and never recover. The sharp drop after reaching a value of +0.5 is an artifact of the filtering process, and is to be ignored.

A video of the solved environment has been uploaded to [YouTube.](https://www.youtube.com/watch?v=kxhpkH3aO0w)

Successful models are available in the `models` folder. 

 
![Results][image3]

 
## Future Work

Future work includes the following: 

1. Solving other environments: Some of the other environments available, such as the Soccer environment was natural extensions of the work presented here and solutions would be attempted as future work using MADDPG. 
2. Other Actor-Critic based methods will be implemented as future work with the Tennis environment. This will allow benchmarking of different algorithms. 
3. A more rigorous hyperparameter study will offer insights into the relative effect of different hyperparameters in learning for DDPG and other methods.
