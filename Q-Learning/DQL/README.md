# Learning Setting: [Open-AI Gymnasium]([https://gymnasium.farama.org/) CartPole
- Rewards
    - +1 for every incremental timestep
    - Env terminate if pole falls over or cart moves more than 2.4 units away from center
- Performance (value)
    - Higher value if scenarios run for longer duration, accumulating larger return
- Input State
    - (position, velocity, etc.)
- Action
    - (left, right)

# DQL Core Concepts
From paper "Playing Atari with Deep Reinforcement Learning"
## Q-Learning With Deep Neural Network
Addresses the issues of:
1. A delay between actions and resulting reward
    - Contrary to direct association between inputs and targets in supervised learning
2. Highly correlated states in sequence
3. Change in data distribution

To address these issues, a few new mechanisms are introduced in DQL.

## Experience Replay (AKA Replay Memory)
During each step of an episode, an agent generates a tuple of experience, often represented as $(\text{state}, \text{action}, \text{next\_state}, \text{reward})$. Replay memory stores these experiences in a buffer, allowing the agent to access and learn from past actions, even if they no longer represent the current state. Experience replay enables the model to randomly sample from previous transitions, which smooths the training distribution over many past behaviors, tackling [Issue 3](#q-learning-with-deep-neural-network).

Since batches of training data are randomly sampled from memory pool, each step of experience can potentially be used in many weight updates, ensuring greater data efficiency. Also, randomized sample breaks correlations between states, reducing the variance of the updates, tackling [Issue 2](#q-learning-with-deep-neural-network).

## Deep Neural Network
There are two neural networks in the model: [policy network](#policy-net) and [target network](#target-net)

The goal of the neural networks is to minimize:
$$\delta = Q(s,a)-r*\gamma*\max_{a'}Q(s',a')$$
Use Huber loss, which acts like the mean squared error when the error is small, but like the mean absolute error when the error is large; more robust to outliers
$$L(\delta) = \begin{cases}
\frac{1}{2}*\delta^2 & \text{ for } |\delta|\leq 1 \\
|\delta|-\frac{1}{2} & \text{ otherwise }
\end{cases}$$
With deep neural networks, there are no need to store each Q-values in a table indexed by $s\in S$ and $a\in A$ anymore. Instead, they are stored as functions (input & output mapping) in the neural networks.

## Policy Net
During training, the policy_net is updated after each step or batch to minimize the difference between the Q-values predicted by the network and the target Q-values. The role of policy_net is to output expected return of current state.

## Target Net
The role of target_net is to output expected return of future states. The key idea is that by keeping this network slightly “behind” the policy_net, it reduces the risk of instability in the training process. If only had a single network, the constantly changing Q-values would lead to a target that is keep shifting, resulting in feedback loops, destabilizing the learning process. target_net ensures that predicted Target Q values remain stable for a short period.
- Soft Update: Use a small factor, TAU, to slowly blend the weights from policy_net into target_net, making incremental updates.

## Gradient Clipping
Gradient Clipping is a technique used to prevent the gradients from becoming too large, which can destabilize the training process, especially in reinforcement learning
- torch.nn.utils.clip_grad_value_ clips the gradients in-place for all parameters in policy_net to ensure they don’t exceed a specified value

# Useful Syntax
- namedtuple: access elements in tuple with "keys"; for example, transition.state=0
- deque: efficient append and pop, fixed-size buffer (useful for replay memory in RL, where older experiences are dropped as new ones are added, keeping a manageable memory size)
- random.sample(array, n): randomly sample an array of length n from array
- mask: A tensor of Boolean values (True or False) where each position indicates whether the corresponding element in another tensor should be selected or ignored
- torch.cat() concatenates the list of tensors into a single tensor along the first dimension (batch dimension)