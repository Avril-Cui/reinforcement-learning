# Value Iteration
Known transition function and reward, need to find the best policy (guiding which action to take) \
Steps:
1. Initialize value table randomly
2. While Loop, continue the loop as long as update step is greater than a threshold (epsilon)
3. In each loop, loop through the states
    - For each state, loop through the actions
    - Update the value of a state as the maximum of (reward of state+gamma*sum(transition probability * value of next state given an action))
\


# Transition
transition = {
    's0': {'a0': {'s0': 0.1, 's1': 0.9}},
    's1': {'a0': {'s1': 0.1, 's2': 0.9}, 'a1': {'s0': 0.9, 's1': 0.1}},
}