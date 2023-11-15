import numpy as np
import tkinter as tk
import numpy as np
# 环境参数
height = 4
width = 12
start_state = (height - 1, 0)
goal_state = (height - 1, width - 1)
cliff_states = [(height - 1, i) for i in range(1, width - 1)]

# 超参数
alpha = 0.5 # 学习率
gamma = 0.9 # 折扣因子
epsilon = 0.1 # 探索率
num_episodes = 500 # 总的游戏回合数

# 行动空间
actions = ['up', 'right', 'down', 'left']
action_indices = {action: i for i, action in enumerate(actions)}

# 初始化Q表
Q = np.zeros((height, width, len(actions)))

def choose_action(state, Q, epsilon):
    if np.random.uniform(0, 1) < epsilon:
        return np.random.choice(len(actions))  # 探索
    else:
        return np.argmax(Q[state])  # 利用

def step(state, action):
    i, j = state
    if action == 'up':
        return max(i - 1, 0), j
    elif action == 'right':
        return i, min(j + 1, width - 1)
    elif action == 'down':
        return min(i + 1, height - 1), j
    elif action == 'left':
        return i, max(j - 1, 0)

def update_Q(Q, state, action_index, reward, next_state, alpha, gamma):
    best_next_action = np.argmax(Q[next_state])
    Q[state][action_index] = (1 - alpha) * Q[state][action_index] + alpha * (reward + gamma * Q[next_state][best_next_action])

# 训练Q表
for episode in range(num_episodes):
    state = start_state
    while state != goal_state:
        action_index = choose_action(state, Q, epsilon)
        next_state = step(state, actions[action_index])
        reward = -1 if next_state not in cliff_states else -100
        if next_state in cliff_states:
            next_state = start_state  # 掉入断崖，重置到起始位置
        update_Q(Q, state, action_index, reward, next_state, alpha, gamma)
        state = next_state

# 输出最佳策略
policy = np.array([[' ' for _ in range(width)] for _ in range(height)])
for i in range(height):
    for j in range(width):
        if (i, j) == start_state:
            policy[i, j] = 'S'
        elif (i, j) == goal_state:
            policy[i, j] = 'G'
        elif (i, j) in cliff_states:
            policy[i, j] = 'C'
        else:
            best_action = actions[np.argmax(Q[i, j])]
            policy[i, j] = best_action[0].upper()

print("最佳策略:")
print(policy)


