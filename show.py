import tkinter as tk
import numpy as np
from main import get_Q, step

Q = get_Q()
# GUI设置
cell_size = 50
agent_color = 'blue'
goal_color = 'green'
cliff_color = 'red'
path_color = 'gray'

# 创建窗口
root = tk.Tk()
root.title("Cliff Walking")

# 参数设置
height = 4
width = 12
start_state = (height - 1, 0)
goal_state = (height - 1, width - 1)
cliff_states = [(height - 1, i) for i in range(1, width - 1)]
# 行动空间
actions = ['up', 'right', 'down', 'left']
action_indices = {action: i for i, action in enumerate(actions)}
# 创建画布
canvas = tk.Canvas(root, width=width*cell_size, height=height*cell_size)
canvas.pack()




# 绘制网格和断崖
for i in range(height):
    for j in range(width):
        canvas.create_rectangle(j*cell_size, i*cell_size, (j+1)*cell_size, (i+1)*cell_size, fill='white')
        if (i, j) in cliff_states:
            canvas.create_rectangle(j*cell_size, i*cell_size, (j+1)*cell_size, (i+1)*cell_size, fill=cliff_color)
        if (i, j) == goal_state:
            canvas.create_rectangle(j*cell_size, i*cell_size, (j+1)*cell_size, (i+1)*cell_size, fill=goal_color)

# 画出起点和终点
canvas.create_text(start_state[1]*cell_size + cell_size/2, start_state[0]*cell_size + cell_size/2, text='Start', fill='black')
canvas.create_text(goal_state[1]*cell_size + cell_size/2, goal_state[0]*cell_size + cell_size/2, text='Goal', fill='black')

# 移动智能体并显示路径
def move_agent(policy):
    state = start_state
    agent = canvas.create_oval(state[1]*cell_size, state[0]*cell_size, (state[1]+1)*cell_size, (state[0]+1)*cell_size, fill=agent_color)

    while state != goal_state:
        action = policy[state]
        next_state = step(state, action)
        canvas.move(agent, (next_state[1] - state[1])*cell_size, (next_state[0] - state[0])*cell_size)
        canvas.create_rectangle(state[1]*cell_size, state[0]*cell_size, (state[1]+1)*cell_size, (state[0]+1)*cell_size, fill=path_color)
        state = next_state
        root.update()
        canvas.after(500)  # 等待500毫秒

    canvas.create_oval(state[1]*cell_size, state[0]*cell_size, (state[1]+1)*cell_size, (state[0]+1)*cell_size, fill=goal_color)

# 转换策略格式，以便于使用
policy_map = {(i, j): actions[np.argmax(Q[i, j])] for i in range(height) for j in range(width) if (i, j) not in cliff_states and (i, j) != goal_state}

# 开始移动智能体
move_agent(policy_map)

# 运行Tkinter事件循环
root.mainloop()
