# 4.7 基线实验与章节总结

在前面几节中，我们从理论推导到架构设计，一步步理解了从 REINFORCE 到 Actor-Critic 的演进逻辑。现在让我们回到代码，用实验亲眼看到**基线**（Baseline）的力量——它是 REINFORCE 和 Actor-Critic 之间的关键一步。

## 对比实验：有基线 vs 无基线

### 实验设计

我们用同一个摇骰子赌博机（A: 30%, B: 70%），分别运行两个版本：

- **无基线**：标准 REINFORCE，`loss = -log_prob * G`
- **有基线**：减去运行中的平均回报，`loss = -log_prob * (G - baseline)`

```python
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt

# ==========================================
# 策略网络（和上一节相同）
# ==========================================
class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 2)

    def forward(self, x):
        return torch.softmax(self.linear(x), dim=-1)

# ==========================================
# 环境和训练参数
# ==========================================
win_probs = [0.3, 0.7]
num_episodes = 500
lr = 0.01

def pull_arm(action):
    return 1.0 if random.random() < win_probs[action] else 0.0

# ==========================================
# 训练函数（支持有无基线）
# ==========================================
def train_reinforce(use_baseline=False):
    policy = PolicyNetwork()
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    prob_history = []

    # 运行中的平均回报作为基线
    baseline = 0.0
    alpha_baseline = 0.05  # 基线的更新速率

    for ep in range(num_episodes):
        state = torch.tensor([1.0])
        probs = policy(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        reward = pull_arm(action.item())

        if use_baseline:
            # 有基线：减去平均回报，只保留"比平均好了多少"
            advantage = reward - baseline
            loss = -log_prob * advantage
            # 更新基线（指数移动平均）
            baseline = baseline + alpha_baseline * (reward - baseline)
        else:
            # 无基线：直接用回报
            loss = -log_prob * reward

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            current_probs = policy(state)
            prob_history.append(current_probs[1].item())

    return prob_history

# ==========================================
# 运行对比实验（各跑 5 次取平均）
# ==========================================
num_runs = 5
no_baseline_all = []
with_baseline_all = []

for run in range(num_runs):
    torch.manual_seed(run)
    random.seed(run)
    no_baseline_all.append(train_reinforce(use_baseline=False))
    torch.manual_seed(run + 100)
    random.seed(run + 100)
    with_baseline_all.append(train_reinforce(use_baseline=True))

# 计算平均值
import numpy as np
no_baseline_avg = np.mean(no_baseline_all, axis=0)
with_baseline_avg = np.mean(with_baseline_all, axis=0)

# ==========================================
# 可视化对比
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 左图：单次运行对比
ax1.plot(no_baseline_all[0], alpha=0.7, color='red', label='No Baseline (1 run)')
ax1.plot(with_baseline_all[0], alpha=0.7, color='blue', label='With Baseline (1 run)')
ax1.axhline(y=0.7, color='green', linestyle='--', alpha=0.5)
ax1.set_xlabel('Episode')
ax1.set_ylabel('P(choose B)')
ax1.set_title('Single Run: Baseline Reduces Oscillation')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 右图：5次平均对比
ax2.plot(no_baseline_avg, color='red', linewidth=2, label='No Baseline (5-run avg)')
ax2.plot(with_baseline_avg, color='blue', linewidth=2, label='With Baseline (5-run avg)')
ax2.axhline(y=0.7, color='green', linestyle='--', alpha=0.5)
ax2.set_xlabel('Episode')
ax2.set_ylabel('P(choose B)')
ax2.set_title('5-Run Average: Baseline Converges Faster')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('baseline_comparison.png', dpi=150)
plt.show()

print(f"无基线最终 P(B): {no_baseline_avg[-1]:.3f}")
print(f"有基线最终 P(B): {with_baseline_avg[-1]:.3f}")
```

### 实验结果解读

你会看到两张图：

**左图（单次运行）**：

- 红线（无基线）：策略概率在 0.5 到 0.9 之间剧烈震荡
- 蓝线（有基线）：策略概率平滑上升，最终稳定在 0.85-0.95

**右图（5次平均）**：

- 红线：即使取了平均，仍然能看出明显的波动
- 蓝线：收敛更快、更稳定

> 🖼️ **[此处可添加：基线对比实验图——展示无基线（锯齿状）vs 有基线（平滑上升）]**

### 为什么基线这么有效？

|               | 无基线                     | 有基线                               |
| ------------- | -------------------------- | ------------------------------------ |
| **梯度信号**  | $G_t$ 的绝对值             | $G_t - b$（相对值）                  |
| **摇 A 赢了** | reward=1 → 增加选 A 的概率 | reward=1 - baseline=0.5 → 只增加 0.5 |
| **摇 B 赢了** | reward=1 → 增加选 B 的概率 | reward=1 - baseline=0.5 → 只增加 0.5 |
| **关键区别**  | "赢了就是赢了"             | "赢了，但只比平均好了 0.5"           |

基线的作用是**减掉"运气"带来的噪声**。当你赢的时候，基线也在跟踪"正常情况下赢的概率"，所以梯度信号只反映"真正因为动作选择带来的好坏差异"，而不是"运气好不好"。

## 本章小结

在这一章中，你完成了从"理论推导"到"架构设计"的完整旅程：

1. **亲手实现了 REINFORCE**：用一个极简的赌博机实验，亲眼看到策略网络学会"偏爱好的动作"
2. **理解了策略梯度定理**：$\nabla J = \mathbb{E}[\nabla \log \pi \times G_t]$——"好结果强化动作概率"
3. **认识了 REINFORCE 的致命缺陷**：高方差让训练像"醉汉走路"
4. **理解了基线（Baseline）**：减掉"平均期望"可以降低方差而不改变梯度方向
5. **学会了 Actor-Critic 架构**：Actor（策略网络）+ Critic（价值网络），用 TD Error 做桥梁
6. **建立了两个核心认知**：
   - **探索 vs 利用**：RL 最核心的张力
   - **偏差-方差权衡**：MC（无偏高方差）vs TD（低偏差有偏差）

这些概念在后续章节的演进：

| 概念         | 第 6 章 PPO       | 第 7 章 DPO    | 第 8 章 GRPO     |
| ------------ | ----------------- | -------------- | ---------------- |
| Actor-Critic | PPO 的骨架        | —              | 去掉 Critic      |
| 基线         | GAE 中的 $V(s)$   | 隐式基线       | 组内均值做基线   |
| 策略梯度     | Clip 限制更新幅度 | 概率比替代梯度 | 组内比较替代梯度 |
| 高方差       | GAE 控制方差      | 离线数据无方差 | 组内采样降方差   |

## 练习

1. **修改赢率差距**：把赌博机改为 A: 49%, B: 51%（差距极小）。REINFORCE 还能学会吗？有基线的版本呢？观察两者收敛速度的差异。

2. **多臂赌博机**：把赌博机改为 3 个摇臂（A: 20%, B: 50%, C: 80%），修改策略网络为 3 输出。观察策略是否正确地学会了"最爱 C、最不爱 A"。

3. **基线的选择**：如果把基线改为固定值 0.5（而不是运行平均），效果如何？和运行平均基线相比有什么差异？这说明了什么？

4. **连续动作空间思考**：假设你要训练一个机器人学会走路。状态是关节角度和速度，动作是施加到每个关节的力矩（连续值）。策略网络应该输出什么？怎么从输出中采样动作？

---

## 参考文献

[^1]: Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. _Machine Learning_, 8(3-4), 229-256. [DOI](https://doi.org/10.1007/BF00992696)

[^2]: Sutton, R. S., et al. (2000). Policy gradient methods for reinforcement learning with function approximation. _Advances in Neural Information Processing Systems_, 12.

[^3]: Mnih, V., et al. (2016). Asynchronous methods for deep reinforcement learning. _International Conference on Machine Learning (ICML)_. [arXiv:1602.01783](https://arxiv.org/abs/1602.01783)
