# 4.1 动手：摇骰子赌博机

在上一章的末尾，我们站在高处俯瞰了 RL 的全景地图，看到了两条路线——Value-Based 和 Policy-Based。现在，让我们正式踏上 **Policy-Based** 路线，亲手写代码让一个策略网络从"什么都不知道"进化到"学会选最优动作"。

我们的实验场是一个极简的赌博机——只有两个摇臂，一个赢率高、一个赢率低。策略网络需要自己摸索出"应该偏爱哪个摇臂"。

## 环境：两臂赌博机

规则非常简单：

- 两个动作：**摇 A**（赢率 30%，赢了 +1 分）和**摇 B**（赢率 70%，赢了 +1 分）
- 每次只能选一个摇，选完立刻知道结果
- 没有任何"状态"——这是一个**无状态**的赌博机（即只有一个状态 $s$）

你可能会想："B 的赢率 70%，当然应该永远摇 B 啊！"没错。但我们的策略网络一开始什么都不知道——它需要**通过试错自己发现**这个事实。

这和第 3 章的猜硬币游戏有本质区别：猜硬币中我们手写了策略（"永远猜正"），而这里我们要**让 AI 自己学出策略**。

## 用 PyTorch 实现策略网络

```python
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt

# ==========================================
# 1. 定义策略网络：只有一个 Softmax 层
# ==========================================
class PolicyNetwork(nn.Module):
    """最简单的策略网络：输入维度=1，输出2个动作的概率"""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 2)  # 1个输入（常数1），2个输出（A和B的概率）

    def forward(self, x):
        logits = self.linear(x)
        return torch.softmax(logits, dim=-1)  # Softmax 保证输出是概率分布

policy = PolicyNetwork()
optimizer = optim.Adam(policy.parameters(), lr=0.01)

# ==========================================
# 2. 环境：两臂赌博机
# ==========================================
win_probs = [0.3, 0.7]  # A: 30%, B: 70%

def pull_arm(action):
    """摇动指定臂，返回奖励"""
    return 1.0 if random.random() < win_probs[action] else 0.0

# ==========================================
# 3. REINFORCE 训练（先看效果，原理下一节详解）
# ==========================================
prob_history = []  # 记录每轮选择 B 的概率
num_episodes = 300

for ep in range(num_episodes):
    state = torch.tensor([1.0])  # 常数输入（无状态环境）

    # --- 采样阶段：策略网络输出动作概率，按概率采样 ---
    probs = policy(state)
    action_dist = torch.distributions.Categorical(probs)
    action = action_dist.sample()  # 按概率随机选一个动作
    log_prob = action_dist.log_prob(action)  # log π(a|s)

    # --- 执行动作，获取奖励 ---
    reward = pull_arm(action.item())

    # --- 计算损失并更新 ---
    # REINFORCE 的核心：loss = -log_prob × reward
    # 直觉："如果这个动作带来了好结果（reward > 0），就增加它的概率"
    loss = -log_prob * reward

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 记录策略概率
    with torch.no_grad():
        current_probs = policy(state)
        prob_history.append(current_probs[1].item())  # 选择 B 的概率

# ==========================================
# 4. 可视化策略演化过程
# ==========================================
print(f"初始策略概率: B = {prob_history[0]:.3f}, A = {1-prob_history[0]:.3f}")
print(f"最终策略概率: B = {prob_history[-1]:.3f}, A = {1-prob_history[-1]:.3f}")

plt.figure(figsize=(10, 4))
plt.plot(prob_history, alpha=0.6, color='orange')
plt.xlabel('Episode')
plt.ylabel('Probability of choosing B')
plt.title('Policy Evolution: How the network learns to prefer B (70% win rate)')
plt.axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='Win rate of B')
plt.axhline(y=0.3, color='red', linestyle='--', alpha=0.5, label='Win rate of A')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('dice_policy_evolution.png', dpi=150)
plt.show()
```

## 你会看到什么？

运行这段代码，你会看到策略概率的演化过程：

1. **开局（Episode 0-50）**：策略概率在 0.5 附近波动——网络"两个都想试试"，相当于在**探索**。
2. **爬升（Episode 50-200）**：选择 B 的概率逐渐上升——网络发现"摇 B 拿到奖励的次数更多"，开始**偏爱 B**。
3. **收敛（Episode 200+）**：概率稳定在 0.85-0.95 之间——网络"学会了"应该主要选 B。

但你也注意到一个现象：**曲线不是平滑上升的，而是有明显的锯齿和波动**。这就是策略梯度的核心痛点——**高方差**——同样的策略，每次采样到的数据不同，导致梯度估计忽大忽小。

> 🖼️ **[此处可添加：策略概率演化曲线图——展示从0.5逐步上升到0.9的锯齿状曲线]**

## 4.2 观察训练曲线：为什么策略会"震荡"？

### 震荡的本质：梯度噪声

策略梯度的更新依赖于**采样**。每次更新时，网络只能看到**这一次**采样到的结果：

- 如果某次恰好摇 B 也输了（30% 概率），网络会收到"reward = 0"的信号，**降低**选 B 的概率
- 如果某次恰好摇 A 也赢了（30% 概率），网络会收到"reward = 1"的信号，**增加**选 A 的概率

这些"运气不好"的采样会导致梯度估计出错，让策略在正确方向上来回摇摆。这就是**高方差**的直观体现。

### 学习率的影响

<details>
<summary><strong>如果把学习率从 0.01 改成 0.1，会发生什么？</strong></summary>

策略会在 A 和 B 之间剧烈摇摆——某次采样到 A 赢了就大幅偏向 A，下次采样到 B 赢了又大幅偏向 B。永远无法稳定收敛。

这就像开车时方向盘打得太猛——每次修正都过了头，反而在目标两侧来回摆动。**学习率太大 = 每步修正太过头 = 高方差被放大。**

试试把代码里的 `lr=0.01` 改成 `lr=0.1`，你会亲眼看到策略的剧烈震荡。

</details>

### 核心概念：探索与利用的权衡

训练曲线的震荡还揭示了 RL 最核心的张力——**探索（Exploration）与利用（Exploitation）**：

|          | 探索                       | 利用               |
| -------- | -------------------------- | ------------------ |
| **含义** | 试试新动作，看会不会更好   | 坚持已知的好动作   |
| **优点** | 可能发现更优策略           | 稳定获取高分       |
| **缺点** | 浪费样本在已知不好的动作上 | 可能错过更好的选择 |

- **训练初期**：策略接近均匀分布（探索为主）——"两个都试试看"
- **训练后期**：策略收敛到确定动作（利用为主）——"就选 B 了"
- **过渡必须平稳**：太快收敛可能只找到局部最优（比如 B 赢率高但没试过就放弃了），太慢则浪费数据

> 第 6 章 PPO 中的 **entropy bonus**（熵奖励）就是强制策略保持探索的机制——它在损失函数里加了一个惩罚项，防止策略过早变得"太确定"。

<details>
<summary><strong>思考题：如果 B 的赢率只有 55%（而不是 70%），策略还能学会吗？</strong></summary>

能学会，但**学得更慢、震荡更大**。因为 A 和 B 的差距更小（55% vs 30%），采样到"误导性数据"的概率更高。这再次说明了策略梯度的核心挑战——**当"好"和"坏"的差距不大时，高方差会让学习变得极其困难**。

这也是为什么后续我们需要引入基线（Baseline）来降低方差——下一节会详细展开。

</details>

**你将获得**：

- 亲眼看到"梯度更新"如何让策略从"两个都试试"变成"专挑好的"
- 理解策略梯度的核心挑战——高方差导致训练震荡
- 理解探索与利用的权衡——RL 最核心的张力之一

但 WHY？为什么`-log_prob * reward`这个公式能让策略学会选 B？背后的数学原理是什么？让我们在下一节——[策略梯度定理与 REINFORCE](./policy-gradient)——中逐块拆解。
