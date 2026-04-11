# 第 4 章：DQN与Atari——让AI看懂像素玩游戏

> **本章目标**：理解基于价值的方法（Value-Based Methods）。你将从零手写 DQN 的三个核心组件（Q-Network、经验回放、目标网络），然后用它玩 Atari 游戏，最后看到从 DQN 到 Rainbow 的完整演进脉络。
>
> **承上启下**：第 3 章我们在 GridWorld 里手算了 Q-Learning——用一张表格存每个状态-动作对的 Q 值。但表格方法无法处理高维状态空间（如 Atari 的像素输入）。本章给 Q-Learning 套上一层神经网络，就是 DQN。**从表格（第 3 章）到神经网络（本章），这条线清晰可见。**

## 9.1 从零手写 DQN：三个核心组件

- 在第 3 章的 GridWorld 中，Q-Learning 用表格存储 Q(s,a)。但 Atari 的输入是 84×84×4 像素帧——状态空间是天文数字，表格根本装不下
- DQN 的解决方案很简单：**用神经网络替代表格**。输入像素，输出每个动作的 Q 值
- 但"直接用神经网络"会训练崩溃，需要两个关键工程技巧：

### 组件 1：Q-Network（Q 网络）

```python
class QNetwork(nn.Module):
    """输入状态，输出每个动作的 Q 值"""
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)  # 输出每个动作的 Q 值
        )

    def forward(self, x):
        return self.net(x)
```

- **你将理解**：Q-Network 就是第 3 章讲的"函数逼近器"——用神经网络 $Q(s,a;\theta)$ 来近似真实的 $Q^*(s,a)$

### 组件 2：经验回放（Experience Replay）

```python
class ReplayBuffer:
    """存储 (s, a, r, s') 转移，随机采样打破相关性"""
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity

    def push(self, transition):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
```

- **为什么需要？** 连续的 Atari 帧几乎一模一样——如果逐帧训练，梯度会被"当前帧"绑架。经验回放把历史经验存进池子，随机采样打破相关性
- **类比**：不是只复习昨天的错题，而是把所有错题打乱顺序随机练习

### 组件 3：目标网络（Target Network）

```python
# 每隔 C 步，把 Q 网络的参数复制到目标网络
target_net.load_state_dict(q_net.state_dict())

# 计算 TD Target 时用目标网络（稳定的靶子）
td_target = r + gamma * target_net(s').max()
```

- **为什么需要？** Q 值更新的目标 $r + \gamma \max Q(s', a')$ 本身也在变——"追自己的尾巴"。目标网络是一个慢更新的副本，提供短期稳定的训练目标
- **类比**：射箭时靶子不能乱动——目标网络就是那个固定的靶子

- **你将获得**：从零理解 DQN 的三个组件，和第 3 章 Q-Learning 手算的经验直接呼应

## 9.2 动手：用 DQN 玩 Pong（Atari）

- 运行现成实现（Stable Baselines3 或 CleanRL），看 AI 从永远漏球到逐渐学会对打
- 观察 Q 值曲线（每个动作的"预期得分"）和 Episode Reward 的变化
- **你将获得**：DQN 训练 Atari 游戏的完整体验——从像素到决策的全流程

## 9.3 观察训练过程

- 训练初期 Q 值爆炸？——因为目标不稳定，网络在追一个不断移动的靶子
- Reward 长期不涨然后突然跳升？——DQN 的学习存在"平台期"，需要足够的经验积累
- 经验回放池的大小如何影响训练？——太小则样本多样性不足，太大则旧数据过时
- **你将思考**：为什么 DQN 需要这么多"工程技巧"才能稳定训练？

## 9.4 理论回顾：从 Q-Learning 到 DQN

- Q-Learning：维护一个 Q 表格，Q(s, a) = "在状态 s 做动作 a 的期望总回报"
- 更新规则：Q(s,a) ← Q(s,a) + α [r + γ max Q(s',a') - Q(s,a)]
- **维度灾难**：CartPole 的状态是 4 维连续 → 无法穷举所有 (s,a) 组合存表格
- Atari 的输入是 84×84×4 的像素帧 → 状态空间是天文数字
- DQN 的解决方案：用神经网络近似 Q 函数，输入像素，输出每个动作的 Q 值
- **你将理解**：DQN 的核心贡献——用函数近似解决了 Q-Learning 的维度灾难

## 9.5 DQN 两大支柱（深入理解）

- **经验回放（Experience Replay）**：
  - 问题：连续样本高度相关（相邻帧几乎一样），违反 SGD 的独立同分布假设
  - 方案：把所有 (s, a, r, s') 存进一个池子，每次随机采样一批训练
  - 效果：打破样本相关性，提高数据利用效率
- **目标网络（Target Network）**：
  - 问题：Q 值更新的目标 r + γ max Q(s',a') 也在变，"追自己尾巴"
  - 方案：维护一个慢更新的目标网络 Q_target，每隔 C 步同步一次
  - 效果：训练目标短期稳定，避免振荡
- **你将理解**：DQN 能成功不是因为"深度学习厉害"，而是这两个工程技巧解决了稳定性问题

## 9.6 DQN 改进谱系

- **Double DQN（2015）**：分离动作选择和动作评估，解决 Q 值过估计问题
  - 原 DQN：用同一个网络选动作和评估 → 倾向于高估
  - Double DQN：一个网络选动作，另一个评估 → 更准确
- **Dueling DQN（2016）**：把 Q 分解为 V(s) + A(s,a)，更高效地学习状态价值
  - 有些状态本身就很好/很差（与动作无关），Dueling 结构让模型单独学这一点
- **Prioritized Experience Replay（2016）**：优先回放 TD Error 大的样本（"让模型多复习它做错的题"）
- **Rainbow（2017）**：集成以上所有改进 + Distributional RL + Noisy Nets，集大成者
- **你将理解**：从 DQN 到 Rainbow 的 5 年演进，每一步都在解决一个具体问题

## 9.7 回看实践：分析 DQN 训练日志

- TD Error 分布 → 哪些样本"惊讶"了模型？（TD Error 大 = 模型预测与实际差距大）
- 目标网络更新频率对稳定性的影响：更新太频繁 → 训练不稳定；太稀疏 → 学习慢
- 经验回放池的热力图：哪些状态被采样最多？
- **你将获得**：DQN 训练的调优经验

## 9.8 视角迁移：从 Atari 到 LLM

- CNN 编码像素 → Transformer 编码 Token——都是"从高维输入中提取有用表示"
- 经验回放 → RLHF 中的偏好数据集——都是"利用历史数据提高训练效率"
- 目标网络 → 参考模型（Reference Model）——都是"提供一个稳定的锚点防止跑偏"
- 两者都面临：高维状态空间、稀疏奖励、训练不稳定的挑战
- **你将理解**：RL 的核心问题在不同领域是相通的，DQN 的思想在今天仍然适用
