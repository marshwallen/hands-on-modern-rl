# 4.3 策略梯度定理——"好结果，就强化对应动作的概率"

在上一节中，你亲眼看到策略网络从"两个都试试"进化到了"学会选 B"。但有一个关键问题我们没有回答：**为什么 `-log_prob × reward` 这个公式能让策略学会选最优动作？**

这背后的数学原理就是**策略梯度定理**（Policy Gradient Theorem）。它是整个 Policy-Based RL 的理论基石，由 Ronald Williams 在 1992 年发表的 REINFORCE 算法论文 [^1] 中首次提出，后来由 Richard Sutton 等人在 2000 年进一步推广和系统化 [^2]。

让我们像搭积木一样，一层一层把策略梯度定理拼出来。

## 第一块积木：目标函数——"策略有多好"

在讲怎么优化之前，先要定义"什么是好"。在策略梯度方法中，我们用**期望累积奖励**来衡量一个策略的好坏：

$$J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \sum_{t=0}^{\infty} \gamma^t r_t \right]$$

符号拆解：

| 符号                      | 含义（大白话）                                             |
| ------------------------- | ---------------------------------------------------------- |
| $J(\theta)$               | 策略的"成绩"——参数为 $\theta$ 的策略平均能拿多少分         |
| $\theta$                  | 策略网络的参数（可以理解为"旋钮"——调它们就改变策略）       |
| $\pi_\theta$              | 参数为 $\theta$ 的策略函数——"给定状态，输出每个动作的概率" |
| $\mathbb{E}_{\pi_\theta}$ | 在策略 $\pi_\theta$ 下取期望（按策略行动很多次取平均）     |
| $\gamma^t r_t$            | 第 $t$ 步的折扣奖励（越远未来的奖励越不值钱）              |

**一句话**：$J(\theta)$ 就是"按策略 $\pi_\theta$ 行动，平均能拿多少分"。我们的目标就是**找到让 $J(\theta)$ 最大的参数 $\theta$**。

## 第二块积木：怎么优化？——梯度上升

怎么让 $J(\theta)$ 变大？和所有深度学习问题一样——**沿着梯度方向走**：

$$\theta \leftarrow \theta + \alpha \, \nabla_\theta J(\theta)$$

| 符号                      | 含义（大白话）                                                 |
| ------------------------- | -------------------------------------------------------------- |
| $\nabla_\theta J(\theta)$ | 目标函数对参数 $\theta$ 的梯度——"往哪个方向调参数能让成绩提升" |
| $\alpha$                  | 学习率——"每步走多大"                                           |

问题在于：$\nabla_\theta J(\theta)$ 怎么算？目标函数 $J(\theta)$ 里面有一个期望 $\mathbb{E}$——它要求你把所有可能的轨迹都跑一遍取平均，这在现实中是不可能的。

## 第三块积木：策略梯度定理——梯度的优雅表达

**策略梯度定理**的伟大之处在于：它把看似不可计算的 $\nabla_\theta J(\theta)$ 转化成了一个可以用**采样**来估计的形式：

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \sum_t \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot G_t \right]$$

让我们逐项拆解这个公式——它是 Policy-Based RL 的核心：

| 符号                                        | 含义（大白话）                    | 为什么重要                         |
| ------------------------------------------- | --------------------------------- | ---------------------------------- |
| $\nabla_\theta$                             | 对参数 $\theta$ 求梯度            | "参数该往哪调"                     |
| $\log \pi_\theta(a_t \| s_t)$               | 策略选择动作 $a_t$ 的**对数概率** | "这个动作被选中的可能性有多高"     |
| $\nabla_\theta \log \pi_\theta(a_t \| s_t)$ | 对数概率对参数的梯度              | "参数怎么调能改变这个动作的概率"   |
| $G_t$                                       | 从时刻 $t$ 到结束的累积回报       | "做了这个动作后，最终拿到了多少分" |
| 外层 $\mathbb{E}$                           | 对所有轨迹取期望                  | "跑很多次取平均"                   |

### 直觉解读：一句话总结策略梯度

**"如果一个动作导致了好的结果（$G_t$ 大），就增加再做这个动作的概率；如果导致了坏的结果（$G_t$ 小），就降低它的概率。"**

这就是策略梯度的全部思想。数学公式只是把这个直觉精确地表达出来。

### 为什么用 $\log$ 而不是直接用概率？

你可能会问：为什么不直接写成 $\nabla_\theta \pi_\theta(a_t|s_t) \cdot G_t$，而非要多一个 $\log$？

原因有两个：

1. **数学上的简化**：根据链式法则，$\nabla_\theta \log \pi = \frac{\nabla_\theta \pi}{\pi}$。这个除以 $\pi$ 的操作恰好抵消了期望中的 $\pi$ 因子，让公式变得干净。这是一个经典的数学技巧，叫做**得分函数**（Score Function）。

2. **数值稳定性**：概率 $\pi$ 的值在 0 到 1 之间，直接对概率求梯度可能会产生极小的数值。$\log$ 把 $(0, 1)$ 映射到 $(-\infty, 0)$，梯度数值更稳定。

<details>
<summary><strong>数学推导：从目标函数到策略梯度定理</strong></summary>

目标函数的梯度：

$$\nabla_\theta J(\theta) = \nabla_\theta \mathbb{E}_{\pi_\theta} \left[ \sum_t r_t \right] = \nabla_\theta \sum_{\tau} P(\tau; \theta) \sum_t r_t(\tau)$$

其中 $\tau = (s_0, a_0, s_1, a_1, \ldots)$ 是一条轨迹，$P(\tau; \theta)$ 是策略 $\pi_\theta$ 产生轨迹 $\tau$ 的概率。

利用 $\nabla_\theta P = P \cdot \nabla_\theta \log P$（对数导数技巧）：

$$\nabla_\theta J(\theta) = \sum_{\tau} P(\tau; \theta) \left( \nabla_\theta \log P(\tau; \theta) \right) \sum_t r_t(\tau)$$

轨迹概率可以分解为：$P(\tau; \theta) = \prod_t \pi_\theta(a_t|s_t) \cdot P(s_{t+1}|s_t, a_t)$

对数后对 $\theta$ 求梯度，环境转移概率 $P(s'|s,a)$ 不依赖于 $\theta$，所以只剩：

$$\nabla_\theta \log P(\tau; \theta) = \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t)$$

代回期望中就得到了策略梯度定理。

</details>

## 4.4 REINFORCE 算法

策略梯度定理告诉了我们梯度的形式。**REINFORCE** 是这个定理的直接实现——它用**蒙特卡洛采样**来估计期望。

### REINFORCE 的完整流程

| 步骤 | 操作                                                                                        | 大白话                                             |
| ---- | ------------------------------------------------------------------------------------------- | -------------------------------------------------- |
| 1    | 用当前策略 $\pi_\theta$ 采样一条完整轨迹                                                    | "按当前策略走一趟，记录每步做了什么"               |
| 2    | 计算每步的回报 $G_t = \sum_{k=t}^{T} \gamma^{k-t} r_k$                                      | "回头看，从每一步到结束总共拿了多少分"             |
| 3    | 计算梯度 $\nabla_\theta J \approx \sum_t \nabla_\theta \log \pi_\theta(a_t\|s_t) \cdot G_t$ | "好结果的动作 → 增加概率；坏结果的动作 → 降低概率" |
| 4    | 更新参数 $\theta \leftarrow \theta + \alpha \nabla_\theta J$                                | "按梯度方向微调参数"                               |

在 PyTorch 中，REINFORCE 的更新可以写成一行：

```python
loss = -log_prob * G_t  # 负号因为 PyTorch 默认做梯度下降（最小化），而我们要梯度上升（最大化）
```

### 用代码验证：摇骰子的 REINFORCE

回顾上一节的代码，`loss = -log_prob * reward` 就是 REINFORCE 在单步情况下的特例（$G_t = r_t$，因为赌博机只有一步）：

```python
# REINFORCE 的核心更新（单步版本）
loss = -log_prob * reward    # 等价于：∇θ J ≈ ∇θ log π(a|s) × G

optimizer.zero_grad()
loss.backward()              # 计算梯度
optimizer.step()             # 更新参数
```

### REINFORCE 的致命缺陷：高方差

REINFORCE 看起来简洁优雅，但有一个致命问题——**方差太大**。

为什么？因为 $G_t$ 是**整条轨迹的累积回报**——它包含了从时刻 $t$ 到结束的**所有随机性**。同一个动作，不同的采样轨迹可能给出截然不同的 $G_t$：

| 情况   | 实际发生的     | $G_t$ |
| ------ | -------------- | ----- |
| 好运气 | 每步都恰好赢了 | 很大  |
| 坏运气 | 每步都恰好输了 | 很小  |

策略梯度用这个 $G_t$ 来判断"这个动作好不好"——但 $G_t$ 的波动意味着，**同一个好的动作，可能因为运气差而被惩罚；同一个差的动作，可能因为运气好而被奖励**。这就是"醉汉走路"——方向大体是对的，但每一步都歪歪扭扭。

### 离散 vs 连续：策略网络的两种输出

本章实验用的是**离散动作空间**（选 A 或选 B）。但策略梯度定理对连续动作空间同样成立：

|                  | 离散动作空间                      | 连续动作空间                                     |
| ---------------- | --------------------------------- | ------------------------------------------------ |
| **例子**         | CartPole 左/右、LLM 选词          | 机器人关节角度、方向盘转角                       |
| **策略输出层**   | **Softmax**（输出每个动作的概率） | **高斯分布**（输出均值 $\mu$ 和标准差 $\sigma$） |
| **采样方式**     | 按 Softmax 概率随机选             | 从 $\mathcal{N}(\mu, \sigma^2)$ 中采样           |
| **log π 的计算** | `log_softmax`                     | 高斯分布的对数密度                               |

这个区别很重要——同一个 PPO 算法，换一个输出层就能从 LLM 对齐（离散）切换到机器人控制（连续）。

<details>
<summary><strong>思考题：REINFORCE 和 Q-Learning 的更新有什么本质区别？</strong></summary>

- **Q-Learning**：更新的是 $Q(s,a)$（价值函数），策略是通过 $\arg\max Q$ 隐式得到的。属于 **Value-Based**。
- **REINFORCE**：直接更新策略参数 $\theta$，不经过 Q 值。属于 **Policy-Based**。

关键差异：

- Q-Learning 是 **off-policy**（可以用旧数据），REINFORCE 是 **on-policy**（必须用当前策略的新数据）
- Q-Learning 只能处理**离散动作**（需要遍历所有动作取 max），REINFORCE 可以处理**连续动作**（直接对概率密度求梯度）

</details>

<details>
<summary><strong>思考题：为什么 REINFORCE 要跑完整个 episode 才能更新？</strong></summary>

因为 $G_t$ 需要从时刻 $t$ 到 episode 结束的**所有奖励**。不跑到终点，你就不知道 $G_t$ 的完整值。

这就是 REINFORCE 和 TD 方法的关键区别——TD 方法（如 Q-Learning）每走一步就能更新（用 $r + \gamma V(s')$ 代替 $G_t$），而 REINFORCE 必须等 episode 结束。

这也暗示了一个优化方向：如果我们能用一个**更稳定的估计**来替代 $G_t$，就可以不必等到 episode 结束——这就是下一节 Actor-Critic 要做的事情。

</details>

**你将理解**：

- 策略梯度定理的核心：$\nabla J = \mathbb{E}[\nabla \log \pi \times G_t]$——"好结果强化动作概率"
- REINFORCE 是策略梯度定理的直接实现——简洁但方差太大
- $\log$ 的作用：数学简化 + 数值稳定性
- 离散 vs 连续动作空间：换一个输出层就行

REINFORCE 能工作，但"高方差"让它几乎不可用。怎么解决？答案藏在第 3 章学过的概念里——**价值函数**。让我们看看 [Actor-Critic 架构](./actor-critic)如何把策略梯度和价值函数结合起来。

---

[^1]: Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. _Machine Learning_, 8(3-4), 229-256. [DOI](https://doi.org/10.1007/BF00992696)

[^2]: Sutton, R. S., et al. (2000). Policy gradient methods for reinforcement learning with function approximation. _Advances in Neural Information Processing Systems_, 12.
