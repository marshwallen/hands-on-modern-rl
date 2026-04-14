# 第1章：传统RL初体验——5分钟让AI学会走平衡木

> **本章目标**：零基础运行第一个 RL 训练脚本，获得"AI 能自己学会一件事"的直接体验。不要求任何理论知识。

> 📁 **本章代码**：[1-ppo_cartpole.py](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/code/chapter01_cartpole/1-ppo_cartpole.py) · [2-ppo_cartpole_tensorboard.py](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/code/chapter01_cartpole/2-ppo_cartpole_tensorboard.py) · [3-pytorch_from_scratch.py](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/code/chapter01_cartpole/3-pytorch_from_scratch.py) · [requirements.txt](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/code/chapter01_cartpole/requirements.txt)

## 1.1 动手：运行 CartPole 训练

经过上面的前置阅读，我们已经对强化学习有了一个模糊的最初印象。在此之前，先让我们复习一下强化学习的核心诉求：它本质上是由一个智能体（Agent）和环境（Environment）组成，通过不断地尝试动作并获取奖励，智能体能够在基于大量数据训练的基础上，推理出在特定状态下最恰当的决策输出。

更具体而言，什么才是"恰当的决策"呢？这要追溯到强化学习界当之无愧的"Hello World"——CartPole（倒立摆）任务。如果说写下 `print("Hello World")` 是所有程序员的第一次代码放电，那么用几十行代码让一根木棍在小车上保持直立，就是你迈向强化学习大门的第一把钥匙。

你可能会问：训练这样一个 AI 难吗？它会不会很重？我的 Mac 能不能跑起来？

大可不必担心。得益于现代开源生态，它非常非常轻量，你的个人电脑（无论是老款 Intel Mac 还是最新的 Apple Silicon，亦或是 Windows/Linux）都可以完美运行！

- **不需要显卡 (GPU)**：这个任务极其简单，完全只用 CPU 就能跑。
- **内存占用极小**：运行时的内存占用通常在 100MB~200MB 左右。
- **模型极其迷你**：我们用到的神经网络策略（`MlpPolicy`）只有两层，每层 64 个神经元，参数量只有区区几千个。在动辄百亿参数的大模型时代，这简直就像一粒尘埃。

在这个过程中，我们将使用 Gymnasium（当前 RL 环境的绝对标准）作为训练场，并引入 Stable Baselines3 (SB3) 算法库。如果说 PyTorch 是造汽车的零件，那么 SB3 就是已经组装好的发动机，它把前沿的 PPO 算法封装成了几行极简的代码。

我们不需要立刻理解复杂的微积分或高维矩阵运算。废话不多说，让我们直接动手，用代码实现一个最简单的倒立摆智能体。

### **第一步：安装依赖**

首先，打开你的终端，安装环境库和算法库：

```bash
pip install "gymnasium[classic-control]" stable-baselines3
```

> **注意**：`stable-baselines3` 依赖于 `PyTorch`。由于 PyTorch 体积较大，这一步可能会因为网速原因稍等片刻。别担心，这是整个第一章唯一的"重体力活"。

### **第二步：编写训练与演示代码**

你可以直接运行配套代码：[1-ppo_cartpole.py](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/code/chapter01_cartpole/1-ppo_cartpole.py)（`pip install -r requirements.txt` 安装依赖后运行）

这段代码包含两部分：首先使用 PPO 算法训练智能体，然后渲染一个可视化窗口，让你亲眼见证它学到的"杂技"。

```python
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

# ==========================================
# 第一阶段：训练智能体
# ==========================================
print("正在创建 CartPole 环境...")
# 创建倒立摆环境
env = gym.make("CartPole-v1")

# 初始化 PPO (近端策略优化) 算法模型，使用多层感知机 (MlpPolicy)
model = PPO("MlpPolicy", env, verbose=1)

print("开始训练，请稍候 (通常只需几秒钟)...")
# 训练 20000 个时间步
model.learn(total_timesteps=20000)

# 评估训练好的模型
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"训练完成！平均奖励: {mean_reward} +/- {std_reward}")

# 保存模型
model.save("ppo_cartpole")
env.close()

# ==========================================
# 第二阶段：可视化展示学习成果
# ==========================================
print("正在展示智能体的学习成果...")
# 重新创建一个带有渲染画面的环境
env = gym.make("CartPole-v1", render_mode="human")
model = PPO.load("ppo_cartpole")

# 运行 5 个回合的视觉演示
for episode in range(5):
    obs, info = env.reset()
    done = False
    truncated = False
    score = 0

    while not (done or truncated):
        # 智能体根据当前观察 (obs) 决定动作
        action, _states = model.predict(obs, deterministic=True)
        # 环境执行动作，返回新的状态和奖励
        obs, reward, done, truncated, info = env.step(action)
        score += reward

    print(f"回合 {episode + 1} 得分: {score}")

env.close()
```

运行这段代码，你会看到控制台快速滚动训练日志。几秒钟后，屏幕上会弹出一个小窗口，一辆小车正在你眼前完美地平衡着木棍。

你看，短短几十行代码，你的电脑就已经通过自我试错，掌握了一项物理控制技能。那么，在这个黑盒里究竟发生了什么？从下一节开始，我们将打开黑盒，一同欣赏隐藏在日志背后的秘密。

## 1.2 观察与疑问：训练曲线会说话

如果你仔细观察训练过程中的控制台输出，你会注意到 SB3 默认会每隔一段时间打印一行日志，类似这样：

```
-----------------------------------------
| time/              |                  |
|    fps             | 5342             |
|    iterations      | 1                |
|    time_elapsed    | 3                |
|    total_timesteps | 2048             |
| train/             |                  |
|    entropy_loss    | -0.683           |
|    learning_rate   | 0.0003           |
|    loss            | 0.0124           |
|    policy_gradient_loss | -0.0187     |
|    value_loss      | 8.2741           |
-----------------------------------------
```

这里面信息量很大，但目前我们只需要关注一件事：**模型到底有没有在变好？**

要回答这个问题，最直接的方式是把每一轮训练的"平均得分"画成曲线。SB3 内置了 TensorBoard 支持，完整代码见 [2-ppo_cartpole_tensorboard.py](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/code/chapter01_cartpole/2-ppo_cartpole_tensorboard.py)，你只需要把训练代码中的 `model.learn(...)` 那一行前面加上一个参数：

```python
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_cartpole_tensorboard/")
model.learn(total_timesteps=20000)
```

然后在终端运行：

```bash
tensorboard --logdir ./ppo_cartpole_tensorboard/
```

打开浏览器访问 `http://localhost:6006`，你将看到一条蜿蜒上升的曲线——这就是 `rollout/ep_rew_mean`（平均回合奖励）。这条曲线就是智能体的"学习成绩单"。

### **典型的训练曲线长什么样？**

你大概率会看到这样的走势：

1. **开局低迷（step 0 ~ 3000）**：曲线在 20~40 分之间晃悠。这是因为智能体的策略是完全随机的——它不知道该往哪推，杆子很快就倒了。就像一个从没摸过方向盘的新手，车子东摇西晃。

2. **快速爬升（step 3000 ~ 10000）**：曲线开始明显上扬，从 40 爬到 150 甚至更高。智能体开始"开窍"了——它发现某些动作模式能让杆子立得更久。这个阶段是最令人兴奋的，你能真切地感受到"它在学习"。

3. **趋于稳定（step 10000 以后）**：曲线在 180~200 之间小幅波动。智能体已经掌握了平衡杆子的诀窍，偶尔有些小失误，但总体表现稳定。

### **值得思考的三个问题**

在看曲线的过程中，试着回答下面三个问题。它们将引导你理解 RL 训练的本质特征：

<details>
<summary><strong>问题一：为什么刚开始得分这么低？</strong></summary>

因为智能体什么都不会。它的策略是一个随机初始化的神经网络——本质上等价于抛硬币决定往哪推。CartPole-v1 的最大步数是 500 步（满分 500），一个随机策略平均只能坚持 20 步左右。

</details>

<details>
<summary><strong>问题二：为什么曲线不是平滑上升，而是锯齿状的？</strong></summary>

RL 训练使用的是随机采样。每次智能体与环境交互时，具体会发生什么带有随机性——就像你连续投篮 10 次，每次进球数不完全一样。这种随机性导致每个回合的得分有波动，所以平均值呈现锯齿状。只要锯齿的整体趋势是向上的，训练就是在正常进行。

</details>

<details>
<summary><strong>问题三：如果我把 <code>total_timesteps</code> 改成 5000（原来是 20000），会发生什么？</strong></summary>

训练会提前结束，智能体可能还没有学会稳定的平衡技巧——你会在演示阶段看到杆子经常倒下。这是 RL 中"数据量不够"的最直接体现：学习需要经验积累，太少的练习时间意味着不够成熟。

</details>

> **动手实验**：试着把 `total_timesteps` 分别改成 5000、10000、50000，对比三次训练后智能体的表现差异。你会直观地感受到"训练时间"和"学习效果"之间的关系。
