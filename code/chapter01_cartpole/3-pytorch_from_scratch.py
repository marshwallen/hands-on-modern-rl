"""
第1章：撕开黑盒 —— 用纯 PyTorch 实现 CartPole 训练
展示 SB3 的 model.learn() 背后的核心逻辑

注意：这是一个教学用的简化实现（REINFORCE 风格），
并非完整的 PPO 算法。完整实现见第 5 章。

运行方式：
    python pytorch_from_scratch.py
"""

import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np


# ==========================================
# 第一部分：策略网络
# ==========================================
class PolicyNetwork(nn.Module):
    """一个迷你的策略网络：4 个状态输入 → 2 个动作概率输出"""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64),  # 输入：4 个状态数字
            nn.ReLU(),
            nn.Linear(64, 64),  # 隐藏层
            nn.ReLU(),
            nn.Linear(64, 2),  # 输出：2 个动作的 logits
        )

    def forward(self, state):
        logits = self.net(state)
        probs = torch.softmax(logits, dim=-1)
        return probs


# ==========================================
# 第二部分：采样轨迹
# ==========================================
def collect_trajectory(policy, env):
    """让策略在环境中玩一局，收集所有 (状态, 动作, 奖励) 数据"""
    obs, _ = env.reset()
    trajectory = []

    for _ in range(500):  # CartPole-v1 最多 500 步
        state_tensor = torch.FloatTensor(obs)

        # 策略网络输出动作概率
        with torch.no_grad():
            probs = policy(state_tensor)

        # 按概率随机采样一个动作（探索）
        action = torch.multinomial(probs, num_samples=1).item()

        # 执行动作
        next_obs, reward, done, truncated, _ = env.step(action)

        trajectory.append((obs.copy(), action, reward))
        obs = next_obs

        if done or truncated:
            break

    return trajectory


# ==========================================
# 第三部分：策略更新
# ==========================================
def compute_loss(policy, trajectory, gamma=0.99):
    """计算 REINFORCE 风格的策略梯度损失"""
    # 计算每一步的折扣累计回报 G_t
    rewards = [r for (_, _, r) in trajectory]
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.FloatTensor(returns)

    # 归一化回报（减少方差）
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    # 计算策略梯度损失
    loss = 0
    for (obs, action, _), G in zip(trajectory, returns):
        state_tensor = torch.FloatTensor(obs)
        probs = policy(state_tensor)
        log_prob = torch.log(probs[action])
        loss += -log_prob * G  # 核心公式：好的结果 → 提高概率

    return loss


# ==========================================
# 第四部分：训练循环
# ==========================================
def train():
    env = gym.make("CartPole-v1")
    policy = PolicyNetwork()
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)

    print("开始训练（纯 PyTorch 实现）...")
    print("-" * 50)

    for iteration in range(100):
        # 第一步：收集经验数据
        trajectory = collect_trajectory(policy, env)
        episode_reward = sum(r for (_, _, r) in trajectory)

        # 第二步：计算损失
        loss = compute_loss(policy, trajectory)

        # 第三步：更新参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 打印进度
        if (iteration + 1) % 10 == 0:
            print(
                f"  迭代 {iteration + 1:3d} | "
                f"回合奖励: {episode_reward:5.0f} | "
                f"损失: {loss.item():.4f}"
            )

    print("-" * 50)

    # 最终评估
    total_rewards = []
    for _ in range(20):
        trajectory = collect_trajectory(policy, env)
        total_rewards.append(sum(r for (_, _, r) in trajectory))

    mean_reward = np.mean(total_rewards)
    print(f"\n训练完成！20 回合平均奖励: {mean_reward:.1f}")

    # 可视化演示（如果有图形界面）
    try:
        vis_env = gym.make("CartPole-v1", render_mode="human")
        print("\n正在演示学习成果（5 个回合）...")
        for ep in range(5):
            obs, _ = vis_env.reset()
            done, truncated, score = False, False, 0
            while not (done or truncated):
                state_tensor = torch.FloatTensor(obs)
                with torch.no_grad():
                    probs = policy(state_tensor)
                action = torch.argmax(probs).item()  # 演示时选最优动作
                obs, reward, done, truncated, _ = vis_env.step(action)
                score += reward
            print(f"  演示回合 {ep + 1} 得分: {score}")
        vis_env.close()
    except Exception:
        print("(跳过可视化演示，无图形界面)")

    env.close()


if __name__ == "__main__":
    train()
