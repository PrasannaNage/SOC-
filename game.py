import gym
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import legacy as legacy_optimizers
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
import matplotlib.pyplot as plt

# this code is for Creating the environment
env = gym.make("CartPole-v1")

states = env.observation_space.shape[0]
actions = env.action_space.n

# this part is for Building the model
model = Sequential()
model.add(Flatten(input_shape=(1, states)))
model.add(Dense(24, activation="relu"))
model.add(Dense(24, activation="relu"))
model.add(Dense(actions, activation="linear"))

# this is to Configure and compile the agent
agent = DQNAgent(
    model=model,
    memory=SequentialMemory(limit=50000, window_length=1),
    policy=BoltzmannQPolicy(),
    nb_actions=actions,
    nb_steps_warmup=10,
    target_model_update=0.01
)

# Using legacy optimizer for compatibility
agent.compile(legacy_optimizers.Adam(learning_rate=0.001), metrics=["mae"])

# Debugging: Print observation space shape
print("Observation space shape:", env.observation_space.shape)
agent.fit(env, nb_steps=100000, visualize=False, verbose=1)


results = agent.test(env, nb_episodes=10, visualize=True)
episode_rewards = results.history["episode_reward"]
plt.figure(figsize=(10, 6))
plt.plot(episode_rewards, marker='o')
plt.title('Episode Rewards')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.grid(True)
plt.tight_layout()
plt.show()
print(np.mean(results.history["episode_reward"]))

env.close()
