import gym
import numpy as np
import random
import time

start_time = time.time()

# Hyperparameters
alpha = 0.1
gamma = 0.6
epsilon = 0.1

# Initialize Q-table to 0
env = gym.make("Taxi-v3").env
q_table = np.zeros([env.observation_space.n, env.action_space.n])

# For plotting metrics
all_epochs = []
all_penalties = []

for i in range(1, 100001):
    state = env.reset()

    epochs, penalties, reward, = 0, 0, 0
    done = False

    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample() # Explore action space
    else:
        action = np.argmax(q_table[state]) # Exploit learned values

    next_state, reward, done, info = env.step(action)

    print("State:", state)
    print("Next State:", next_state)
    print("Action:", action)

    old_value = q_table[state, action]
    next_max = np.max(q_table[next_state])

    new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
    q_table[state, action] = new_value

    if reward == -10:
            penalties += 1

    state = next_state
    epochs += 1

    if i % 30 == 0:  # Changed from 1000 to 100
        elapsed_time = time.time() - start_time
        episodes_left = 100000 - i
        time_per_episode = elapsed_time / i
        estimated_time_left = episodes_left * time_per_episode

        print(f"Episode: {i}")
        print(f"Estimated time remaining: {estimated_time_left} seconds")

print("Training finished.\n")