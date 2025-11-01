#!/usr/bin/env python3
"""
play.py
Display a game played by the agent trained by train.py.
"""
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStack
import numpy as np
import tensorflow as tf
from rl.agents.dqn import DQNAgent
from rl.policy import GreedyQPolicy
from rl.memory import SequentialMemory

ENV_ID = "ALE/Breakout-v5"
POLICY_WEIGHTS_FILE = "policy.h5"
WINDOW_LENGTH = 4

def make_env(env_id=ENV_ID):
    env = gym.make(env_id, render_mode="human")
    env = AtariPreprocessing(env, grayscale_obs=True, frame_skip=4, screen_size=84, scale_obs=False)
    env = FrameStack(env, num_stack=WINDOW_LENGTH)
    return env

def build_model(obs_shape, nb_actions):
    model = tf.keras.Sequential()
    model.add(tf.keras.InputLayer(input_shape=obs_shape))

    if obs_shape[0] == WINDOW_LENGTH:
        model.add(tf.keras.Permute((2, 3, 1)))
    model.add(tf.keras.Conv2D(32, kernel_size=8, strides=4, activation="relu"))
    model.add(tf.keras.Conv2D(64, kernel_size=4, strides=2, activation="relu"))
    model.add(tf.keras.Conv2D(64, kernel_size=3, strides=1, activation="relu"))
    model.add(tf.keras.Flatten())
    model.add(tf.keras.Dense(512, activation="relu"))
    model.add(tf.keras.Dense(nb_actions, activation="linear"))
    return model

def build_agent(model, nb_actions):
    policy = GreedyQPolicy()
    memory = SequentialMemory(limit=1000, window_length=WINDOW_LENGTH)
    dqn = DQNAgent(
        model=model,
        nb_actions=nb_actions,
        memory=memory,
        policy=policy,
        target_model_update=1e-2,
        gamma=0.99
    )
    dqn.compile(tf.keras.Adam(learning_rate=0.00025), metrics=["mae"])
    return dqn

def main():
    env = make_env()
    obs_shape = env.observation_space.shape
    nb_actions = env.action_space.n

    model = build_model(obs_shape, nb_actions)
    dqn = build_agent(model, nb_actions)
    dqn.load_weights(POLICY_WEIGHTS_FILE)

    dqn.test(env, nb_episodes=5, visualize=True)

    env.close()

if __name__ == "__main__":
    main()
