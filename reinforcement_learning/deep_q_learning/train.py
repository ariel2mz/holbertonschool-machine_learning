#!/usr/bin/env python3
"""
safas
"""
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStack
import numpy as np
import tensorflow as tf
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory

ENV_ID = "ALE/Breakout-v5"
NB_STEPS = 1_000_000
WARMUP_STEPS = 50_000
MEMORY_LIMIT = 1_000_000
TARGET_MODEL_UPDATE = 10_000
LEARNING_RATE = 0.00025
POLICY_WEIGHTS_FILE = "policy.h5"
WINDOW_LENGTH = 4

def make_env(env_id=ENV_ID):
    env = gym.make(env_id, render_mode=None)
    env = AtariPreprocessing(
        env, 
        grayscale_obs=True, 
        frame_skip=4, 
        screen_size=84, 
        scale_obs=False
    )
    env = FrameStack(env, num_stack=WINDOW_LENGTH)
    return env

def build_model(obs_shape, nb_actions):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=obs_shape))
    model.add(tf.keras.layers.Conv2D(32, kernel_size=8, strides=4, activation="relu"))
    model.add(tf.keras.layers.Conv2D(64, kernel_size=4, strides=2, activation="relu"))
    model.add(tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, activation="relu"))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation="relu"))
    model.add(tf.keras.layers.Dense(nb_actions, activation="linear"))
    return model

def build_agent(model, nb_actions):
    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(),
        attr='eps',
        value_max=1.0,
        value_min=0.1,
        value_test=0.05,
        nb_steps=NB_STEPS // 2
    )
    
    memory = SequentialMemory(limit=MEMORY_LIMIT, window_length=1)
    
    dqn = DQNAgent(
        model=model,
        nb_actions=nb_actions,
        memory=memory,
        policy=policy,
        nb_steps_warmup=WARMUP_STEPS,
        target_model_update=TARGET_MODEL_UPDATE,
        gamma=0.99,
        enable_double_dqn=True,
        enable_dueling_network=True,
        dueling_type='avg'
    )
    
    dqn.compile(
        tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, epsilon=1e-4),
        metrics=["mae"]
    )
    return dqn

def main():
    env = make_env()
    obs_shape = env.observation_space.shape
    nb_actions = env.action_space.n
    
    print(f"Observation shape: {obs_shape}")
    print(f"Number of actions: {nb_actions}")
    
    model = build_model(obs_shape, nb_actions)
    print(model.summary())
    
    dqn = build_agent(model, nb_actions)

    history = dqn.fit(
        env, 
        nb_steps=NB_STEPS, 
        visualize=False, 
        verbose=2,
        log_interval=10000
    )
    
    dqn.save_weights(POLICY_WEIGHTS_FILE, overwrite=True)
    dqn.test(env, nb_episodes=5, visualize=False)
    
    env.close()

if __name__ == "__main__":
    main()