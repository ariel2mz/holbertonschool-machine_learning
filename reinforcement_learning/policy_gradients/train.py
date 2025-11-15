#!/usr/bin/env python3
"""
afsfsafsa
"""
import numpy as np


def train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False):
    """
    asfgsasafsa
    """

    policy_gradient = __import__('policy_gradient').policy_gradient

    acts = env.action_space.n
    feats = env.observation_space.shape[0]
    weight = np.random.rand(feats, acts)

    scores = []

    for ep in range(nb_episodes):
        state, _ = env.reset()
        eprew = []
        done = False
        if show_result and ep % 1000 == 0:
            env.render()

        while not done:

            act, grad = policy_gradient(state, weight)

            nextstate, reward, terminated, truncated, _ = env.step(act)
            done = terminated or truncated
            weight += alpha * grad * reward

            eprew.append(reward)
            state = nextstate

        score = sum(eprew)
        scores.append(score)

        print("Episode: {} Score: {}".format(ep, score))

    return scores
