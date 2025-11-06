#!/usr/bin/env python3
"""sagasgas"""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1,
                gamma=0.99):
    """

    """
    for x in range(episodes):
        sta, _ = env.reset()
        epi = []

        for _ in range(max_steps):
            act = policy(sta)
            nextstate, rew, ter, trunc, _ = env.step(act)
            epo.append((sta, rew))
            sta = nextstate
            if ter or trunc:
                break
        G = 0
        epi = np.array(episode, dtype=int)

        for t in range(len(episodes) - 1, -1, -1):
            sta, rew = epi[t]
            G = rew + gamma * G

            if sta not in epi[:x, 0]:
                V[sta] += alpha * (G - V[sta])

    return V
