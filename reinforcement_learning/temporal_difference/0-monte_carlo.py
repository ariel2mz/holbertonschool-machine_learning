#!/usr/bin/env python3
"""sagasgas"""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1,
                gamma=0.99):
    """
h
    """
    for x in range(episodes):
        sta, _ = env.reset()
        epi = []

        for _ in range(max_steps):
            act = policy(sta)
            nextsta, rew, ter, trunc, _ = env.step(act)
            epi.append((sta, rew))
            sta = nextsta
            if ter or trunc:
                break
        G = 0

        for t in range(len(epi) - 1, -1, -1):
            sta, rew = epi[t]
            G = rew + gamma * G

            # Check if this is the first time we see this state in the episode
            first_visit = True
            for i in range(t):
                if epi[i][0] == sta:
                    first_visit = False
                    break
            
            # Only update if first visit AND state is not a terminal state (hole)
            if first_visit and V[sta] != -1:
                V[sta] += alpha * (G - V[sta])

    return V
