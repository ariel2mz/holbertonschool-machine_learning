#!/usr/bin/env python3
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99):
    """
    asasgasfsagsa
    """
    for _ in range(episodes):
        sta, _ = env.reset()
        epi = []

        for _ in range(max_steps):
            act = policy(sta)
            nextsta, rew, term, trunc, _ = env.step(act)
            epi.append((sta, rew))
            sta = nextsta
            if term or trunc:
                break

        vis = set()
        G = 0
        for sta, rew in reversed(epi):
            G = rew + gamma * G
            if sta not in vis:
                V[sta] = V[sta] + alpha * (G - V[sta])
                vis.add(sta)

    return V