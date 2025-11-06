#!/usr/bin/env python3
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99):
    """
    asasgasfsagsa
    """
    for i in range(episodes):
        sta, i = env.reset()
        epi = []

        for i in range(max_steps):
            act = policy(sta)
            nexsta, rew, term, trunc, i = env.step(act)
            epi.append((sta, rew))
            sta = nextsta
            if term or trunc:
                break

        G = 0
        for sta, rew in reversed(epi):
            G = rew + gamma * G
            V[sta] = V[sta] + alpha * (G - V[sta])

    return V
