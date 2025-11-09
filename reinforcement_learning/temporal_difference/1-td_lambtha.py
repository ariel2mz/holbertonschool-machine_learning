#!/usr/bin/env python3
"""tdlambtha"""
import numpy as np


def td_lambtha(env, V, policy, lambtha,
               episodes=5000, max_steps=100, alpha=0.1, gamma=0.99):
    E = np.zeros_like(V)

    for _ in range(episodes):
        sta, _ = env.reset()
        E.fill(0)

        for _ in range(max_steps):
            act = policy(sta)
            nextsta, rew, ter, trunc, _ = env.step(act)

            td_err = rew + gamma * V[nextsta] - V[sta]

            E[sta] += 1

            V += alpha * td_err * E
            E *= gamma * lambtha

            sta = nextsta

            if ter or trunc:
                break

    return V
