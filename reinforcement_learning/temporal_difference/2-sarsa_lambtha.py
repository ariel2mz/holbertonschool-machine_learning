#!/usr/bin/env python3
"""sarsalambtha"""
import numpy as np


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1,
                  gamma=0.99, epsilon=1, min_epsilon=0.1,
                  epsilon_decay=0.05):
    """
    asgashasgas
    """
    E = np.zeros_like(Q)
    nS, nA = Q.shape
    
    for _ in range(episodes):
        sta, _ = env.reset()
        E.fill(0)
        
        if np.random.uniform() < epsilon:
            act = np.random.randint(nA)
        else:
            act = np.argmax(Q[sta])
        
        for _ in range(max_steps):
            nextsta, rew, ter, trunc, _ = env.step(act)
            
            if np.random.uniform() < epsilon:
                nextact = np.random.randint(nA)
            else:
                nextact = np.argmax(Q[nextsta])
            
            td_err = rew + gamma * Q[nextsta, nextact] - Q[sta, act]
            
            E[sta, act] += 1
            
            Q += alpha * td_err * E
            E *= gamma * lambtha
            
            sta, act = nextsta, nextact
            
            if ter or trunc:
                break
        
        epsilon = max(min_epsilon, epsilon - epsilon_decay)
                
    return Q