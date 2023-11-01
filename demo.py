# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 20:50:35 2023

@author: EE720A
"""

# 使用訓練好的模型進行遊戲
import gym
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

 

env = gym.make('ALE/SpaceInvaders-v5', render_mode='human')

# 載入模型
mode_path = os.path.join('C:/Users/EE720A/Desktop/final/use/v5.h5')
model = load_model(mode_path)
# 使用載入的模型進行遊戲
num_episodes = 3  # 要玩幾個遊戲
set_reward = []

for episode in range(num_episodes):
    state = env.reset()
    state_array = state[0].astype(np.uint8)
    state = np.reshape(state_array, (1,) + state_array.shape)
    state = tf.convert_to_tensor(state, dtype=tf.uint8)
    done = False
    total_reward = 0
    while not done:
        # 使用模型預測動作
        q_values = model.predict(state, verbose=0)
        action = np.argmax(q_values[0])

        next_state, reward, done, info, _ = env.step(action)
        #env.render()
        next_state = np.array(next_state)
        next_state = np.reshape(next_state, (1,) + next_state.shape)
        next_state = tf.convert_to_tensor(next_state, tf.uint8)
        total_reward += reward

        state = next_state

        if done:
            print(f"Reward: {total_reward}")
            break
    
       
        
    

env.close()