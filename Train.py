import gym
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.optimizers import Adam
import numpy as np
from collections import deque
import random
from tensorflow.keras.models import load_model

class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done  = True

    def step(self, action):
        next_state, reward, done, info, _  = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return next_state, reward, done, info, _ 


env = gym.make('ALE/SpaceInvaders-v5', render_mode='rgb_array')
env = EpisodicLifeEnv(env)
#env = gym.make('ALE/SpaceInvaders-v5', render_mode='human')
random_number = lambda:random.randint(0,5)
tf.keras.backend.clear_session()
model = load_model('C:/Users/EE720A/Desktop/final/use/v5.h5')
#model = Sequential()
#model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=env.observation_space.shape))
#model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
#model.add(Conv2D(64, (3, 3), activation='relu'))
#model.add(Flatten())
#model.add(Dense(512, activation='relu'))
#model.add(Dense(256, activation='relu'))
#model.add(Dense(env.action_space.n, activation='linear'))

#model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))

# Training
epsilon = 0.1 #Exploration rate
epsilon_decay = 0.999  # Decay rate of epsilon
epsilon_min = 0.01  # Minimum exploration rate
batch_size = 64  # Batch size for training
num_episodes = 6000  # Total number of episodes for training
memory = deque(maxlen=15000) # Replay buffer
set_reward = []
step = 0


for episode in range(num_episodes):
    state = env.reset()
    state_array = state[0].astype(np.uint8)
    state = np.reshape(state_array, (1,) + state_array.shape)
    state = tf.convert_to_tensor(state, dtype=tf.uint8)
    done = False
    total_reward = 0
    stop = 0
    same_action = 0;
    same_stop = 0
    
    while not done:
        if np.random.rand() <= epsilon:
            action = random_number()
        else:
            q_values = model.predict(state, verbose=0)
            action = np.argmax(q_values[0])
            if action == 0:
                stop = stop+1
            elif same_action == action:
                same_stop = same_stop+1
            else:
                stop = 0
                same_stop = 0
            same_action = action

        next_state, reward, done, info, _ = env.step(action)
        #env.render()
        next_state = np.array(next_state)
        next_state = np.reshape(next_state, (1,) + next_state.shape)
        next_state = tf.convert_to_tensor(next_state, tf.uint8)
        total_reward += reward
        if reward > 199:
            reward = 150
            same_stop = 0
        elif reward > 1:
            reward = 150 
            same_stop = 0
        elif done:
            reward = -3000
        elif stop > 40 or same_stop > 60 :
            reward = -100
            
        
            
        
        
        memory.append((state, action, reward, next_state, done))
        state = next_state
        step = step + 1
        
        if step%5000 == 0 and len(set_reward)!=0:
            ave = int(sum(set_reward)/len(set_reward))
            save_path = os.path.join( 'C:/Users/EE720A/Desktop/final/model/model0611/',f's{step}.h5')
            model.save(save_path)
            model.save('C:/Users/EE720A/Desktop/final/model/model0611/space_invaders_model.h5')
            with open("C:/Users/EE720A/Desktop/final/model/model0611/aver.txt", "a") as file:
                file.write("step:"+str(step)+"epsilon:"+str(epsilon)+" aver:" +str(ave) + "\n")
            set_reward = []

        if done:
            set_reward.append(total_reward)
            break

    if len(memory) >= batch_size:
        batch = random.sample(memory, batch_size)

        states = tf.concat([s[0] for s in batch], axis=0)
        actions = np.array([s[1] for s in batch])
        rewards = np.array([s[2] for s in batch])
        next_states = tf.concat([s[3] for s in batch], axis=0)
        dones = np.array([s[4] for s in batch])

        targets = model.predict(states, verbose=0)
        next_q_values = model.predict(next_states, verbose=0)
        targets[np.arange(batch_size), actions] = rewards + 0.99 * np.max(next_q_values, axis=1) * (1 - dones)

        model.fit(states, targets, epochs=1, verbose=0)

    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
    

    print(f"Episode: {episode + 1}, Reward: {total_reward}, Epsilon: {epsilon:.4f}, Step: {step:d}, Mem: {len(memory)}")
env.close()
model.save('space_invaders_model.h5')

#%%
# 使用訓練好的模型進行遊戲
import gym
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow as tf

env = gym.make('ALE/SpaceInvaders-v5', render_mode='rgb_array')
# 載入模型


# 使用載入的模型進行遊戲
num_episodes = 3  # 要玩幾個遊戲
set_reward = []
for modtest in range(0,95):
    
    model_path = os.path.join( 'C:/Users/EE720A/Desktop/final/model/model0608/',f's{(modtest+1)*5000}.h5')
    
    model = load_model(model_path)
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
                set_reward.append(total_reward)
                break
        print(f"Model: {(modtest+1)*5000}, Episode: {episode + 1}, Reward: {total_reward}")
    ave = int(sum(set_reward)/len(set_reward))
    set_reward = []
    with open("C:/Users/EE720A/Desktop/final/model/log/aver.txt", "a") as file:
        file.write("Model:"+str((modtest+1)*5000)+" aver:" +str(ave) + "\n")   
    

env.close()
#%%
import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

env = gym.make('ALE/SpaceInvaders-v5', render_mode='human')
# 載入模型
model = load_model('C:/Users/EE720A/Desktop/final/model/model0608/s170000.h5')

# 使用載入的模型進行遊戲
num_episodes = 10  # 要玩幾個遊戲

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
        env.render()
        next_state = np.array(next_state)
        next_state = np.reshape(next_state, (1,) + next_state.shape)
        next_state = tf.convert_to_tensor(next_state, tf.uint8)
        total_reward += reward

        state = next_state

        if done:
            break

    print(f"Episode: {episode + 1}, Reward: {total_reward}")

env.close()
