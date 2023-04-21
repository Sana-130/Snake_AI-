import numpy as np
from game import Game
from nueral_network import DQN_network, Target_network, ReplayBuffer
import pickle

#global variables


class Agent():
    def __init__(self):
        self.epsilon = 1.0 # Initial exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.99
        self.target_freq = 5
        self.gamma = 0.99
        self.ln_rate=0.001
        self.n_state_size=16
        self.n_nuerons=32
        self.output_layer=4
        self.m_buffer_size=600
        self.m_batch_size=64
        self.m_max_pointer=74
        self.q_network=DQN_network(self.n_state_size, self.n_nuerons, self.output_layer)
        self.target_network=Target_network(self.n_state_size, self.n_nuerons, self.output_layer)
        self.game=Game()
        self.memory=ReplayBuffer(self.m_buffer_size, [1, self.n_state_size], [1,1])
        self.n_episodes=1000

    def exploration_exp(self, output):
    
        if np.random.uniform() < self.epsilon:
            action= np.random.randint(0,4)
        else:
            #print(output)    
            action = self.choose_action(output)

        return action
    
    def choose_action(self, output):
        return np.argmax(output)


    def copy(self):
        self.target_network.dense1.weights=self.q_network.dense1.weights
        self.target_network.dense2.weights=self.q_network.dense2.weights
        self.target_network.dense1.biases=self.q_network.dense1.biases
        self.target_network.dense2.biases=self.q_network.dense2.biases

    def addtomemory(self, states, action, reward, done, next_state):

        self.memory.add(states, action, reward, done, next_state)
  
        if self.memory.pointer>=self.m_max_pointer:
            state, actions, rewards, dones, next_states = self.memory.sample(self.m_batch_size)

            #print(actions)   
            actions = np.array(actions).reshape(self.m_batch_size, 1)
            #print(actions)
            rewards = np.array(rewards).reshape(self.m_batch_size, 1)
            dones = np.array(dones).reshape(self.m_batch_size, 1)
            state= np.array(state).reshape(self.m_batch_size, self.n_state_size)
            next_states = np.array(next_states).reshape(self.m_batch_size, self.n_state_size)


            current_qvalues= self.q_network.forward(state)
            next_qvalues=self.target_network.forward(next_states)

            target_qvalues = current_qvalues.copy()
        
        
            for i in range(self.m_batch_size):
                action = int(actions[i])
                reward = rewards[i]
                done = dones[i]
                next_q = np.max(next_qvalues[i])

                #print(action, reward, done, next_q)
                if done:
                    target_qvalues[i, action] = reward
                else:
                    target_qvalues[i, action] = reward + self.gamma * next_q


            #print(target_qvalues[0])
            self.q_network.backprop(state, target_qvalues, self.ln_rate)

    def save_model(self):
        file = open('save', 'wb')
        pickle.dump(self.q_network, file)
        file.close()
    
    def train(self):
        done=False
        max_score=0
        self.game.build()

        for episode in range(self.n_episodes):
        
            ep_reward=0
            
            self.game.set_up()
            while done != True:
                
                self.game.draw()
              
                current_state = self.game.get_state()
                output = self.q_network.forward(current_state)
                
                action= self.exploration_exp(output)
                next_state ,  reward, done = self.game.execute(action)
                            
                self.addtomemory(current_state, action, reward, done, next_state)
               
                ep_reward+= reward
                
                if self.n_episodes % self.target_freq == 0:
                    self.copy()
                    
         
                self.game.set_fps()
                self.game.update()
              
            if self.game.total_score > max_score:
                max_score=self.game.total_score
                self.save_model()

            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            print(f"{episode} total reward {ep_reward} ------- score {self.game.total_score} epsilon {self.epsilon}")
            if done == True:
                self.game.reset()
                self.game.total_score=0
                done=False


            self.game.update()
    

if __name__ == "__main__":  
    agent=Agent()
    agent.train()