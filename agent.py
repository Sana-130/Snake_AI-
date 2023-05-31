import numpy as np
from game import Game
from nueral_network import DQN_network, Target_network, ReplayBuffer
import pickle

#global variables


class Agent():
    def __init__(self):
        self.epsilon = 1.0 # Initial exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.9#0.99999
        self.target_freq = 5
        self.gamma = 0.99
        self.ln_rate=0.001
        self.n_state_size=16
        self.n_nuerons=128
        self.output_layer=4
        self.m_buffer_size=1000
        self.m_batch_size=128
        self.m_max_pointer=300
        self.q_network=DQN_network(self.n_state_size, self.n_nuerons, self.output_layer)
        self.target_network=Target_network(self.n_state_size, self.n_nuerons, self.output_layer)
        self.game=Game()
        self.memory=ReplayBuffer(self.m_buffer_size, [1, self.n_state_size], [1,1])
        self.n_episodes=1000
        self.pre_train_episodes=999
        self.pretrain()
        self.max_step=3000
        self.decay_step=0
        self.training=True
        self.maxTau=100
        self.tau=0
        self.new_episode=False
        self.step_no=0
        self.episode_no=0
        self.training_step = 0
        self.state=None

    def exploration_exp(self, output):
    
        if np.random.uniform() < self.epsilon:
            action= np.random.randint(0,4)
        else:
            #print(output)    
            action = self.choose_action(output)

        return action
    
    def train_short_memory(self, state , q_value, action , reward, next_state):  
        state= np.array(state).reshape(1, self.n_state_size)
        next_states = np.array(next_state).reshape(1, self.n_state_size)

        next_qvalues=self.target_network.forward(next_states)
        target_qvalues = q_value.copy()
        print(target_qvalues[0][action], reward ,self.gamma , next_qvalues)
        target_qvalues[0][action]= np.array(reward) + self.gamma * np.array(next_qvalues)

        #self.q_network.backprop(np.array(state), target_qvalues, self.ln_rate)
        pass

    #def train_long_memory(self):
    #    pass
        
    
    def choose_action(self, output):
        return np.argmax(output)


    def copy(self):
        self.target_network.dense1.weights=self.q_network.dense1.weights
        self.target_network.dense2.weights=self.q_network.dense2.weights
        self.target_network.dense1.biases=self.q_network.dense1.biases
        self.target_network.dense2.biases=self.q_network.dense2.biases

    def store(self, states, action, reward, done, next_state):
        self.memory.add(states, action, reward, done, next_state)

    def sample_and_backprop(self):
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
        loss=self.q_network.backpropagation( target_qvalues)
        return loss

                            
    def save_model(self):
        file = open('save', 'wb')
        pickle.dump(self.q_network, file)
        file.close()
        

    def load_model(self):
        with open('save', 'rb') as file:
            self.q_network=pickle.load(file)

    def set_model(self):
        with open('TheDumbSnake\models\model_5', 'rb') as file:
            data=pickle.load(file)
            self.q_network=data
            self.target_network=data

    def pretrain(self):
        self.game.build()
        for i in range(self.pre_train_episodes):
            if i==0:
                state=self.game.get_state()

            action= np.random.randint(0,4) #self.exploration_exp(output)
            next_state ,  reward, done = self.game.execute(action)
            if done:
                self.game.reset()
                state=self.game.get_state()
            else:
                state=next_state

            self.store(state, action, reward, done, next_state)
            #print(i , state, action, reward, done, next_state)

        print("pretraining done")

    def train(self):
        self.set_model()

        while self.training:
            if self.new_episode:
                self.state=self.game.get_state()

            #if self.step_no < self.max_step:
            #    self.step_no+=1

            #if self.step_no < self.max_step:
            #    self.step_no+=1
            #    self.decay_step+=1
            #    self.training_step+=1

            self.tau+=1

            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            if np.random.uniform() < self.epsilon:
                action= np.random.randint(0,4)
            else:
                    #print(output)  
                output=self.q_network.forward(self.state)
                action = self.choose_action(output)

            next_state ,  reward, done = self.game.execute(action)

                
            #if done:
            #    print("DEAD")
            #else:
            #    print("Reward bro", reward)

            self.store(self.state, action, reward, done, next_state)
            self.state=next_state

            loss=self.sample_and_backprop()
            print("loss", loss, "reward", reward, "episode no", self.episode_no, "ep", self.epsilon)

            if done:
                self.episode_no+=1
                self.new_episode=True
                self.game.reset()
                if self.episode_no > self.n_episodes:
                    self.training=False
                if self.episode_no % 100 ==0:
                    self.save_model()
                    print("model saved")
                if self.episode_no % 5 ==0:
                    self.copy()
                    self.tau=0
                    print("target net updated")

                  
    def test(self):
        self.load_model()
        
        for _ in range(10000): 
            self.game.draw()
            state=self.game.get_state()
            output=self.q_network.forward(state)
            action = self.choose_action(output)
            next_state ,  reward, done =self.game.execute(action)
           
            if done:
                self.game.reset()

            self.game.set_fps()
            self.game.update()
        

    def run(self):

        #for _ in range(5):
        #self.train()
        self.test()
        

if __name__ == "__main__":  
    agent=Agent()
    #agent.train()
    agent.run()