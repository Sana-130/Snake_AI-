import numpy as np

class Layer_Dense:

    def __init__(self,  n_inputs, n_nuerons):
        self.n_nueron=n_nuerons
        self.n_inputs=n_inputs
        self.weights=0.1 * np.random.randn(n_inputs, n_nuerons)
        self.biases=np.zeros((1, n_nuerons))

    def forward(self, inputs):
        self.output= np.dot(inputs, self.weights) + self.biases


class Activation_ReLU:
    
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
        return self.output

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

class Activation_Softmax:
    def forward(self, inputs):

        exp_values = np.exp(inputs - np.max(inputs, axis=1,
keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1,
keepdims=True)
        self.output = probabilities
        return self.output

    def backward(self, logits, targets):
    
        samples = len(logits)
        if len(targets.shape) == 2:
            y_true = np.argmax(targets, axis=1)
        self.dinputs = logits.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples
        return self.dinputs

class Target_network():
    def __init__(self, state_size, nuerons, output_size):
        self.dense1= Layer_Dense(state_size, nuerons)
        self.activation1= Activation_ReLU()
        self.dense2 = Layer_Dense(nuerons, output_size)
        self.activation2 = Activation_Softmax()

    def forward(self, state):
        self.dense1.forward(state)
        self.activation1.forward(self.dense1.output)
        self.dense2.forward(self.activation1.output)
        self.activation2.forward(self.dense2.output)

        return self.activation2.output
    
class Mse_Loss:
    def mean_squared_error(y_pred, y_true):
        return 0.5 * np.mean((y_pred - y_true) ** 2)

    def backward(y_pred, y_true):
        return 2 * (y_pred - y_true) / y_true.shape[0]
    
class DQN_network():
    def __init__(self, state_size, nuerons, output_size):
        self.dense1 = Layer_Dense(state_size, nuerons)
        self.activation1 = Activation_ReLU()
        self.dense2 = Layer_Dense(nuerons, output_size)
        self.activation2 = Activation_Softmax()
        self.mse=Mse_Loss

    def forward(self, state):
        self.dense1.forward(state)
        self.activation1.forward(self.dense1.output)
        self.dense2.forward(self.activation1.output)
        self.activation2.forward(self.dense2.output)

        return self.activation2.output
    
    def backprop(self, X, y_true, learning_rate):
        global epsilon
      
        relu_output=self.activation1.output
        final_output=self.activation2.output

     
        delta3 = self.mse.backward(final_output, y_true)

        dW2 = np.dot(relu_output.T, delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        
    
        dh = np.dot(delta3, self.dense2.weights.T)

        dh[relu_output <= 0] = 0
        dh *= (relu_output > 0)
    
        dW1 = np.dot(X.T, dh)
        db1 = np.sum(dh, axis=0, keepdims=True)
    
        self.dense2.weights -= learning_rate * dW2
        self.dense2.biases -= learning_rate * db2
        self.dense1.weights -= learning_rate * dW1
        self.dense1.biases -= learning_rate * db1
     
        loss= self.mse.mean_squared_error(final_output, y_true)

#DQN Memory   

class ReplayBuffer:
    def __init__(self, buffer_size, state_shape, action_shape):
        self.buffer_size = buffer_size
        self.state_shape = state_shape
        self.action_shape = action_shape
        
        # Create empty numpy arrays to store the memory
        self.states = np.zeros((buffer_size, *state_shape), dtype=np.float32)
        self.actions = np.zeros((buffer_size, *action_shape), dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.next_states = np.zeros((buffer_size, *state_shape), dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        
        # Initialize the pointer to the current position in the buffer
        self.pointer = 0
        
    def add(self, state, action, reward, done, next_state):
        # Store the transition in the memory buffer
        self.states[self.pointer] = state
        self.actions[self.pointer] = action
        self.rewards[self.pointer] = reward
        self.dones[self.pointer]= done
        self.next_states[self.pointer] = next_state
        #self.dones[self.pointer] = done
        
        # Update the pointer
        self.pointer = (self.pointer + 1) % self.buffer_size
        
    def sample(self, batch_size):
        # Randomly sample a batch of transitions from the memory buffer
        max_pointer = min(self.pointer, self.buffer_size)
        indices = np.random.choice(max_pointer, batch_size, replace=False)
       
        states = self.states[indices]
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        dones=self.dones[indices]
        next_states = self.next_states[indices]
        #dones = self.dones[indices]
        
       
        return states, actions, rewards, dones, next_states
    