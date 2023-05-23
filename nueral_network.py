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
        self.optimizer=Optimizer_Adam(decay=5e-7)

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

        self.optimizer.pre_update_params()
        self.optimizer.update_params(self.dense1, dW1, db1)
        self.optimizer.update_params(self.dense2, dW2, db2)
        self.optimizer.post_update_params()
        #self.dense2.weights -= learning_rate * dW2
        #self.dense2.biases -= learning_rate * db2
        #self.dense1.weights -= learning_rate * dW1
        #self.dense1.biases -= learning_rate * db1
     
        loss= self.mse.mean_squared_error(final_output, y_true)
        return loss

class Optimizer_Adam:

    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7,
    beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2
   
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
        (1. / (1. + self.decay * self.iterations))


    def update_params(self, layer, dw, db):
    # If layer does not contain cache arrays, create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update momentum with current gradients

        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * dw
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * db

        # Get corrected momentum self.iteration is 0 at first pass and we need to start with 1 here
        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))

        # Update cache with squared current gradients
        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * dw**2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * db**2
             
        # Get corrected cache
        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))

        # Vanilla SGD parameter update + normalization with square rooted cache
        layer.weights += -self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) +self.epsilon)
        layer.biases += -self.current_learning_rate * \
        bias_momentums_corrected / \
        (np.sqrt(bias_cache_corrected) +
        self.epsilon)

        
         
    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1


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
        #max_pointer = min(self.pointer, self.buffer_size)
        
        indices = np.random.choice(self.buffer_size-1, batch_size, replace=False)
       
        states = self.states[indices]
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        dones=self.dones[indices]
        next_states = self.next_states[indices]
        #dones = self.dones[indices]
        
       
        return states, actions, rewards, dones, next_states
    