import numpy as np

class GlassBoxNetwork:
    """
    A highly transparent Neural Network built entirely from scratch using NumPy.
    Supports Classification (Sigmoid/Cross-Entropy) and Regression (Linear/MSE).
    """
    def __init__(self, layer_sizes, learning_rate=0.01, activation='ReLU', regularization='None', lambda_reg=0.1, problem_type='Classification'):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.activation_choice = activation
        self.regularization = regularization
        self.lambda_reg = lambda_reg 
        self.problem_type = problem_type # NEW: Tracks Regression vs Classification
        
        self.parameters = {}
        self.cache = {}      
        
        for i in range(1, len(layer_sizes)):
            scale_factor = np.sqrt(1.0 / layer_sizes[i-1])
            self.parameters[f'W{i}'] = np.random.randn(layer_sizes[i], layer_sizes[i-1]) * scale_factor
            self.parameters[f'b{i}'] = np.zeros((layer_sizes[i], 1))

    # --- ACTIVATION FUNCTIONS ---
    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def sigmoid_derivative(self, A):
        return A * (1 - A)

    def relu(self, Z):
        return np.maximum(0, Z)
        
    def relu_derivative(self, Z):
        return (Z > 0).astype(float)

    def tanh(self, Z):
        return np.tanh(Z)
        
    def tanh_derivative(self, A):
        return 1 - np.power(A, 2)

    # --- CORE MECHANICS ---
    def forward_propagation(self, X):
        self.cache['A0'] = X
        A_prev = X
        L = len(self.layer_sizes) - 1 

        for i in range(1, L + 1):
            W = self.parameters[f'W{i}']
            b = self.parameters[f'b{i}']
            
            Z = np.dot(W, A_prev) + b
            self.cache[f'Z{i}'] = Z
            
            # NEW: Dynamic Final Layer
            if i == L:
                if self.problem_type == 'Regression':
                    A = Z # Linear Activation (No squishing)
                else:
                    A = self.sigmoid(Z) # Classification Probability
            else:
                if self.activation_choice == 'ReLU':
                    A = self.relu(Z)
                elif self.activation_choice == 'Tanh':
                    A = self.tanh(Z)
                else:
                    A = self.sigmoid(Z)
            
            self.cache[f'A{i}'] = A
            A_prev = A
            
        return A_prev 

    def compute_loss(self, Y_true, Y_pred):
        m = Y_true.shape[1] 
        
        # NEW: Dynamic Loss Function
        if self.problem_type == 'Regression':
            # Mean Squared Error
            base_loss = (1 / (2 * m)) * np.sum(np.square(Y_pred - Y_true))
        else:
            # Binary Cross-Entropy
            epsilon = 1e-15
            Y_pred = np.clip(Y_pred, epsilon, 1 - epsilon)
            base_loss = -(1/m) * np.sum(Y_true * np.log(Y_pred) + (1 - Y_true) * np.log(1 - Y_pred))
        
        # Regularization Penalty
        L = len(self.layer_sizes) - 1
        penalty = 0
        if self.regularization == 'L1':
            for i in range(1, L + 1):
                penalty += np.sum(np.abs(self.parameters[f'W{i}']))
            base_loss += (self.lambda_reg / m) * penalty
        elif self.regularization == 'L2':
            for i in range(1, L + 1):
                penalty += np.sum(np.square(self.parameters[f'W{i}']))
            base_loss += (self.lambda_reg / (2 * m)) * penalty
            
        return base_loss

    def backward_propagation(self, Y_true):
        gradients = {}
        L = len(self.layer_sizes) - 1
        m = Y_true.shape[1]
        
        Y_pred = self.cache[f'A{L}']
        
        # Mathematical Elegance: 
        # Derivative of MSE (Linear) AND Cross-Entropy (Sigmoid) equal the same thing!
        dZ = Y_pred - Y_true
        
        for i in reversed(range(1, L + 1)):
            A_prev = self.cache[f'A{i-1}']
            
            dW = (1 / m) * np.dot(dZ, A_prev.T)
            db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
            
            if self.regularization == 'L1':
                dW += (self.lambda_reg / m) * np.sign(self.parameters[f'W{i}'])
            elif self.regularization == 'L2':
                dW += (self.lambda_reg / m) * self.parameters[f'W{i}']
            
            gradients[f'dW{i}'] = dW
            gradients[f'db{i}'] = db
            
            if i > 1:
                W = self.parameters[f'W{i}']
                Z_prev_layer = self.cache[f'Z{i-1}']
                A_prev_layer = self.cache[f'A{i-1}']
                
                if self.activation_choice == 'ReLU':
                    dZ = np.dot(W.T, dZ) * self.relu_derivative(Z_prev_layer)
                elif self.activation_choice == 'Tanh':
                    dZ = np.dot(W.T, dZ) * self.tanh_derivative(A_prev_layer)
                else:
                    dZ = np.dot(W.T, dZ) * self.sigmoid_derivative(A_prev_layer)
                
        return gradients

    def update_parameters(self, gradients):
        L = len(self.layer_sizes) - 1
        for i in range(1, L + 1):
            self.parameters[f'W{i}'] -= self.learning_rate * gradients[f'dW{i}']
            self.parameters[f'b{i}'] -= self.learning_rate * gradients[f'db{i}']

    def train(self, X, Y_true, epochs=1000):
        loss_history = []
        for epoch in range(epochs):
            Y_pred = self.forward_propagation(X)
            loss = self.compute_loss(Y_true, Y_pred)
            loss_history.append(loss)
            
            gradients = self.backward_propagation(Y_true)
            self.update_parameters(gradients)
            
        return loss_history