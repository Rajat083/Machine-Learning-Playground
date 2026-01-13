import numpy as np
import random


class MPNeuron:
    def __init__(self, d_model: int, threshold: int = 0) -> None:
        self.d_model = d_model 
        self.weights = np.array([[1] * d_model], dtype=np.int32)
        self.threshold = threshold
        
    def set_weights(self, new_weights: np.ndarray) -> None:
        arr = np.asarray(new_weights)
        if arr.ndim == 1:
            if arr.shape[0] != self.d_model:
                raise ValueError("Shape mismatch: expected weight length %d, got %d" % (self.d_model, arr.shape[0]))
            arr = arr.reshape(1, -1)
        elif arr.ndim == 2:
            if arr.shape[1] != self.d_model:
                raise ValueError("Shape mismatch: expected second dim %d, got %d" % (self.d_model, arr.shape[1]))
        else:
            raise ValueError("Weights must be a 1D or 2D array")

        self.weights = arr.astype(self.weights.dtype)
        
    def forward(self, x:np.ndarray):
        arr = np.asarray(x)
        if arr.ndim == 1:
            if arr.shape[0] != self.d_model:
                raise ValueError("Input length mismatch: expected %d, got %d" % (self.d_model, arr.shape[0]))
            arr = arr.reshape(1, -1)
        elif arr.ndim == 2:
            if arr.shape[1] != self.d_model:
                raise ValueError("Input second-dim mismatch: expected %d, got %d" % (self.d_model, arr.shape[1]))
        else:
            raise ValueError("Input must be a 1D or 2D array")

        return np.matmul(arr, self.weights.T)
    
    def get_result(self, x:np.ndarray):
        return [1 if val >= self.threshold else 0 for val in x.flatten()]
    
if __name__ == "__main__":
    ch = ''
    while ch != 'q':
        print("Enter dimension of model (d_model): ")
        d_model = int(input().strip())
        neuron = MPNeuron(d_model)
        print("Enter weights (space-separated): ")
        weights = list(map(int, input().strip().split()))
        neuron.set_weights(np.array(weights))
        print("Enter threshold: ")
        threshold = int(input().strip())
        neuron.threshold = threshold
        while ch != 'f' and ch != 'q':
            print("Enter input vector (space-separated): ")
            input_vector = list(map(int, input().strip().split()))
            output = neuron.forward(np.array(input_vector))
            print("Output:", output)
            result = neuron.get_result(output)
            print(f"Result after thresholding ({threshold}):", result)
            print("\nEnter 'f' to change model or 'q' to quit (Press Enter to continue the same model): ")
            ch = input().strip()
            
        