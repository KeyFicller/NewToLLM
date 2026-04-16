import torch
import torch.nn.functional as F
from torch.autograd import grad

class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        self.layers = torch.nn.Sequential(
            # First hidden layer
            torch.nn.Linear(num_inputs, 30),
            torch.nn.ReLU(),
            # Second hidden layer
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),
            # Output layer
            torch.nn.Linear(20, num_outputs),
        )

    def forward(self, x):
        logits = self.layers(x)
        return logits

def brief_torch():
    # Scalar/Vector/Matrix/Tensor
    tensor0d = torch.tensor(1)
    print("tensor0d: ", tensor0d)
    tensor1d = torch.tensor([1, 2, 3])
    print("tensor1d: ", tensor1d)
    tensor2d = torch.tensor([[1, 2, 3], [4, 5, 6]])
    print("tensor2d: ", tensor2d)
    tensor3d = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    print("tensor3d: ", tensor3d)

    # Tensor data type
    tensorint = torch.tensor([1, 2, 3])
    print("tensorint.dtype: ", tensorint.dtype)
    tensorfloat = torch.tensor([1.0, 2.0, 3.0])
    print("tensorfloat.dtype: ", tensorfloat.dtype)
    tensorfloat64 = tensorfloat.to(torch.float64)
    print("tensorfloat64.dtype: ", tensorfloat64.dtype)

    # Basic operations
    print("tensor2d: ", tensor2d)
    print("tensor2d.reshape: ", tensor2d.reshape(tensor2d.shape[1], tensor2d.shape[0]))
    print("tensor2d.view: ", tensor2d.view(tensor2d.shape[1], tensor2d.shape[0]))
    print("tensor2d.T: ", tensor2d.T)

    print("tensor2d.matmul(tensor2d.T): ", tensor2d.matmul(tensor2d.T))
    print("tensor2d @ tensor2d.T: ", tensor2d @ tensor2d.T)

    # Auto grad
    a = torch.tensor([1.0])
    b = torch.tensor([2.0])
    x = torch.tensor([3.0], requires_grad=True)

    y = a * x + b
    print("y = a * x + b, a = 1, grad(y, x) ", grad(y, x, retain_graph=True))

    y.backward()
    print("x.grad: ", x.grad)

    # Neural network
    torch.manual_seed(1234)
    model = NeuralNetwork(50, 3)
    print("neural network: ", model)
    num_params = sum(p.numel() for p in model.parameters())
    print("number of parameters: ", num_params)
    print("weights for layer[0]: ", model.layers[0].weight)
    print("shape of weights for layer[0]: ", model.layers[0].weight.shape)

    X = torch.rand((1, 50))
    print("Neural network input: ", X)
    out = model(X)
    print("Neural network output with grad: ", out)
    with torch.no_grad():
        out = torch.softmax(model(X), dim=1)
    print("Neural network output without grad: ", out)