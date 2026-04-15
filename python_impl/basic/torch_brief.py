import torch

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