import numpy as np
import torch ## The function of torch is the same as numpy, but it can run on GPU

## first part : convert data type
np_data = np.arange(6).reshape((2, 3)) 
torch_data = torch.from_numpy(np_data)
tensor2array = torch_data.numpy()
print(np_data,torch_data,tensor2array)
#
data = [-1,-2,1,-2]
tensor = torch.FloatTensor(data)
print("\nabs",
      "\nnumpy:",np.abs(data),
      "\ntorch:",torch.abs(tensor))
print("\nsin",
      "\nnumpy:",np.sin(data),
      "\ntorch:",torch.sin(tensor))
print("\nmean",
      "\nnumpy:",np.mean(data),
      "\ntorch:",torch.mean(tensor))
#
data = np.arange(4).reshape((2,2))
tensor = torch.FloatTensor(data)
print("\nmatrix multiplication",
      "\nnumpy:",np.matmul(data),
      "\ntensor:",torch.mm(data))

## second part : some basic usage