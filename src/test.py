import torch

# Check the number of available GPUs
num_gpus = torch.cuda.device_count()
print("Number of GPUs available:", num_gpus)

# Specify which GPU to use
device0 = torch.device("cuda:0")  # First GPU
device1 = torch.device("cuda:1")  # Second GPU

# Example code to run on multiple GPUs
tensor_gpu0 = torch.randn(3, 3, device=device0)
tensor_gpu1 = torch.randn(3, 3, device=device1)

# Perform operations on tensors on different GPUs
result_gpu0 = tensor_gpu0 * 2
result_gpu1 = tensor_gpu1 + 3

# Move tensors back to CPU if needed
result_gpu0_cpu = result_gpu0.cpu()
result_gpu1_cpu = result_gpu1.cpu()
