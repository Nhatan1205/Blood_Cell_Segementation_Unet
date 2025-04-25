import torch

# Clear the GPU memory cache
torch.cuda.empty_cache()

# Check memory status
allocated_memory = torch.cuda.memory_allocated()  # Memory currently allocated by tensors
reserved_memory = torch.cuda.memory_reserved()    # Memory reserved by the caching allocator
free_memory = torch.cuda.get_device_properties(0).total_memory - reserved_memory  # Free memory

# Print the memory information
print(f"Allocated Memory: {allocated_memory / (1024 ** 2):.2f} MB")
print(f"Reserved Memory: {reserved_memory / (1024 ** 2):.2f} MB")
print(f"Free Memory: {free_memory / (1024 ** 2):.2f} MB")
