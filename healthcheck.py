import os
import platform
import torch

def get_system_info():
    print("=== System Information ===")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"OS Version: {platform.version()}")
    print(f"Machine: {platform.machine()}")
    print(f"Processor: {platform.processor()}")
    print(f"Python Version: {platform.python_version()}")
    print(f"Hostname: {platform.node()}")
    print(f"Number of CPUs: {os.cpu_count()}")

def get_gpu_info():
    if torch.cuda.is_available():
        print("\n=== GPU Information ===")
        gpu_count = torch.cuda.device_count()
        print(f"Number of GPUs: {gpu_count}")
        for i in range(gpu_count):
            print(f"--- GPU {i} ---")
            print(f"Name: {torch.cuda.get_device_name(i)}")
            print(f"Compute Capability: {torch.cuda.get_device_capability(i)}")
            print(f"Memory Allocated: {torch.cuda.memory_allocated(i) / 1e9:.2f} GB")
            print(f"Memory Reserved: {torch.cuda.memory_reserved(i) / 1e9:.2f} GB")
            print(f"Total Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
    else:
        print("\n=== GPU Information ===")
        print("No GPU found or CUDA not available.")

def get_cuda_info():
    print("\n=== CUDA Information ===")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"cuDNN Version: {torch.backends.cudnn.version()}")
    print(f"cuDNN Enabled: {torch.backends.cudnn.enabled}")

def get_environment_info():
    print("\n=== Environment Variables ===")
    for key, value in os.environ.items():
        if key.startswith("CUDA") or key.startswith("NVIDIA") or key.startswith("TORCH"):
            print(f"{key}: {value}")

if __name__ == "__main__":
    get_system_info()
    get_gpu_info()
    get_cuda_info()
    get_environment_info()
