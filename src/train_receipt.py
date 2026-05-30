import torch
print("="*50)
print(f"PyTorch Version: {torch.__version__}")
print(f"Is CUDA (GPU) Available? : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Hardware Name: {torch.cuda.get_device_name(0)}")
print("="*50)