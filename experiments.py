import sys
import os
import torch
import torchvision
import numpy as np
import pandas as pd
import sklearn
import matplotlib
import platform

print("=== Python ===")
print(f"Version: {sys.version}")
print(f"Path:    {sys.executable}")
print(f"Platform: {platform.platform()}")

print("\n=== PyTorch ===")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

print("\n=== Libraries ===")
print(f"torchvision: {torchvision.__version__}")
print(f"numpy:       {np.__version__}")
print(f"pandas:      {pd.__version__}")
print(f"sklearn:     {sklearn.__version__}")
print(f"matplotlib:  {matplotlib.__version__}")

print("\n=== Working Directory ===")
print(os.getcwd())
