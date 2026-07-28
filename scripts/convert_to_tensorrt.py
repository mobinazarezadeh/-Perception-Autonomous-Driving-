import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import onnx
import time

print("TensorRT version:", trt.__version__)

# Simple check
print("✅ TensorRT is ready to use!")

# Later we'll add full conversionpip