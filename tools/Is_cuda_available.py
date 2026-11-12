import os
import torch
from torch.utils.cpp_extension import load

# 打印环境信息
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))

module_path = os.path.dirname(__file__)
print("Module path:", module_path)

cpp_path = os.path.join(module_path, "fused_bias_act.cpp")
cuda_path = os.path.join(module_path, "fused_bias_act_kernel.cu")

print("CPP file exists:", os.path.exists(cpp_path))
print("CUDA file exists:", os.path.exists(cuda_path))

# 尝试编译
try:
    fused = load(
        "fused",
        sources=[cpp_path, cuda_path],
        verbose=True
    )
    print("Compilation successful!")
except Exception as e:
    print("Compilation failed with error:", e)
    import traceback
    traceback.print_exc()