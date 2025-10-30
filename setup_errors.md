# 🐞 Known Issues — Capture & Share Animation (ComfyUI + RunPod )

This document lists all **known issues**, **root causes**, and **recommended fixes** encountered while setting up and training the animation capture workflow using **Stable Diffusion**, **LoRA fine-tuning**, and **ComfyUI** in **RunPod** or **Colab**.

---

## Common Issues That Occured During the SetUp and Execution.

### Module Not Found Errors
**Error:**
```bash
ModuleNotFoundError: No Module Named '<modulename>' 
```
**Fix:**
```bash
pip install <modulename>
```

### Red Box Around the Custom Node in ComfyUI
**Error:**
```bash
A red box appears around a node indicates that there is an error or issue with that specific node in the workflow.
```
**Fix:**
```
This typically means that a required component, such as a model, custom node, or file, is missing or not properly loaded.
Install the missing components or place them in the proper folder where they are intended to be
```
### Red nodes in your ComfyUI workflow while using WAN 2.2 Animate.

**Error:**
```
Wan Animate Process has a Red Colored Boundary while execution of the ComfyUI
```
**Fix:**
```
 It indicates that some custom nodes are missing.
 To resolve this issue, open the Manager tab in ComfyUI, click "Install Missing Custom Nodes," and then restart ComfyUI to apply the changes. This process ensures that all necessary components, including WAN 2.2 Animate, LightX2V LoRA, Relight LoRA, and detection nodes, are properly installed and ready for use. After restarting, the red nodes should disappear, and the workflow should load without errors.
```

### Value not in list Error:
**Error:**
```
OnnxDetectionModelLoader :
 - Value not in list: vitpose_model: 'vitpose-l-wholebody.onnx' not in [] - Value not in list: yolo_model: 'yolov10m.onnx' not in [] 
WanVideoTextEncodeCached : 
 - Value not in list: model_name: 'umt5-xxl-enc-bf16.safetensors' not in [] 
WanVideoVAELoader : 
 - Value not in list: model_name: 'wan_2.1_vae.safetensors' not in ['Wan2_1_VAE_bf16.safetensors'] 
CLIPVisionLoader 71: 
 - Value not in list: clip_name: 'clip_vision_h.safetensors' not in [] 
WanVideoModelLoader 22: 
 - Value not in list: model: 'Wan2_2-Animate-14B_fp8_e4m3fn_scaled_KJ.safetensors' not in [] WanVideoLoraSelectMulti 171: 
 - Value not in list: lora_1: 'lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors' not in ['none'] - Value not in list: lora_0: 'WanAnimate_relight_lora_fp16.safetensors' not in ['none']
```
**Fix:**
```bash
cd /workspace/ComfyUI/custom_nodes
git clone https://github.com/kijai/ComfyUI-WanVideoWrapper.git
git clone https://github.com/kijai/ComfyUI-WanAnimatePreprocess.git
git clone https://github.com/kijai/ComfyUI-KJNodes.git
git clone https://github.com/kijai/ComfyUI-VideoHelperSuite.git
git clone https://github.com/kijai/ComfyUI-segment-anything-2.git

```
```bash
%cd /workspace/ComfyUI

# 🌀 WAN22 Animate Model
!aria2c -x 16 -s 16 -d models/diffusion_models -o Wan2_2-Animate-14B_fp8_e4m3fn_scaled_KJ.safetensors \
  "https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/resolve/main/Wan22Animate/Wan2_2-Animate-14B_fp8_e4m3fn_scaled_KJ.safetensors"

# 🌀 LoRAs
aria2c -x 16 -s 16 -d models/loras -o lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors \
  "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors"

aria2c -x 16 -s 16 -d models/loras -o WanAnimate_relight_lora_fp16.safetensors \
  "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/LoRAs/Wan22_relight/WanAnimate_relight_lora_fp16.safetensors"

# 🌀 CLIP Vision
aria2c -x 16 -s 16 -d models/clip_vision -o clip_vision_h.safetensors \
  "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors"

# 🌀 VAE
aria2c -x 16 -s 16 -d models/vae -o wan_2.1_vae.safetensors \
  "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors"

# 🌀 Text Encoder
aria2c -x 16 -s 16 -d models/text_encoders -o umt5-xxl-enc-bf16.safetensors \
  "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/umt5-xxl-enc-bf16.safetensors"

# 🌀 Detection Models
aria2c -x 16 -s 16 -d models/detection -o yolov10m.onnx \
  "https://huggingface.co/Wan-AI/Wan2.2-Animate-14B/resolve/main/process_checkpoint/det/yolov10m.onnx"

aria2c -x 16 -s 16 -d models/detection -o vitpose-l-wholebody.onnx \
  "https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/onnx/wholebody/vitpose-l-wholebody.onnx"

```
All this process can be ignored if you have installed it correctly in the beginning.

### Disk Quota Exceeded:
**Error:**
```bash
Exception: [AbstractDiskWriter.cc:459] errNum=122 errorCode=17 Failed to write into the file
cause: Disk quota exceeded
```
**Fix:**
```
Increase the disk size on Run Pod. Edit Pod -> Incerease the Volume and Container Disk.
```
**Fix:**
```bash
rm -rf models/diffusion_models/*.aria2 models/diffusion_models/*.2.safetensors
rm -rf models/detection/*.aria2
```

### While Running ComfyUI only CPU is used, GPU is unused:
**Error:**
```
GPU is not utilized for running the models and the workflow executes slowly because of only utilizing CPU leaving the GPU resources unused.
```
**Fix:**

Verify if pytorch is able to see your GPU
```bash
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
Expected Output
```bash
True RTX 4090
```
If it prints False or errors, it means:
- PyTorch was not installed with CUDA.
- Or your NVIDIA drivers aren’t visible inside Docker.

**Fix:**

Run CUDA compatible pytorch
```bash
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Fix:**

Explicitly run on your GPU (CUDA) and use FP16 precision for faster inference`
```bash
python3 main.py --force-fp16 --cuda-device 0 --listen 0.0.0.0 --port 8188
```
```
--force-fp16
    Forces 16-bit floating point (half precision) computation.
    Greatly reduces VRAM usage.
    Doubles speed on RTX GPUs (like your 4090).
    Slightly reduces precision — but for image/video generation, this is perfectly fine.
```

**Fix:**

To Force Comfy to always use GPU.

Open the file:
```
vi ComfyUI/comfy/model_management.py
```
Search for the line containing
```python
def get_torch_device():
```
You’ll see something like this (default code):
```python
def get_torch_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

```
Modify it to force CUDA(GPU):
```python
def get_torch_device():
    if torch.cuda.is_available():
        print("[INFO] Forcing CUDA (GPU) for all ComfyUI operations.")
        return torch.device("cuda")
    else:
        print("[WARNING] CUDA not available. Falling back to CPU.")
        return torch.device("cpu")
```
Restart ComfyUI with:
```bash
python3 main.py --listen 0.0.0.0 --port 8188
```

### The Video Upload Button Doesnot Show Up in the VideoHelperSuite
The “Load Video” node appears, but there’s no upload button (no file picker UI in the node).
This usually happens because the **web frontend of ComfyUI isn’t loading the JavaScript hooks** that handle video uploads.

**Error:**
```
No Upload Button in the in the "Load Video" of Video Helper Suite
```
**Fix:**
```bash
cd /workspace/ComfyUI/custom_nodes
rm -rf ComfyUI-VideoHelperSuite
git clone https://github.com/kijai/ComfyUI-VideoHelperSuite.git
cd ComfyUI-VideoHelperSuite
git checkout main
git pull
```
💡 The latest commit (after Aug 2024) adds video_upload.js which defines the upload button.

Clear any broken frontend cache
```bash
cd /workspace/ComfyUI
rm -rf web/extensions/*
rm -rf web/tmp/*

```
Then restart ComfyUI — it will regenerate the frontend build automatically.

Hard Refresh Your Browser
```
Press Ctrl + F5 (Windows/Linux) or Cmd + Shift + R (Mac)
```

If still missing then run the diagnostic command:
```
grep -R "def load_video" -n custom_nodes/ComfyUI-VideoHelperSuite
```
If the video_upload.js file in that repo, your clone might be old or broken. Manually fetch the fixed version:
```
cd /workspace/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite
git fetch origin main
git reset --hard origin/main
```
Then restart ComfyUI again.

### Integer Modulo By Zero Error
Excellent — this error is a known issue with VHS_LoadVideo (VideoHelperSuite).
Let’s break it down properly

**Error:**
```
VHS_LoadVideo: integer modulo by zero
```
This error means that the FPS (frames per second) value or frame count read from your input video is zero or invalid, and later in the node’s processing, a modulo (%) operation is attempted with it.

**Fix:**

In the VHS_LoadVideo node:
```
Make sure “Select every nth frame” is ≥ 1 (default is often 0 → bug).
Set it to 1 to process all frames.
Set it to 2 to take every 2nd frame, etc.
```