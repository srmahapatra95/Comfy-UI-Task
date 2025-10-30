# 🧰 WAN 2.2 Animate — ComfyUI Implementation (RunPod GPU Setup)

Recreate a complete video-to-video animation workflow using **Wan 2.2 Animate** inside **ComfyUI**, deployed on **RunPod** (A100 / RTX 4090 GPU).

---

## 📦 1️⃣ Prerequisites

### 🔧 System Requirements
| Component | Recommended |
|------------|--------------|
| **GPU** | NVIDIA RTX 4090 / A100 (24 GB+ VRAM) |
| **CUDA** | ≥ 12.1 |
| **Python** | ≥ 3.10 |
| **Torch** | ≥ 2.1 |
| **OS** | Ubuntu 22.04 (RunPod Base Image Recommended) |
| Runtime | RunPod Cloud / Local Ubuntu |



## ⚙️ Hardware Configuration & Requirements

---

### 🧠 RunPod Instance Configuration

| **Resource**     | **Recommended**                                | **Notes** |
|------------------|------------------------------------------------|------------|
| **GPU**          | NVIDIA A6000 / A100 / RTX 4090 (≥ 24 GB VRAM) | Required for Stable Diffusion / LoRA training and image generation. |
| **CPU**          | 8–16 vCPUs                                   | Useful for data preprocessing and I/O-heavy tasks. |
| **Memory (RAM)** | 32–64 GB                                     | Training pipelines and ComfyUI workflows benefit from higher memory. |
| **Storage**      | 100–200 GB (NVMe SSD)                        | For datasets, checkpoints, and generated outputs. |
| **OS**           | Ubuntu 22.04 (Recommended)                   | Works best with Python 3.10+ and CUDA 12.0+. |

---

### 🧩 Software Environment

| **Component**    | **Version**             | **Purpose** |
|------------------|------------------------|--------------|
| **Python**       | 3.10+                  | Base runtime for all modules. |
| **PyTorch**      | 2.0+ with CUDA 12.0    | GPU acceleration for diffusion models. |
| **Diffusers**    | ≥ 0.30.0               | Stable Diffusion & LoRA fine-tuning. |
| **Transformers** | ≥ 4.40                 | Tokenizer and text encoder support. |
| **xFormers**     | Latest                 | Memory-efficient attention mechanism. |
| **Accelerate**   | ≥ 0.28                 | Multi-GPU and mixed precision training. |
| **ComfyUI**      | Latest release         | Visual node-based diffusion workflow interface. |

---

### 🖼️ ComfyUI Requirements

| **Resource** | **Minimum** | **Recommended** | **Notes** |
|--------------|-------------|-----------------|-----------|
| **VRAM**     | 12 GB       | 24 GB+          | Larger VRAM allows high-res image generation and complex node graphs. |
| **Disk Space** | 50 GB     | 100 GB+         | For model weights (.safetensors, .ckpt) and workflow outputs. |
| **Python Version** | 3.10 | 3.10+           | Tested stable with PyTorch 2.0+. |
| **Dependencies** | See `requirements.txt` | — | Includes torch, xformers, tqdm, numpy, opencv-python, gradio, etc. |
---

### 🕒 Runtime and Costs
**Pod Selected on RunPod:**
| RTX 4090 | $0.59/hr |
|------|------|
|<small>24GB VRAM</small> | <center><small>7 max</small></center>|
|<small>36GB RAM . 6 vCPUs</small> | <center><small>High</small>|

**GPU Used:**
| RTX 4090 x1 | $0.61/hr  |
|------|------|
|<small>AMD EPYC 75F3 32-Core Processor</small> ||
|<small>62GB RAM . 16 vCPUs</small> ||
|<small>Volume Disk</small> | <small>80.0 GB</small>||
|<small>Container Disk</small> | <small>70.0 GB</small>||



### 🧩 Preinstalled Tools
Ensure these are installed:
```bash
sudo apt update && sudo apt install -y git python3 python3-venv python3-pip ffmpeg aria2 wget curl
```
### 📦 Clone & Configure ComfyUI
```bash
mkdir workspace
cd /workspace || cd
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI
```

###🐍 5️⃣ Create Virtual Environment & Install Dependencies
```bash
python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```
### 🧩 6️⃣ Install Custom Nodes (WAN 2.2 Dependencies)
Run this inside a bash script to install WAN 2.2 Dependencies
```bash
cd custom_nodes
for repo in \
  "https://github.com/kijai/ComfyUI-WanVideoWrapper.git" \
  "https://github.com/kijai/ComfyUI-WanAnimatePreprocess.git" \
  "https://github.com/kijai/ComfyUI-segment-anything-2.git" \
  "https://github.com/kijai/ComfyUI-KJNodes.git" \
  "https://github.com/kijai/ComfyUI-VideoHelperSuite.git"
do
  name=$(basename "$repo" .git)
  if [ ! -d "$name" ]; then
    git clone "$repo"
  fi
done
cd ..
```
### 🗂️ 7️⃣ Create Model Folders
```bash
mkdir -p models/diffusion_models \
         models/loras \
         models/vae \
         models/text_encoders \
         models/clip_vision \
         models/detection
```
### ⬇️ 8️⃣ Download All Required Models
Use aria2c for high-speed parallel downloads (auto-resumes if interrupted):
```bash
aria2c --file-allocation=none -x 16 -s 16 -d models/diffusion_models -o Wan2_2-Animate-14B_fp8_e4m3fn_scaled_KJ.safetensors \
  "https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/resolve/main/Wan22Animate/Wan2_2-Animate-14B_fp8_e4m3fn_scaled_KJ.safetensors"

aria2c --file-allocation=none -x 16 -s 16 -d models/loras -o lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors \
  "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors"

aria2c --file-allocation=none -x 16 -s 16 -d models/loras -o WanAnimate_relight_lora_fp16.safetensors \
  "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/LoRAs/Wan22_relight/WanAnimate_relight_lora_fp16.safetensors"

aria2c --file-allocation=none -x 16 -s 16 -d models/clip_vision -o clip_vision_h.safetensors \
  "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors"

aria2c --file-allocation=none -x 16 -s 16 -d models/vae -o wan_2.1_vae.safetensors \
  "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors"

aria2c --file-allocation=none -x 16 -s 16 -d models/text_encoders -o umt5-xxl-enc-bf16.safetensors \
  "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/umt5-xxl-enc-bf16.safetensors"

aria2c --file-allocation=none -x 16 -s 16 -d models/detection -o yolov10m.onnx \
  "https://huggingface.co/Wan-AI/Wan2.2-Animate-14B/resolve/main/process_checkpoint/det/yolov10m.onnx"

aria2c --file-allocation=none -x 16 -s 16 -d models/detection -o vitpose-l-wholebody.onnx \
  "https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/onnx/wholebody/vitpose-l-wholebody.onnx"
  ```

### Install other dependencies that might break the launch of ComfyUI
If you get ModuleNotFound error for certain modules:
```bash
pip install <modulename>
```

🎮 9️⃣ Launch ComfyUI (GPU Mode)
Start ComfyUI with GPU acceleration and FP16 enabled:
```bash
cd /workspace/ComfyUI
python3 main.py --force-fp16 --cuda-device 0 --listen 0.0.0.0 --port 8188
```
Open the Web Interface:
```bash
http://0.0.0.0:8188
```
For Launching from the RunPod:
1. Go to the your deployed Pods and click on the same.
2. In Http Services, launch Port 8188 -> Http Service.(If the port is not enabled, right click on the _More Actions_ -> _Edit Pod_ -> _Expose HTTP Ports_ add 8188)


### 🎬 9️⃣ Load Workflow

1. Drag wan22_animate.json into ComfyUI.

2. Ensure all nodes initialize successfully (no red nodes).

3. Load your input image (character) and input video (reference motion).

4. Click Run to start animation generation.


### 🧹 2️⃣ Cleanup (Optional)

To free GPU memory after crash:
```bash
pkill -9 python
rm -rf /workspace/ComfyUI/output/* /workspace/ComfyUI/temp/*
python3 - <<'EOF'
import torch, gc
gc.collect(); torch.cuda.empty_cache()
print("✅ GPU memory released")

# Release GPU cache
nvidia-smi --gpu-reset -i 0
```
By default, ComfyUI stores a lot of intermediate junk in:

Run:
```bash
rm -rf /workspace/ComfyUI/output/*
rm -rf /workspace/ComfyUI/temp/*
rm -rf /workspace/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite/temp/*
```

Optional system level clean up:
```bash
apt-get clean
rm -rf /var/lib/apt/lists/*
rm -rf /tmp/*
```
Restart the Comfy UI from the terminal again.

