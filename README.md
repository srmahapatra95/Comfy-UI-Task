# Comfy UI Task 

## Video generation from a input video and an input image

The 2 video generation task include:
• Changing only the character while keeping the background constant.
• Changing both the character and the background.

## Screen Recoding Link for the Comfy UI Workflow Execution.

Screen recording link for the video replaced with both the character and the background.

[https://www.youtube.com/watch?v=GDAr4xFrywE](https://www.youtube.com/watch?v=GDAr4xFrywE)

Screen recording link for the video with the background retained and the character replaced

[https://www.youtube.com/watch?v=WFv2ZG8DwQc](https://www.youtube.com/watch?v=WFv2ZG8DwQc)


## For Instruction Regarding Setup
[Setup Instruction](https://github.com/srmahapatra95/Comfy-UI-Task/blob/main/setup_instructions.md).

## For Known Errors and Fixes
[Errors and Fixes](https://github.com/srmahapatra95/Comfy-UI-Task/blob/main/setup_errors.md).


## Set up and the Execution of the Comfy UI

### Workflow Execution Steps 
```
1. Click File -> Open , load the "wan22_animate.json" file.

2. If all nodes initialize correctly (no red node errors) else install Comfy UI manager and install the missing nodes either through the manager or Manually.

3. Replace the placeholder image and the video with the the input videos and the images. set Width and Set Height determine the exact resolution to which a video output will be resized to. (Here Square resolution of 512 x 512 has been taken for video workflows for WAN 2.2 animate models.)

4. frame_load_cap parameter is set to 0. This allows the Suite to take all the frames into consideration. This is done as the input video is very small, if the input video is very large then the then we can take certain frames for video generations by setting the parameter.

5. Setting the Grow Mask with Blur Node. The GrowMaskWithBlur node in ComfyUI is a tool designed to expand a given mask and apply a Gaussian blur to its edges, resulting in a smoother and more natural transition between the masked and unmasked areas. The expand parameter in the GrowMaskWithBlur node within ComfyUI controls the number of pixels by which the input mask is expanded or contracted. For the video execution this was set to 25.

6. In WAN Video Animate Embeds Node, the get_background_image and the get_mask parameter
    - If connected then only replaces the character from the image while retaining the background of the video.
    - If disconnected then both the character and the background of the video are replaced.

7. WAN Video TextEncoded Cached Node is used to set the prompt that guides the model to generate particular kind of video. The text encoded node transforms a textual description into embeddings, which serve as essential inputs to video generation models, thereby directing the model to produce content that aligns with the provided prompt.

8.The frame rate parameter in the Video Combine node of ComfyUI determines the number of frames displayed per second in the resulting video. This setting directly affects the smoothness of the motion; a higher frame rate results in smoother playback, while a lower frame rate can make motion less smooth.

9. Click the Run Button

10. After the video is generated save the workflow and save the output video.

```


## ⚙️ 1️⃣ Key Factors That Affect Generation Time
| Factor                        | Description                                                               | Effect on Speed               |
| ----------------------------- | ------------------------------------------------------------------------- | ----------------------------- |
| **GPU Model**                 | You’re using RTX 4090 — extremely fast for diffusion models.              | ✅ Very good performance       |
| **Resolution**                | 512×512 runs much faster than 720×720 or 1080p.                           | 🔺 Higher resolution = slower |
| **Clip Length**               | Each second of video adds many diffusion frames.                          | 🔺 Longer video = slower      |
| **FPS (Frames Per Second)**   | Each frame = 1 generation step. 20 fps × 5 s = 100 frames total.          | 🔺 Higher FPS = slower        |
| **Diffusion Steps per Frame** | Usually 8–25. More steps = better quality but slower.                     | 🔺 More steps = slower        |
| **Batching / Caching**        | If the pipeline reuses context (like I2V) it speeds up subsequent frames. | ✅ Faster after 1st frame      |
| **LoRA / Model size**         | Wan 2.2 Animate is large (14 B parameters).                               | ⚠️ Heavy VRAM & slower I/O    |


🕐 2️⃣ Rough Time Estimates (for RTX 4090)
| Video Length | Resolution | FPS    | Diffusion Steps | Approx Time   |
| ------------ | ---------- | ------ | --------------- | ------------- |
| 3 s          | 512×512    | 20 fps | 12              | **2–4 min**   |
| 5 s          | 512×512    | 20 fps | 15              | **4–7 min**   |
| 10 s         | 512×512    | 20 fps | 20              | **8–15 min**  |
| 5 s          | 720×720    | 20 fps | 20              | **10–20 min** |

🧠 3️⃣ Pro Tips to Optimize

- ✅ Use 512×512 resolution for early testing — 2–4 minutes typical.
- ✅ Start with 5 s clips (100 frames @ 20 fps).
- ✅ Set diffusion steps ≤ 15 for draft quality.
- ✅ Enable FP16 or BF16 precision for faster tensor operations.
- ✅ Ensure xformers or flash-attention is installed in ComfyUI (big speedup).
- ✅ Disable live preview if enabled — it adds render overhead.


🎬 4️⃣ Typical End-to-End Workflow Timeline

| Step                           | Duration           |
| ------------------------------ | ------------------ |
| Model & LoRA loading           | 1–2 min            |
| Pose detection & preprocessing | 1 min              |
| Animation generation           | 3–10 min           |
| Video combine/export           | 1 min              |
| **Total (5 s clip @ 512×512)** | **≈ 6–12 minutes** |
