# ComfyUI-OllamaPromptEnhancer ⚡

Ultra-fast, low-VRAM **Ollama** prompt enhancer custom node for ComfyUI with system prompt presets, vision/video frame sampling, and elastic VRAM handoff.

## Features

- **Elastic VRAM Handoff (`comfy_vram = auto`)**: Automatically calculates the exact GPU memory needed by the LLM and offloads only the required amount from ComfyUI before loading the model, preventing Windows shared-VRAM slowdowns.
- **Zero-Footprint Unload (`keep_alive = 0`)**: Immediately frees GPU VRAM after prompt generation so diffusion/video models get 100% of the GPU, while keeping GGUF weights warm in system RAM cache for ~1–3s reloads.
- **Built-in System Prompt Presets + Custom Rules**:
  - `General (natural language)`
  - `Flux / Qwen-Image / HiDream`
  - `SDXL / Pony / Illustrious (tags)`
  - `Wan / video (motion)`
  - `Bernini-R (R2V / R2I / V2V / RV2V)`
  - `MiniMax-H3 (Ref2VA / I2VA / T2VA)`
  - `MiniMax-H3 JSON (Chain Plan)`
  - `Photoreal photography`
  - `Image -> prompt (vision)`
  - `Translate to English only`
  - `Custom` (or append your own `system_prompt` rules to any preset)
- **Vision & Video Batch Ready**: Accepts `IMAGE` inputs (single images or full video batches) and automatically samples `max_images` evenly spaced keyframes while dynamically expanding `num_ctx`.
- **Smart Thinking Bypass**: Disables reasoning overhead (`think = off`) and automatically applies a prefill bypass for thinking-only VLMs (such as `qwen3-vl`).
- **Instant Cancel & LRU Cache**: Streams tokens with a ComfyUI progress bar, aborts generation immediately on **Cancel**, and caches identical requests in memory.

## Nodes (Category: `Ollama`)

1. **`Ollama Prompt Enhancer ⚡`** — Enhances text/image/video prompts (`enhanced_prompt`, `original_prompt`, `system_prompt_used`).
2. **`Ollama Unload Models (free VRAM)`** — Pass-through string node that explicitly unloads all running Ollama models from VRAM.

## Installation

Clone into your `ComfyUI/custom_nodes` directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/LBayani/ComfyUI-OllamaPromptEnhancer.git
```

Restart ComfyUI. Make sure [Ollama](https://ollama.com/) is running (`http://127.0.0.1:11434`).
