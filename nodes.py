"""
Ollama Prompt Enhancer for ComfyUI.

Design goals (RTX 3090 24 GB + 42 GB RAM):
  * fast:   one persistent HTTP session, small context window (num_ctx), capped output (num_predict),
            thinking disabled, streaming so we can stop on ComfyUI "Cancel", LRU cache of results;
  * light:  by default the LLM is unloaded from VRAM right after the answer (keep_alive=0) so the
            diffusion model gets the whole card back (the "VRAM handoff" idea from Strata's
            elastic-VRAM mode); the GGUF stays in the OS file cache, so the next load takes ~1 s.
"""

import base64
import hashlib
import io
import json
import re
from collections import OrderedDict

import requests

from .presets import PRESETS, PRESET_NAMES

DEFAULT_URL = "http://127.0.0.1:11434"
KEEP_ALIVE_OPTIONS = ["0 (unload after run)", "30s", "2m", "5m", "30m", "-1 (keep loaded)"]
THINK_OPTIONS = ["off", "on", "model default"]

_session = requests.Session()  # keep-alive TCP connection to Ollama
_cache: "OrderedDict[str, str]" = OrderedDict()
_CACHE_MAX = 64
_needs_prefill: set = set()  # models detected as "thinking-only"
_sizes: dict = {}
VRAM_MODES = ["auto (free only what LLM needs)", "always unload ComfyUI models", "never touch ComfyUI"]

_IMG_TOKENS = 1100  # rough context cost of one <=768px image for VLMs (gemma3 ~260, qwen3-vl ~1024)
_CTX_MAX = 32768
_CTX_ERR_RE = re.compile(r"request \((\d+) tokens\) exceeds the available context", re.I)


def _ctx_round(n: int) -> int:
    return min(_CTX_MAX, -(-int(n) // 256) * 256)


# ----------------------------------------------------------------------------- helpers
def _base(url: str) -> str:
    return (url or DEFAULT_URL).strip().rstrip("/")


def _list_models(url: str = DEFAULT_URL):
    try:
        r = _session.get(f"{_base(url)}/api/tags", timeout=4)
        r.raise_for_status()
        data = r.json().get("models", [])
        # small models first: they are what you want for prompt enhancing
        models = [m["name"] for m in sorted(data, key=lambda m: (m.get("size", 0), m["name"]))]
        return models or ["<no models: run `ollama pull qwen3:4b`>"]
    except Exception:
        return ["<ollama offline - start it and press R>"]


def _keep_alive_value(choice: str):
    v = choice.split(" ")[0]
    return int(v) if v.lstrip("-").isdigit() else v


_THINK_RE = re.compile(r"<think>.*?</think>", re.S | re.I)
_FENCE_RE = re.compile(r"^\s*```(?:json|text|markdown)?\s*(.*?)\s*```\s*$", re.S | re.I)
_PREFIX_RE = re.compile(
    r"^\s*(here(?:'s| is)[^:\n]*:|enhanced prompt:|prompt:|final prompt:|output:)\s*", re.I
)


def _clean(text: str) -> str:
    text = _THINK_RE.sub("", text)
    if "</think>" in text.lower():  # unterminated opening tag cut by the template
        text = re.split(r"</think>", text, flags=re.I)[-1]
    text = text.strip()
    m = _FENCE_RE.match(text)
    if m:
        text = m.group(1).strip()
    text = _PREFIX_RE.sub("", text)
    text = text.strip().strip("`").strip()
    if len(text) > 1 and text[0] == text[-1] and text[0] in "\"'«“":
        text = text[1:-1].strip()
    if text.endswith("”") and text.startswith("“"):
        text = text[1:-1].strip()
    return text


def _images_to_b64(image, max_side: int, max_images: int = 4):
    """ComfyUI IMAGE tensor [B,H,W,C] float 0..1 -> list of base64 JPEGs (downscaled for speed).

    A video batch (hundreds of frames) is sampled down to `max_images` evenly spaced frames:
    every image costs hundreds of context tokens and prompt-processing time.
    """
    import numpy as np
    from PIL import Image

    b = int(image.shape[0])
    k = max(1, min(max_images, b))
    idx = [0] if k == 1 else sorted({round(i * (b - 1) / (k - 1)) for i in range(k)})
    if b > k:
        print(f"[OllamaPromptEnhancer] image batch {b} -> sending {len(idx)} frames {idx}")

    out = []
    for i in idx:
        arr = (image[i].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(arr).convert("RGB")
        if max(img.size) > max_side:
            img.thumbnail((max_side, max_side), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        out.append(base64.b64encode(buf.getvalue()).decode("ascii"))
    return out


def _model_size(url: str, model: str) -> int:
    """Bytes of the model file (from /api/tags), cached."""
    key = (_base(url), model)
    if key not in _sizes:
        try:
            r = _session.get(f"{_base(url)}/api/tags", timeout=4)
            for m in r.json().get("models", []):
                _sizes[(_base(url), m["name"])] = int(m.get("size", 0))
        except Exception:
            pass
    return _sizes.get(key, 0)


def _is_loaded(url: str, model: str) -> bool:
    try:
        r = _session.get(f"{_base(url)}/api/ps", timeout=3)
        return any(m["name"] == model for m in r.json().get("models", []))
    except Exception:
        return False


def _prepare_vram(mode: str, url: str, model: str, num_ctx: int, gpu_layers: int):
    """VRAM handoff ComfyUI -> Ollama (idea of Strata's elastic VRAM).

    auto: ask ComfyUI to offload just enough of its cached models so the LLM fits fully on the GPU.
          Without this, Ollama loads into an almost full card and Windows' sysmem fallback makes the
          load take tens of seconds instead of ~1-3 s.
    """
    if mode.startswith("never") or gpu_layers == 0:
        return
    try:
        import comfy.model_management as mm
        mm.get_torch_device  # noqa: B018  (make sure it's ComfyUI's module)
    except Exception:
        return
    if mode.startswith("always"):
        mm.unload_all_models()
        mm.soft_empty_cache()
        return
    if _is_loaded(url, model):
        return
    size = _model_size(url, model)
    if not size:
        return
    # weights + KV cache/compute buffers (~0.4 MB per ctx token is generous for <=8B models) + margin
    need = int(size * 1.1 + num_ctx * 0.4 * 1024 * 1024 + 768 * 1024 * 1024)
    dev = mm.get_torch_device()
    free = mm.get_free_memory(dev)
    if free < need:
        print(f"[OllamaPromptEnhancer] VRAM free {free / 2**30:.1f} GB < needed {need / 2**30:.1f} GB "
              f"-> offloading ComfyUI models")
        mm.free_memory(need, dev)
        mm.soft_empty_cache()


def _interrupted():
    try:
        import comfy.model_management as mm

        mm.throw_exception_if_processing_interrupted()
    except ImportError:
        pass


def _unload(url: str, model: str):
    try:
        _session.post(
            f"{_base(url)}/api/generate", json={"model": model, "keep_alive": 0}, timeout=10
        )
    except Exception as e:
        print(f"[OllamaPromptEnhancer] unload failed: {e}")


class _ThinkingLeak(Exception):
    """Model streams reasoning although think=off (thinking-only models like qwen3-vl)."""


# Short reasoning block placed as the start of the assistant turn: the model sees a non-empty
# completed `<think>...</think>` block and outputs the final prompt immediately.
_NO_THINK_PREFILL = {"role": "assistant", "content": "<think>\nI will write the prompt directly.\n</think>\n\n"}


def _chat(url, payload, num_predict, abort_on_thinking=False):
    """Streaming /api/chat. Returns (content, thinking). Aborts on ComfyUI cancel."""
    try:
        from comfy.utils import ProgressBar

        pbar = ProgressBar(max(1, num_predict))
    except Exception:
        pbar = None

    content, thinking, n = [], [], 0
    with _session.post(f"{_base(url)}/api/chat", json=payload, stream=True, timeout=(5, 600)) as r:
        if r.status_code != 200:
            try:
                err = r.json().get("error", r.text)
            except Exception:
                err = r.text
            raise RuntimeError(f"Ollama HTTP {r.status_code}: {err}")
        for line in r.iter_lines():
            _interrupted()  # raises -> context manager closes the stream, Ollama stops generating
            if not line:
                continue
            chunk = json.loads(line)
            if "error" in chunk:
                raise RuntimeError(f"Ollama: {chunk['error']}")
            msg = chunk.get("message") or {}
            if msg.get("content"):
                content.append(msg["content"])
                n += 1
            if msg.get("thinking"):
                if abort_on_thinking:
                    raise _ThinkingLeak()  # closing the stream stops generation in Ollama
                thinking.append(msg["thinking"])
            if pbar is not None and n:
                pbar.update_absolute(min(n, num_predict), num_predict)
            if chunk.get("done"):
                ev, ed = chunk.get("eval_count", 0), chunk.get("eval_duration", 0)
                ld = chunk.get("load_duration", 0) / 1e9
                tps = ev / (ed / 1e9) if ed else 0
                print(
                    f"[OllamaPromptEnhancer] {payload['model']}: {ev} tok, "
                    f"{tps:.1f} tok/s, load {ld:.2f}s, total {chunk.get('total_duration', 0) / 1e9:.2f}s"
                )
                break
    return "".join(content), "".join(thinking)


# ----------------------------------------------------------------------------- nodes
class OllamaPromptEnhancer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "a cat astronaut on the moon"}),
                "model": (_list_models(),),
                "preset": (PRESET_NAMES, {"default": "General (natural language)"}),
                "system_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "placeholder": "Custom system prompt. With preset=Custom it replaces the preset, "
                        "otherwise it is appended to the preset as extra rules.",
                    },
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0x7FFFFFFF}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05}),
                "max_tokens": ("INT", {"default": 300, "min": 16, "max": 4096, "step": 16}),
                "num_ctx": ("INT", {"default": 2048, "min": 512, "max": 32768, "step": 256,
                    "tooltip": "Minimum context. It grows automatically when prompt + images + answer need more."}),
                "keep_alive": (KEEP_ALIVE_OPTIONS, {"default": KEEP_ALIVE_OPTIONS[0]}),
                "think": (THINK_OPTIONS, {"default": "off"}),
            },
            "optional": {
                "image": ("IMAGE",),
                "ollama_url": ("STRING", {"default": DEFAULT_URL}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "num_gpu_layers": (
                    "INT",
                    {"default": -1, "min": -1, "max": 999,
                     "tooltip": "-1 = auto (all on GPU, fastest). 0 = CPU only (no VRAM, slower)."},
                ),
                "image_max_side": ("INT", {"default": 768, "min": 224, "max": 2048, "step": 32}),
                "max_images": ("INT", {"default": 4, "min": 1, "max": 32,
                    "tooltip": "Video/batch input is sampled to this many evenly spaced frames."}),
                "comfy_vram": (VRAM_MODES, {"default": VRAM_MODES[0],
                    "tooltip": "How to make room on the GPU for the LLM before it loads."}),
                "bypass": ("BOOLEAN", {"default": False, "tooltip": "Pass the prompt through unchanged."}),
                "use_cache": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("enhanced_prompt", "original_prompt", "system_prompt_used")
    FUNCTION = "enhance"
    CATEGORY = "Ollama"

    @classmethod
    def VALIDATE_INPUTS(cls, model):
        # model list is fetched live; don't fail a saved workflow if Ollama was offline at startup
        return True

    def enhance(self, prompt, model, preset, system_prompt, seed, temperature, max_tokens, num_ctx,
                keep_alive, think, image=None, ollama_url=DEFAULT_URL, top_p=0.9, num_gpu_layers=-1,
                image_max_side=768, max_images=4, comfy_vram=VRAM_MODES[0], bypass=False, use_cache=True):
        preset_text = PRESETS.get(preset, "")
        extra = (system_prompt or "").strip()
        system = extra if preset == "Custom" else (preset_text + ("\n\n" + extra if extra else ""))

        if bypass or not prompt.strip() and image is None:
            return (prompt, prompt, system)
        if model.startswith("<"):
            raise RuntimeError(f"Ollama model not available: {model}")

        is_structured = "MiniMax-H3" in preset
        eff_max_tokens = max(max_tokens, 900 if is_structured else max_tokens)

        options = {
            "temperature": temperature,
            "top_p": top_p,
            "seed": seed,
            "num_predict": eff_max_tokens,
            "num_ctx": num_ctx,
        }
        if num_gpu_layers >= 0:
            options["num_gpu"] = num_gpu_layers

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        user = {"role": "user", "content": prompt.strip() or "Describe the image."}
        img_hash = ""
        if image is not None:
            user["images"] = _images_to_b64(image, image_max_side, max_images)
            img_hash = hashlib.sha1("".join(user["images"]).encode()).hexdigest()
        messages.append(user)

        # context = max(user minimum, rough need); grows only when needed (bigger ctx = more VRAM)
        n_img = len(user.get("images", []))
        est = (len(system) + len(user["content"])) // 3 + n_img * _IMG_TOKENS + eff_max_tokens + 64
        options["num_ctx"] = _ctx_round(max(num_ctx, est))
        if options["num_ctx"] > num_ctx:
            print(f"[OllamaPromptEnhancer] num_ctx {num_ctx} -> {options['num_ctx']} (est. {est} tokens)")

        key = hashlib.sha1(
            json.dumps([_base(ollama_url), model, messages[0]["content"] if system else "",
                        user["content"], img_hash, options, think], sort_keys=True).encode()
        ).hexdigest()
        if use_cache and key in _cache:
            _cache.move_to_end(key)
            print("[OllamaPromptEnhancer] cache hit")
            return (_cache[key], prompt, system)

        _prepare_vram(comfy_vram, ollama_url, model, options["num_ctx"], num_gpu_layers)

        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": options,
            "keep_alive": _keep_alive_value(keep_alive),
        }
        if think != "model default":
            payload["think"] = think == "on"

        no_think = think == "off"
        try:
            for attempt in range(3):
                try:
                    content, _thinking = self._call(ollama_url, payload, messages, model, eff_max_tokens, no_think)
                    break
                except RuntimeError as e:
                    m = _CTX_ERR_RE.search(str(e))
                    if not m or attempt == 2:
                        raise
                    need = _ctx_round(int(m.group(1)) + eff_max_tokens + 64)
                    if need <= options["num_ctx"] or need > _CTX_MAX:
                        raise
                    print(f"[OllamaPromptEnhancer] context too small: num_ctx {options['num_ctx']} -> {need}")
                    options["num_ctx"] = need
        except requests.ConnectionError:
            raise RuntimeError(f"Cannot connect to Ollama at {_base(ollama_url)}. Is `ollama serve` running?")
        except BaseException:
            # cancelled from ComfyUI: honour keep_alive=0 so VRAM is freed anyway
            if payload["keep_alive"] == 0:
                _unload(ollama_url, model)
            raise

        result = _clean(content)
        if "JSON" in preset and "{" in result and "}" in result:
            s, e = result.find("{"), result.rfind("}")
            candidate = result[s : e + 1]
            try:
                result = json.dumps(json.loads(candidate), ensure_ascii=False, indent=2)
            except Exception:
                result = candidate
        if not result:
            raise RuntimeError(
                "Ollama returned an empty answer (thinking model ran out of max_tokens?). "
                "Set think=off or raise max_tokens."
            )

        if use_cache:
            _cache[key] = result
            while len(_cache) > _CACHE_MAX:
                _cache.popitem(last=False)
        return (result, prompt, system)

    @staticmethod
    def _call(url, payload, messages, model, max_tokens, no_think):
        """One generation, handling models that ignore or reject the `think` flag."""
        prefill = no_think and model in _needs_prefill
        if prefill:
            payload.pop("think", None)
            payload["messages"] = messages + [_NO_THINK_PREFILL]
        try:
            return _chat(url, payload, max_tokens, abort_on_thinking=no_think and not prefill)
        except _ThinkingLeak:
            print(f"[OllamaPromptEnhancer] {model} ignores think=false -> using empty-think prefill")
            _needs_prefill.add(model)
            payload.pop("think", None)
            payload["messages"] = messages + [_NO_THINK_PREFILL]
            return _chat(url, payload, max_tokens)
        except RuntimeError as e:
            # models without thinking support reject the "think" field -> retry without it
            if "think" in str(e).lower() and "think" in payload:
                payload.pop("think")
                return _chat(url, payload, max_tokens)
            raise


class OllamaUnloadModels:
    """Frees VRAM held by Ollama. Wire any STRING through it to control execution order."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
                "unload": ("BOOLEAN", {"default": True}),
            },
            "optional": {"ollama_url": ("STRING", {"default": DEFAULT_URL})},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "run"
    CATEGORY = "Ollama"

    @classmethod
    def IS_CHANGED(cls, text="", unload=True, ollama_url=DEFAULT_URL):
        # Return a deterministic hash of inputs so downstream CLIPTextEncode can cache its conditioning
        return hashlib.sha1(f"{text}|{unload}|{ollama_url}".encode("utf-8")).hexdigest()

    def run(self, text, unload=True, ollama_url=DEFAULT_URL):
        if unload:
            try:
                r = _session.get(f"{_base(ollama_url)}/api/ps", timeout=3)
                for m in r.json().get("models", []):
                    _unload(ollama_url, m["name"])
                    print(f"[OllamaPromptEnhancer] unloaded {m['name']}")
            except Exception as e:
                print(f"[OllamaPromptEnhancer] /api/ps failed: {e}")
        return (text,)


NODE_CLASS_MAPPINGS = {
    "OllamaPromptEnhancer": OllamaPromptEnhancer,
    "OllamaUnloadModels": OllamaUnloadModels,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "OllamaPromptEnhancer": "Ollama Prompt Enhancer ⚡",
    "OllamaUnloadModels": "Ollama Unload Models (free VRAM)",
}
