import os
import re
from PIL import Image
import base64
from io import BytesIO
from functools import lru_cache
import threading
import time

# Global variable to store selected VLM backend
_VLM_BACKEND = "gemini"  # options: gemini, gpt4o
_VLM_MAX_IMAGE_EDGE = int(os.getenv("REST3D_VLM_MAX_IMAGE_EDGE", "1536"))
_VLM_STATS_LOCK = threading.Lock()
_VLM_STATS = {
    "requests": 0,
    "images": 0,
    "image_bytes": 0,
    "elapsed_seconds": 0.0,
    "file_cache_hits": 0,
    "file_cache_misses": 0,
}
_GEMINI_CLIENT = None
_GEMINI_CLIENT_KEY = None


def set_vlm_backend(backend):
    """Set which VLM backend to use: 'gemini' or 'gpt4o'"""
    global _VLM_BACKEND
    _VLM_BACKEND = backend


def set_vlm_image_max_edge(max_edge):
    """Set the longest image edge sent to a remote VLM (0 disables resizing)."""
    global _VLM_MAX_IMAGE_EDGE
    max_edge = int(max_edge)
    if max_edge != 0 and max_edge < 256:
        raise ValueError("VLM image max edge must be 0 or at least 256 pixels")
    if max_edge != _VLM_MAX_IMAGE_EDGE:
        _VLM_MAX_IMAGE_EDGE = max_edge
        _image_file_to_png_bytes.cache_clear()


def get_vlm_stats(reset=False):
    """Return aggregate request/payload/cache statistics for the current process."""
    with _VLM_STATS_LOCK:
        stats = dict(_VLM_STATS)
        if reset:
            for key in _VLM_STATS:
                _VLM_STATS[key] = 0.0 if key == "elapsed_seconds" else 0
    return stats


def _resize_for_vlm(image, max_edge):
    image = image.convert("RGB")
    if max_edge <= 0 or max(image.size) <= max_edge:
        return image
    scale = max_edge / float(max(image.size))
    size = tuple(max(1, round(dim * scale)) for dim in image.size)
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    return image.resize(size, resampling)


def resize_image_for_vlm(image):
    """Return an RGB PIL image resized with the same policy used for VLM uploads."""
    if not isinstance(image, Image.Image):
        raise TypeError(f"Expected PIL.Image, got {type(image).__name__}")
    return _resize_for_vlm(image, _VLM_MAX_IMAGE_EDGE)


def _encode_png(image, max_edge):
    buffered = BytesIO()
    _resize_for_vlm(image, max_edge).save(buffered, format="PNG", optimize=True)
    return buffered.getvalue()


@lru_cache(maxsize=128)
def _image_file_to_png_bytes(path, mtime_ns, file_size, max_edge):
    # mtime_ns and file_size form part of the cache key so changed files are not reused.
    del mtime_ns, file_size
    with Image.open(path) as image:
        return _encode_png(image, max_edge)


def image_to_png_bytes(image):
    """Encode a PIL image or image path for VLM upload at the configured size."""
    if isinstance(image, (str, os.PathLike)):
        path = os.path.abspath(os.fspath(image))
        stat = os.stat(path)
        before = _image_file_to_png_bytes.cache_info()
        data = _image_file_to_png_bytes(
            path, stat.st_mtime_ns, stat.st_size, _VLM_MAX_IMAGE_EDGE
        )
        after = _image_file_to_png_bytes.cache_info()
        with _VLM_STATS_LOCK:
            if after.hits > before.hits:
                _VLM_STATS["file_cache_hits"] += 1
            else:
                _VLM_STATS["file_cache_misses"] += 1
        return data
    if not isinstance(image, Image.Image):
        raise TypeError(f"Expected PIL.Image or image path, got {type(image).__name__}")
    return _encode_png(image, _VLM_MAX_IMAGE_EDGE)


def image_to_base64(image):
    """Convert PIL Image to base64 string for API calls"""
    return base64.b64encode(image_to_png_bytes(image)).decode()


def _record_vlm_request(image_count, image_bytes, elapsed_seconds):
    with _VLM_STATS_LOCK:
        _VLM_STATS["requests"] += 1
        _VLM_STATS["images"] += image_count
        _VLM_STATS["image_bytes"] += image_bytes
        _VLM_STATS["elapsed_seconds"] += elapsed_seconds
    print(
        f"VLM request: {image_count} image(s), {image_bytes / (1024 * 1024):.2f} MiB, "
        f"{elapsed_seconds:.1f}s"
    )


def generate_vlm_response_gpt4o(messages, save_dir=None, save_name='test'):
    """Generate response using GPT-4o via OpenAI API (uses requests, no openai package needed)"""
    import requests
    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    # Convert messages to GPT-4o format
    gpt_messages = []
    image_count = 0
    image_bytes = 0
    for msg in messages:
        content_list = []
        for item in msg["content"]:
            if item["type"] == "text":
                content_list.append({"type": "text", "text": item["text"]})
            elif item["type"] == "image":
                img_bytes = image_to_png_bytes(item["image"])
                base64_image = base64.b64encode(img_bytes).decode()
                image_count += 1
                image_bytes += len(img_bytes)
                content_list.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                })
        gpt_messages.append({"role": msg["role"], "content": content_list})

    # Prepare API request
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "gpt-4o",
        "messages": gpt_messages,
        "max_tokens": 4096,
        "temperature": 0
    }

    # Make API call with retries
    max_retries = 3
    started = time.monotonic()
    for attempt in range(max_retries):
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            output_text = result["choices"][0]["message"]["content"]
            break
        except Exception as e:
            if attempt == max_retries - 1:
                raise Exception(f"GPT-4o API call failed after {max_retries} attempts: {e}")
            print(f"   ⚠️  API call failed (attempt {attempt+1}/{max_retries}), retrying...")
            time.sleep(2)

    _record_vlm_request(image_count, image_bytes, time.monotonic() - started)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, f'{save_name}.txt'), 'w') as file:
            file.write(output_text)

    return output_text


def generate_vlm_response_gemini(messages, save_dir=None, save_name="test",
                                 model="gemini-3-flash-preview", max_retries=None):
    """Generate response using Gemini via Google API (uses requests, no google-generativeai package needed)"""
    from google import genai
    from google.genai import types

    # Get API key from environment
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")

    project = os.getenv("GOOGLE_CLOUD_PROJECT", "phsyscene")
    os.environ["GOOGLE_CLOUD_PROJECT"] = project

    global _GEMINI_CLIENT, _GEMINI_CLIENT_KEY
    if _GEMINI_CLIENT is None or _GEMINI_CLIENT_KEY != api_key:
        _GEMINI_CLIENT = genai.Client(api_key=api_key)
        _GEMINI_CLIENT_KEY = api_key
    client = _GEMINI_CLIENT

    # Build a single user turn from your messages (you can extend to multi-turn later)
    parts = []
    image_count = 0
    image_bytes = 0
    for msg in messages:
        for item in msg["content"]:
            if item["type"] == "text":
                parts.append(types.Part.from_text(text=item["text"]))
            elif item["type"] == "image":
                img_bytes = image_to_png_bytes(item["image"])
                image_count += 1
                image_bytes += len(img_bytes)
                parts.append(types.Part.from_bytes(data=img_bytes, mime_type="image/png"))
            else:
                raise ValueError(f"Unknown content type: {item['type']}")

    contents = [types.Content(role="user", parts=parts)]

    if max_retries is None:
        max_retries = max(1, int(os.getenv("REST3D_VLM_MAX_RETRIES", "3")))
    rate_limit_base_delay = max(
        1.0, float(os.getenv("REST3D_VLM_RATE_LIMIT_BASE_DELAY", "20"))
    )
    retry_max_delay = max(
        rate_limit_base_delay,
        float(os.getenv("REST3D_VLM_RETRY_MAX_DELAY", "60")),
    )

    started = time.monotonic()
    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=0,
                    max_output_tokens=8192,
                ),
            )
            output_text = resp.text or ""
            break
        except Exception as e:
            if attempt == max_retries - 1:
                raise RuntimeError(f"Gemini API call failed after {max_retries} attempts: {e}")
            is_rate_limit = "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)
            wait = min(rate_limit_base_delay * (2 ** attempt), retry_max_delay) \
                if is_rate_limit else 5
            print(f"⚠️ API call failed (attempt {attempt+1}/{max_retries}), waiting {wait}s... {e}")
            time.sleep(wait)

    _record_vlm_request(image_count, image_bytes, time.monotonic() - started)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, f"{save_name}.txt"), "w") as f:
            f.write(output_text)

    return output_text


def generate_vlm_response(messages, save_dir=None, save_name='test', save=True):
    """
    Generate VLM response using the selected backend.

    Args:
        messages: List of message dicts with role and content
        save_dir: Directory to save response
        save_name: Name for saved file
        save: Whether to save (kept for backwards compatibility)

    Returns:
        str: VLM response text
    """
    if _VLM_BACKEND == "gpt4o":
        return generate_vlm_response_gpt4o(messages, save_dir, save_name)
    elif _VLM_BACKEND == "gemini":
        return generate_vlm_response_gemini(messages, save_dir, save_name)
    else:
        raise ValueError(f"Unknown VLM backend: {_VLM_BACKEND}. Choose from: gpt4o, gemini")


def analyze_scene_object_lists(image, save_dir=None, vlm_prompt_file=None):
    """
    Use VLM to identify all salient objects in an image.

    Args:
        image: PIL Image or file path
        save_dir: Directory to save vlm_objects.json
        vlm_prompt_file: Path to txt file containing VLM prompt. If None,
            defaults to ``rest3d/prompts/list_objects.txt`` shipped with
            this package. Relative paths are resolved against ``PROMPTS_DIR``.

    Returns:
        list[str]: Object description prompts, one per object
    """
    from rest3d import PROMPTS_DIR

    # Resolve prompt file path
    if vlm_prompt_file is None:
        vlm_prompt_file = os.path.join(PROMPTS_DIR, "list_objects.txt")
    elif not os.path.isabs(vlm_prompt_file):
        vlm_prompt_file = os.path.join(PROMPTS_DIR, vlm_prompt_file)
    with open(vlm_prompt_file, "r", encoding="utf-8") as f:
        vlm_prompt = f.read().strip()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": vlm_prompt},
            ],
        }
    ]

    response = generate_vlm_response(messages, save_dir, 'scene_object_lists')
    print(f"VLM response:\n{response}")

    # Parse: one object per line, strip numbering prefixes
    objects = []
    for line in response.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        line = re.sub(r'^[\d]+[.\)]\s*', '', line)
        line = re.sub(r'^[-*]\s*', '', line)
        line = line.strip()
        if line and len(line) < 100:
            objects.append(line)
    objects.append("the floor")

    return objects
