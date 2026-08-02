import cv2
import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from datetime import datetime
import numpy as np
import pycocotools.mask as mask_utils
from collections import defaultdict
import re
from PIL import Image, ImageDraw, ImageOps
from glob import glob
from rest3d.utils.vis import visualize_masks_on_image_cv2, save_seg_obj
from rest3d.utils.vlm import (
    analyze_scene_object_lists,
    generate_vlm_response,
    get_vlm_stats,
    resize_image_for_vlm,
    set_vlm_backend,
    set_vlm_image_max_edge,
)
from rest3d.utils.io import load_image
from sam3.agent.client_sam3 import call_sam_service as call_sam_service_orig
from sam3.agent.inference import run_single_image_inference
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from rest3d.utils.log import get_logger, attach_file_handler

logger = get_logger("stage1")

_SCENE_TREE_PROMPT_VERSION = "batch-overview-v5-support-semantics"
_SCENE_TREE_RELATIONS = {
    "on", "inside", "supported-by", "attach", "hang", "on-attach"
}
_SCENE_TREE_PHYSICS_ROLES = {"dynamic", "kinematic", "fixed"}


def _excepthook(exc_type, exc_value, exc_tb):
    """Route uncaught exceptions through the logger so the per-image FileHandler
    captures the traceback into stage1_log_*.txt before the script exits."""
    tb_text = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    logger.error("Uncaught exception:\n" + tb_text)
    sys.__excepthook__(exc_type, exc_value, exc_tb)

sys.excepthook = _excepthook


def _explicit_parent_hint(child, available_parents):
    """Infer a support relation explicitly encoded in an object prompt/id."""
    child_text = re.sub(r"_\d+$", "", child.lower())
    match = re.search(r"_(on|in|inside)_(?:the_)?(.+)$", child_text)
    if match is None:
        return None
    cue, target_text = match.groups()
    target_tokens = {token for token in target_text.split("_") if token}
    if not target_tokens:
        return None

    candidates = []
    for parent in available_parents:
        if parent == child or parent == "floor-wall":
            continue
        parent_text = re.sub(r"_\d+$", "", parent.lower())
        # Do not match a target mentioned only in another object's location
        # qualifier (for example, "person_behind_the_desk" is not the desk).
        parent_subject = re.split(
            r"_(?:on|in|inside|behind|under|above|beside|near)_(?:the_)?",
            parent_text,
            maxsplit=1,
        )[0]
        parent_tokens = {token for token in parent_subject.split("_") if token}
        if target_tokens.issubset(parent_tokens):
            candidates.append((len(parent_tokens - target_tokens), len(parent_subject), parent))
    if not candidates:
        return None
    parent = min(candidates)[2]
    if cue in {"in", "inside"}:
        relation = "inside"
    elif parent in {"wall", "ceiling"}:
        relation = "attach"
    else:
        relation = "on"
    return parent, relation


def _normalize_scene_tree_edge(item, obj_ids, available_parents):
    """Validate one VLM-produced scene-tree edge and return its canonical form."""
    if not isinstance(item, dict):
        return None
    child_raw = str(item.get("child", "")).strip().strip('"`')
    parent_raw = str(item.get("parent", "")).strip().strip('"`')
    child = next((name for name in obj_ids if name.lower() == child_raw.lower()), None)
    parent = next(
        (name for name in available_parents if name.lower() == parent_raw.lower()), None
    )
    if child is None or parent is None or child == parent:
        return None
    relation = str(item.get("relation", "on")).lower().strip()
    relation = {
        "supported_by": "supported-by",
        "supported by": "supported-by",
        "support": "supported-by",
    }.get(relation, relation)
    if relation not in _SCENE_TREE_RELATIONS:
        relation = "on"
    obj_type = str(item.get("type", item.get("object_type", "movable"))).lower().strip()
    if obj_type not in {"fixed", "movable"}:
        obj_type = "movable"
    physics_role = str(item.get("physics_role", "")).lower().strip()
    explicit_hint = _explicit_parent_hint(child, available_parents)
    if explicit_hint is not None and explicit_hint != (parent, relation):
        hinted_parent, hinted_relation = explicit_hint
        logger.info(
            f"  Explicit prompt relation overrides VLM edge for '{child}': "
            f"{parent}/{relation} -> {hinted_parent}/{hinted_relation}"
        )
        parent, relation = explicit_hint
    if physics_role not in _SCENE_TREE_PHYSICS_ROLES:
        if obj_type == "fixed" or relation in {"attach", "hang", "on-attach"}:
            physics_role = "fixed"
        elif relation == "supported-by":
            physics_role = "kinematic"
        else:
            physics_role = "dynamic"
    return {
        "child": child,
        "parent": parent,
        "relation": relation,
        "type": obj_type,
        "physics_role": physics_role,
    }


def _parse_scene_tree_batch_response(response, obj_ids, available_parents):
    """Parse JSON (preferred) or legacy line output into validated edges by child id."""
    parsed_items = None
    text = response.strip()
    try:
        parsed = json.loads(text)
        parsed_items = parsed.get("edges") if isinstance(parsed, dict) else parsed
    except json.JSONDecodeError:
        match = re.search(r"\[[\s\S]*\]", text)
        if match:
            try:
                parsed_items = json.loads(match.group(0))
            except json.JSONDecodeError:
                parsed_items = None

    if not isinstance(parsed_items, list):
        parsed_items = []
        for line in text.splitlines():
            if "->" not in line:
                continue
            child_raw, rest = line.split("->", 1)
            fields = [field.strip() for field in rest.split("|")]
            if not fields:
                continue
            parsed_items.append({
                "child": child_raw.strip(),
                "parent": fields[0],
                "relation": fields[1] if len(fields) > 1 else "on",
                "type": fields[2] if len(fields) > 2 else "movable",
                "physics_role": fields[3] if len(fields) > 3 else "",
            })

    by_child = {}
    for item in parsed_items:
        edge = _normalize_scene_tree_edge(item, obj_ids, available_parents)
        if edge is not None:
            by_child[edge["child"]] = edge
    return by_child


def _make_scene_tree_contact_sheets(obj_ids, obj_data, per_sheet=6):
    """Pack full-scene object overlays into compact, numbered 3x2 contact sheets."""
    tile_w, tile_h, label_h, columns = 384, 512, 34, 3
    rows = (per_sheet + columns - 1) // columns
    sheets = []
    for start in range(0, len(obj_ids), per_sheet):
        chunk = obj_ids[start:start + per_sheet]
        sheet = Image.new("RGB", (tile_w * columns, (tile_h + label_h) * rows), "white")
        draw = ImageDraw.Draw(sheet)
        for offset, obj_id in enumerate(chunk):
            index = start + offset + 1
            row, col = divmod(offset, columns)
            x, y = col * tile_w, row * (tile_h + label_h)
            overlay = ImageOps.contain(obj_data[obj_id]["overlay"].convert("RGB"), (tile_w, tile_h))
            px = x + (tile_w - overlay.width) // 2
            py = y + (tile_h - overlay.height) // 2
            sheet.paste(overlay, (px, py))
            draw.rectangle((x, y + tile_h, x + tile_w, y + tile_h + label_h), fill="white")
            draw.text((x + 8, y + tile_h + 8), f"[{index}] {obj_id}", fill="black")
        sheets.append(sheet)
    return sheets


def _make_compact_mask_overlay(scene_image, binary_mask):
    """Render a mask overlay directly at VLM resolution without a 4K intermediate."""
    mask = Image.fromarray((binary_mask > 0).astype(np.uint8) * 255)
    resampling = getattr(Image, "Resampling", Image).NEAREST
    mask = mask.resize(scene_image.size, resampling)
    color = Image.new("RGB", scene_image.size, (255, 45, 45))
    highlighted = Image.blend(scene_image, color, 0.55)
    return Image.composite(highlighted, scene_image, mask)


def _query_scene_tree_batch(image_path, target_ids, obj_data, available_parents, debug_dir=None):
    """Infer all object relations in one VLM request using compact contact sheets."""
    sheets = _make_scene_tree_contact_sheets(target_ids, obj_data)
    index_map = "\n".join(f"[{idx}] {name}" for idx, name in enumerate(target_ids, 1))
    parents_str = ", ".join(f'"{name}"' for name in available_parents)
    content = [
        {"type": "image", "image": image_path},
        {"type": "text", "text": "Full scene image above. Numbered mask overview sheets follow."},
    ]
    for sheet_idx, sheet in enumerate(sheets, 1):
        content.extend([
            {"type": "image", "image": sheet},
            {"type": "text", "text": f"Mask overview sheet {sheet_idx}."},
        ])
        if debug_dir is not None:
            sheet.save(os.path.join(debug_dir, f"batch_overview_{sheet_idx}.png"))

    content.append({"type": "text", "text": f"""
The numbered overlays correspond to these exact object IDs:
{index_map}

For every object, determine the object or root that directly supports, holds, or anchors it.
Every parent MUST be one of: {parents_str}

Relation rules:
- "on": rests on the topmost surface of its parent.
- "inside": rests on an intermediate surface while the parent extends above it.
- "supported-by": partially or jointly supported while child and parent can overlap
  vertically; use this for a body supported by furniture, an object cradled by a stand,
  or another multi-contact arrangement whose reconstructed relative pose must be retained.
- "attach": fixed to a wall or ceiling.
- "hang": draped or hung from a rod, rail, or hook.
- "on-attach": rests on the floor and is also fixed against a wall.
- Prefer another object over floor when that object directly supports the child.
- Mounting evidence takes precedence over apparent bottom contact: a vertical light panel,
  display panel, window, picture, or similar installation aligned against a wall is
  parent="wall", relation="attach", even if its lower edge is close to the floor.
- Ceiling lights and panels are parent="ceiling", relation="attach".
- Use parent="floor" only when the object is genuinely supported by the ground, not merely
  because its 2D mask reaches the lower part of the image.
- IDs with the same descriptive prefix are instances of one semantic class and normally use
  the same support/mounting rule unless the overview clearly shows a real difference.

Type rules:
- "fixed": built-in furniture or anything attached to wall/ceiling.
- "movable": a person or an object that can be picked up or pushed.

Physics-role rules (separate from semantic type):
- "dynamic": a free rigid object that should settle under gravity.
- "kinematic": a posed, articulated, or multi-contact object whose observed pose must be
  retained during static-scene stabilization.
- "fixed": an immovable or anchored object.

Return ONLY a JSON array with exactly one entry per object, using the exact IDs above:
[
  {{"child": "exact_object_id", "parent": "exact_parent", "relation": "on", "type": "movable", "physics_role": "dynamic"}}
]
"""})
    response = generate_vlm_response([{"role": "user", "content": content}])
    logger.info(f"  Batch scene-tree response: {response.strip()}")
    return _parse_scene_tree_batch_response(response, target_ids, available_parents)


def _hash_file(path, hasher):
    with open(path, "rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            hasher.update(chunk)


def _scene_tree_fingerprint(image_path, seg_files, llm_name, batch_scene_tree):
    hasher = hashlib.sha256()
    hasher.update(_SCENE_TREE_PROMPT_VERSION.encode())
    hasher.update(llm_name.encode())
    hasher.update(str(bool(batch_scene_tree)).encode())
    _hash_file(image_path, hasher)
    for path in seg_files:
        hasher.update(os.path.basename(path).encode())
        _hash_file(path, hasher)
    return hasher.hexdigest()


def build_agent_components(args):
    """Build VLM adapters and lazily initialize SAM3 on the first mask request."""
    sam3_processor = None

    # llm_config is used only to name output files
    llm_config = {"name": args.vlm_backend}

    # Replace the vLLM-server send_generate_request with our VLM backend
    def send_generate_request(messages, _max_retries=3):
        def _convert(msgs):
            converted = []
            for msg in msgs:
                content = msg["content"]
                if isinstance(content, str):
                    converted.append({"role": msg["role"], "content": [{"type": "text", "text": content}]})
                    continue
                new_content = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image" and isinstance(item.get("image"), str):
                        # Keep paths intact so the VLM encoder can reuse resized PNG bytes
                        # across agent rounds without reopening/re-encoding the image.
                        new_content.append({"type": "image", "image": item["image"]})
                    else:
                        new_content.append(item)
                converted.append({"role": msg["role"], "content": new_content})
            return converted

        def _clean_think(text):
            """Strip <think> blocks (closed or truncated/unclosed)."""
            text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
            if '<think>' in text:
                text = text[:text.index('<think>')]
            return text.strip()

        FORMAT_HINT = (
            "IMPORTANT: Keep <think> under 100 words. "
            "If multiple masks partially cover the target, just select all of them with select_masks_and_return. "
            "Disambiguation rule for 'X on/in Y' queries: "
            "(1) If Y is a physical container that holds X (vase, bowl, jar, cup, basket, tray, pot), "
            "use Y as the text_prompt to capture the whole unit (e.g. 'flowers in glass vase' → text_prompt='glass vase'). "
            "(2) If Y is furniture or a surface that X merely rests on (table, shelf, bookcase, desk, counter, floor), "
            "X is the actual grounding target — use X as the text_prompt (e.g. 'books on coffee table' → text_prompt='books'). "
            "You MUST end with a <tool> JSON call. Example: "
            '<tool> {"name": "segment_phrase", "parameters": {"text_prompt": "noun phrase"}} </tool>'
        )

        # Detect if this is the examine_each_mask (Accept/Reject) phase by checking system prompt content
        _is_checking_phase = any(
            isinstance(m.get("content"), str) and "detail-oriented visual understanding" in m["content"]
            for m in messages if m.get("role") == "system"
        )

        # Only inject FORMAT_HINT on Round 1 (no prior segment_phrase assistant message yet)
        _is_first_round = not any(
            m.get("role") == "assistant" and
            any(isinstance(c, dict) and "segment_phrase" in c.get("text", "") for c in (m.get("content") if isinstance(m.get("content"), list) else []))
            for m in messages
        )

        prev_response = None
        for attempt in range(_max_retries):
            converted = _convert(messages)
            if prev_response is None:
                # First attempt: inject FORMAT_HINT only in Round 1 (non-checking phase)
                if not _is_checking_phase and _is_first_round:
                    converted.append({"role": "user", "content": [{"type": "text", "text": FORMAT_HINT}]})
            else:
                # Retry: feed the truncated response back to the model
                converted.append({"role": "assistant", "content": [{"type": "text", "text": prev_response}]})
                if _is_checking_phase:
                    converted.append({"role": "user", "content": [{"type": "text", "text":
                        "Your response was cut off. Output ONLY your verdict now: "
                        "<verdict>Accept</verdict> or <verdict>Reject</verdict>."}]})
                else:
                    converted.append({"role": "user", "content": [{"type": "text", "text":
                        "Your response was cut off before the <tool> call. "
                        "Output ONLY the <tool> JSON call now, no thinking. Example: "
                        '<tool> {"name": "segment_phrase", "parameters": {"text_prompt": "noun phrase"}} </tool>'}]})
            response = generate_vlm_response(converted)
            clean = _clean_think(response)
            if _is_checking_phase:
                # checking phase expects <verdict>...</verdict>; do not check <tool>
                return clean
            if "<tool>" in clean and "</tool>" in clean:
                return clean
            prev_response = response
            if attempt < _max_retries - 1:
                logger.info(f"    ⚠️ VLM response missing <tool> tags (attempt {attempt+1}), retrying...")
        return clean

    def call_sam_service(*call_args, **call_kwargs):
        nonlocal sam3_processor
        if sam3_processor is None:
            started = time.monotonic()
            logger.info("Initializing SAM3 model for the first uncached segmentation target...")
            model = build_sam3_image_model(
                checkpoint_path=os.path.expandvars("$HOME/sam3/checkpoints/sam3.pt"),
                load_from_HF=False,
            )
            sam3_processor = Sam3Processor(model, confidence_threshold=0.5)
            logger.info(f"SAM3 model initialized in {time.monotonic() - started:.1f}s")
        return call_sam_service_orig(
            *call_args, sam3_processor=sam3_processor, **call_kwargs
        )

    return llm_config, send_generate_request, call_sam_service


def decode_agent_masks(output_json_path):
    """Decode the RLE mask from the agent's JSON output into a numpy array."""
    with open(output_json_path, 'r') as f:
        pred = json.load(f)

    h = pred["orig_img_h"]
    w = pred["orig_img_w"]
    rle_masks = pred.get("pred_masks", [])
    scores = pred.get("pred_scores", [])

    masks = []
    valid_scores = []
    for i, rle_str in enumerate(rle_masks):
        rle = {"counts": rle_str, "size": [h, w]}
        binary_mask = mask_utils.decode(rle).astype(np.float32)
        masks.append(binary_mask)
        if i < len(scores):
            valid_scores.append(scores[i])

    if masks:
        masks = np.stack(masks)
    else:
        masks = np.zeros((0, h, w), dtype=np.float32)

    return masks, np.array(valid_scores)


def analyze_scene_tree(image_path, seg_obj_dir, agent_output_dir, image_stem, llm_name,
                       save_dir, save_debug=False, batch_scene_tree=True,
                       force_scene_tree=False):
    """
    After segmentation, build the scene tree from compact SAM3 mask overlays.
    The object list is taken strictly from filenames under segemented_obj/ (per mask id), skipping the_floor_*.

    Args:
        image_path: path to the source image
        seg_obj_dir: segemented_obj/ directory defining the full object list
        agent_output_dir: directory of agent outputs (contains *_pred.json)
        image_stem: image filename without extension
        llm_name: VLM backend name (used in output filenames)
        save_dir: directory in which to save the scene tree
        save_debug: if True, persist debug_scene_tree/ on disk (per-object overlays
            and the per-message VLM input images). When False, overlays stay
            in-memory only and no debug folder is created.
    """
    debug_dir = os.path.join(save_dir, "debug_scene_tree") if save_debug else None
    if debug_dir is not None:
        os.makedirs(debug_dir, exist_ok=True)

    # Step 1: scan segemented_obj/ for the full per-id object list, skipping the_floor_*
    seg_files = sorted(glob(os.path.join(seg_obj_dir, "*.png")))
    obj_ids = []
    for f in seg_files:
        obj_id = os.path.splitext(os.path.basename(f))[0]
        if obj_id.startswith("the_floor"):
            continue
        obj_ids.append(obj_id)

    tree_path = os.path.join(save_dir, "scene_tree.json")
    manifest_path = os.path.join(save_dir, "scene_tree_manifest.json")
    fingerprint = _scene_tree_fingerprint(
        image_path, seg_files, llm_name, batch_scene_tree
    )
    if not force_scene_tree and os.path.exists(tree_path) and os.path.exists(manifest_path):
        try:
            with open(manifest_path) as file_obj:
                manifest = json.load(file_obj)
            with open(tree_path) as file_obj:
                cached_tree = json.load(file_obj)
            cached_children = [edge.get("child") for edge in cached_tree.get("edges", [])]
            if manifest.get("fingerprint") == fingerprint \
                    and cached_tree.get("nodes") == obj_ids \
                    and sorted(cached_children) == sorted(obj_ids):
                logger.info(
                    f"Reusing cached scene tree: {tree_path} "
                    f"(fingerprint {fingerprint[:12]})"
                )
                return cached_tree
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            logger.info(f"Scene-tree cache is invalid and will be rebuilt: {exc}")

    logger.info(f"\n{'='*40}")
    logger.info(f"Constructing scene tree for {len(obj_ids)} objects (from segemented_obj)...")

    # Parse obj_id -> (prompt_safe, mask_idx), grouped by prompt_safe
    def parse_obj_id(obj_id):
        parts = obj_id.rsplit("_", 1)
        return parts[0], int(parts[1])

    groups = defaultdict(list)
    for obj_id in obj_ids:
        prompt_safe, mask_idx = parse_obj_id(obj_id)
        groups[prompt_safe].append((obj_id, mask_idx))

    # Step 2: generate or reuse an overlay per obj_id, compute mask centers
    obj_data = {}
    pred_cache = {}
    with Image.open(image_path) as scene_image:
        scene_overlay_base = resize_image_for_vlm(scene_image)

    for prompt_safe, items in groups.items():
        base_filename = f"{image_stem}_{prompt_safe}_agent_{llm_name}"
        json_path = os.path.join(agent_output_dir, f"{base_filename}_pred.json")
        if not os.path.exists(json_path):
            logger.info(f"  Scene tree: no pred JSON for '{prompt_safe}', skipping")
            continue

        if json_path not in pred_cache:
            with open(json_path) as f:
                pred_cache[json_path] = json.load(f)
        pred_json = pred_cache[json_path]
        h, w = int(pred_json["orig_img_h"]), int(pred_json["orig_img_w"])

        for obj_id, mask_idx in items:
            if mask_idx >= len(pred_json.get("pred_masks", [])):
                logger.info(f"  Scene tree: mask_idx {mask_idx} out of range for '{obj_id}', skipping")
                continue

            # Decode the single mask once for both its center and compact overlay.
            rle_str = pred_json["pred_masks"][mask_idx]
            rle = {"counts": rle_str, "size": [h, w]}
            binary = mask_utils.decode(rle)
            ys, xs = np.where(binary > 0)
            center = (float(xs.mean()), float(ys.mean())) if len(xs) > 0 else (w / 2.0, h / 2.0)
            floor_check_mask = np.asarray(
                Image.fromarray((binary > 0).astype(np.uint8) * 255).resize(
                    (256, 256), Image.Resampling.NEAREST
                )
            ) > 0

            # Reuse an existing compact debug overlay or render directly at VLM
            # resolution. The old path rendered two 4K images per object first.
            overlay_path = (
                os.path.join(debug_dir, f"overlay_{obj_id}.png")
                if debug_dir is not None else None
            )
            overlay = None
            if overlay_path is not None and os.path.exists(overlay_path):
                try:
                    overlay = Image.open(overlay_path).convert("RGB")
                except Exception:
                    os.remove(overlay_path)  # truncated/corrupted, delete and regenerate
                    overlay = None
            if overlay is None:
                try:
                    overlay = _make_compact_mask_overlay(scene_overlay_base, binary)
                    if overlay_path is not None:
                        overlay.save(overlay_path)
                except Exception as e:
                    logger.info(f"  Scene tree: failed to visualize '{obj_id}': {e}")
                    continue

            # Existing debug overlays from older runs may still be full resolution.
            overlay = resize_image_for_vlm(overlay)

            obj_data[obj_id] = {
                "overlay": overlay,
                "center": center,
                "floor_check_mask": floor_check_mask,
            }

    # Step 3: query the VLM in one batch, with per-object fallback if needed
    available_parents = ["floor", "wall", "ceiling", "floor-wall"] + obj_ids
    edges = []
    result_lines = []

    # Preload the_floor mask to detect the floor-copy case (and avoid self-loops)
    floor_mask = None
    floor_png = os.path.join(seg_obj_dir, "the_floor.png")
    if os.path.exists(floor_png):
        with Image.open(floor_png) as floor_image:
            floor_channel = floor_image.getchannel("A") \
                if "A" in floor_image.getbands() else floor_image.convert("L")
            floor_mask = np.asarray(
                floor_channel.resize((256, 256), Image.Resampling.NEAREST)
            ) > 0

    def _iou_with_floor(obj_id):
        if floor_mask is None or obj_id not in obj_data:
            return 0.0
        obj_mask = obj_data[obj_id]["floor_check_mask"]
        inter = np.logical_and(floor_mask, obj_mask).sum()
        union = np.logical_or(floor_mask, obj_mask).sum()
        return float(inter) / float(union) if union > 0 else 0.0

    batch_edges = {}
    if batch_scene_tree:
        batch_targets = [
            obj_id for obj_id in obj_ids
            if obj_id in obj_data and _iou_with_floor(obj_id) <= 0.99
        ]
        if batch_targets:
            logger.info(
                f"  Querying parents for {len(batch_targets)} objects in one batched VLM request..."
            )
            try:
                batch_edges = _query_scene_tree_batch(
                    image_path, batch_targets, obj_data, available_parents, debug_dir
                )
                logger.info(
                    f"  Batch scene-tree parsed {len(batch_edges)}/{len(batch_targets)} edges"
                )
            except Exception as exc:
                logger.warning(
                    f"  Batch scene-tree request failed; falling back to per-object requests: {exc}"
                )

    for obj_id in obj_ids:
        if obj_id not in obj_data:
            edges.append({"child": obj_id, "parent": "floor", "relation": "on"})
            result_lines.append(f"{obj_id} -> floor | on")
            continue

        # Self-loop guard: if the mask is identical to the_floor, force parent=floor.
        if _iou_with_floor(obj_id) > 0.99:
            logger.info(f"  ⚠️  '{obj_id}' mask is identical to the_floor (floor-copy fallback) — forcing parent=floor")
            edges.append({"child": obj_id, "parent": "floor", "relation": "on", "type": "movable"})
            result_lines.append(f"{obj_id} -> floor | on | movable  [floor-copy forced]")
            continue

        if obj_id in batch_edges:
            edge = batch_edges[obj_id]
            edges.append(edge)
            result_lines.append(
                f'{obj_id} -> {edge["parent"]} | {edge["relation"]} | {edge["type"]}'
            )
            continue

        # Find nearby objects by distance
        cx, cy = obj_data[obj_id]["center"]
        distances = []
        for other in obj_ids:
            if other == obj_id or other not in obj_data:
                continue
            ox, oy = obj_data[other]["center"]
            dist = ((cx - ox)**2 + (cy - oy)**2)**0.5
            distances.append((dist, other))
        distances.sort()
        nearby = [n for _, n in distances[:5]]

        # Build VLM message: original image + current obj overlay + nearby obj overlays
        content = [
            {"type": "image", "image": image_path},
            {"type": "text", "text": "Full scene image above.\n\n"},
            {"type": "image", "image": obj_data[obj_id]["overlay"]},
            {"type": "text", "text": f'Current object: "{obj_id}" (colored mask overlay above)\n\nNearby objects:\n'},
        ]
        for nearby_id in nearby:
            content.append({"type": "image", "image": obj_data[nearby_id]["overlay"]})
            content.append({"type": "text", "text": f'Nearby object: "{nearby_id}"\n'})

        parents_str = ", ".join(f'"{p}"' for p in available_parents if p != obj_id)
        content.append({"type": "text", "text": (
            f'\nDetermine what supports or holds "{obj_id}" in this scene.\n'
            f"The parent MUST be one of: {parents_str}\n\n"
            "Rules:\n"
            '- "on": object rests on the TOPMOST surface of the parent — the parent does not extend above the object\n'
            '- "inside": object rests on an INTERMEDIATE horizontal surface of the parent — the parent\'s structure\n'
            "  extends above the object (e.g. item on a countertop that is part of a merged cabinet system which also has upper cabinets; item stored inside a basket, box, or drawer unit).\n"
            "  Judge by looking at the WHOLE parent mask as a single object, not individual parts.\n"
            '- "supported-by": object receives partial or multi-point support and can overlap the parent vertically;\n'
            "  use it when bottom-to-top stacking would destroy the visible pose (e.g. a body supported by furniture\n"
            "  or an object cradled by a stand).\n"
            '- "attach": mounted/fixed to a surface — use "wall" or "ceiling" as parent\n'
            "  - wall attach: picture frame, window, wall shelf, wall-mounted TV\n"
            "  - ceiling attach: hanging lamp, ceiling fan\n"
            '- "hang": object is draped/hung from a rod, rail, or hook — use the rod/rail as parent\n'
            "  - e.g. curtains hang from curtain rod, coats hang from hook\n"
            "- If object sits on another object, parent is that object, not floor\n"
            "- If object rests on floor AND is fixed against/to a wall (radiator, large cabinet, built-in unit):\n"
            '  use parent "floor-wall", relation "on-attach"\n\n'
            "Type rules:\n"
            '- "fixed": immovable furniture (cabinets, shelves, radiators, built-in units, bookcases) or anything attached to wall/ceiling\n'
            '- "movable": can be picked up or pushed (chairs, cups, books, toys, etc.)\n\n'
            "Physics-role rules (separate from type):\n"
            '- "dynamic": free rigid object that should settle under gravity\n'
            '- "kinematic": posed, articulated, or multi-contact object whose observed pose must be retained\n'
            '- "fixed": immovable or anchored object\n\n'
            f"Output ONLY one line:\n"
            f"{obj_id} -> parent_name | relation | type | physics_role\n"
        )})

        if debug_dir is not None:
            img_idx = 0
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image":
                    debug_image = item["image"]
                    debug_path = os.path.join(
                        debug_dir, f"msg_{obj_id}_{img_idx}.png"
                    )
                    if isinstance(debug_image, (str, os.PathLike)):
                        with Image.open(debug_image) as source_image:
                            source_image.convert("RGB").save(debug_path)
                    else:
                        debug_image.save(debug_path)
                    img_idx += 1

        messages = [{"role": "user", "content": content}]
        logger.info(f"  Batch result missing/invalid; querying parent for: '{obj_id}'...")
        response = generate_vlm_response(messages)
        logger.info(f"    Response: {response.strip()}")

        # Parse format: obj_id -> parent | relation | type | physics_role
        found = False
        for line in response.strip().split("\n"):
            line = line.strip()
            if "->" not in line:
                continue
            parts = line.split("->")
            if len(parts) != 2:
                continue
            rest = parts[1].strip()
            fields = [f.strip() for f in rest.split("|")]
            if len(fields) >= 3:
                parent, relation, obj_type = fields[0], fields[1], fields[2]
            elif len(fields) == 2:
                parent, relation = fields[0], fields[1]
                obj_type = "movable"
            else:
                parent = fields[0]
                relation = "on"
                obj_type = "movable"
            physics_role = fields[3].lower().strip() if len(fields) >= 4 else ""
            # normalize type
            obj_type = obj_type.lower().strip()
            if obj_type not in ("fixed", "movable"):
                obj_type = "movable"
            if physics_role not in _SCENE_TREE_PHYSICS_ROLES:
                if obj_type == "fixed" or relation in {"attach", "hang", "on-attach"}:
                    physics_role = "fixed"
                elif relation == "supported-by":
                    physics_role = "kinematic"
                else:
                    physics_role = "dynamic"
            parent = parent.strip()
            # Case-insensitive match
            parent_lower = parent.lower()
            matched = next((p for p in available_parents if p.lower() == parent_lower), None)
            if matched is None:
                logger.info(f"    ⚠️ Invalid parent '{parent}' (not in segemented_obj or roots), fallback to floor")
                parent = "floor"
                relation = "on"
                obj_type = "movable"
                physics_role = "dynamic"
            else:
                parent = matched  # use canonical casing
            edges.append({
                "child": obj_id,
                "parent": parent,
                "relation": relation,
                "type": obj_type,
                "physics_role": physics_role,
            })
            result_lines.append(
                f"{obj_id} -> {parent} | {relation} | {obj_type} | {physics_role}"
            )
            found = True
            break
        if not found:
            edges.append({"child": obj_id, "parent": "floor", "relation": "on", "type": "movable"})
            result_lines.append(f"{obj_id} -> floor | on | movable")

    # Save aggregated scene tree as JSON
    os.makedirs(save_dir, exist_ok=True)
    scene_tree = {
        "roots": ["floor", "wall", "ceiling", "floor-wall"],
        "nodes": obj_ids,
        "edges": edges,
    }
    with open(tree_path, 'w') as f:
        json.dump(scene_tree, f, indent=2, ensure_ascii=False)
    with open(manifest_path, "w") as file_obj:
        json.dump({
            "version": _SCENE_TREE_PROMPT_VERSION,
            "fingerprint": fingerprint,
            "backend": llm_name,
            "mode": "batch" if batch_scene_tree else "per-object",
            "object_count": len(obj_ids),
        }, file_obj, indent=2)
    logger.info(f"Scene tree saved to {tree_path} ({len(edges)} edges)")

    return scene_tree


def _pick_most_floor_like(seg_obj_dir, image_rgb):
    """
    Called when the floor agent fails after 10 rounds. Picks the most floor-like mask
    from the existing segemented_obj/ masks, scored as x_span * bottom_pos * bbox_area_frac.
    The selected mask is COPIED as the_floor (the original PNG is kept).
    Note: the_floor.png will be identical to one object mask; dedup and scene-tree logic handle this case.
    """
    seg_files = sorted(glob(os.path.join(seg_obj_dir, "*.png")))
    if not seg_files:
        return None

    if hasattr(image_rgb, 'shape'):
        img_h, img_w = image_rgb.shape[:2]
    else:
        img_w, img_h = image_rgb.size

    best_score, best_name, best_mask = -1.0, None, None
    for fpath in seg_files:
        name = os.path.splitext(os.path.basename(fpath))[0]
        if name == "the_floor" or name.startswith("the_floor_"):
            continue
        m = np.array(Image.open(fpath).convert("L")) > 0
        if m.sum() == 0:
            continue
        # Use bounding box rather than pixel count: an occluded floor has sparse pixels but a wide bbox
        rows = np.where(m.any(axis=1))[0]
        cols = np.where(m.any(axis=0))[0]
        y_min, y_max = rows[0], rows[-1]
        x_min, x_max = cols[0], cols[-1]
        x_span = (x_max - x_min + 1) / img_w       # horizontal extent: a floor should span the image
        bottom_pos = y_max / img_h                  # bbox-bottom location (lower in image = better)
        bbox_area_frac = ((x_max - x_min + 1) * (y_max - y_min + 1)) / (img_w * img_h)
        score = x_span * bottom_pos * bbox_area_frac
        if score > best_score:
            best_score, best_name, best_mask = score, name, m

    if best_name is None:
        return None

    logger.info(f"    ⚠️  floor fallback (copy): using '{best_name}' as floor (score={best_score:.2f}), original PNG kept")

    # Encode best_mask as RLE and write a synthetic pred.json
    img_h, img_w = best_mask.shape
    rle = mask_utils.encode(np.asfortranarray(best_mask.astype(np.uint8)))
    result_json = {
        "orig_img_h": img_h,
        "orig_img_w": img_w,
        "pred_boxes": [],
        "pred_masks": [rle["counts"].decode("utf-8") if isinstance(rle["counts"], bytes) else rle["counts"]],
        "pred_scores": [1.0],
    }
    return result_json


def dedup_seg_masks(seg_obj_dir, agent_output_dir, img_output_dir, image_stem, llm_name,
                    iou_thresh=0.3, overlap_thresh=0.8):
    """
    Pairwise-deduplicate masks inside segemented_obj/:
    - if one side is the_floor, keep the_floor and drop the other;
    - otherwise keep the alphabetically-earlier filename.
    """
    seg_files = sorted(glob(os.path.join(seg_obj_dir, "*.png")))
    names = [os.path.splitext(os.path.basename(f))[0] for f in seg_files]

    def is_floor(n):
        return n == "the_floor" or n.startswith("the_floor_")

    # Pairwise overlap at the original 13 MP resolution is needlessly expensive
    # and retains hundreds of MiB for scenes with many instances. All masks share
    # the same source resolution, so a common 512x512 nearest-neighbor thumbnail
    # preserves their relative overlap while making the O(n^2) pass inexpensive.
    masks = {}
    for name, fpath in zip(names, seg_files):
        with Image.open(fpath) as image:
            channel = image.getchannel("A") \
                if "A" in image.getbands() else image.convert("L")
            m = np.asarray(
                channel.resize((512, 512), Image.Resampling.NEAREST)
            ) > 0
        if m.sum() > 0:
            masks[name] = m

    removed = set()
    name_list = list(masks.keys())
    for i in range(len(name_list)):
        a = name_list[i]
        if a in removed:
            continue
        for j in range(i + 1, len(name_list)):
            b = name_list[j]
            if b in removed:
                continue
            ma, mb = masks[a], masks[b]
            intersection = np.logical_and(ma, mb).sum()
            if intersection == 0:
                continue
            union = np.logical_or(ma, mb).sum()
            iou = intersection / union
            overlap_a = intersection / ma.sum()
            overlap_b = intersection / mb.sum()
            if iou > iou_thresh or max(overlap_a, overlap_b) > overlap_thresh:
                # If one is the floor, it is likely the copy-fallback case: warn but do not delete
                if is_floor(a) != is_floor(b):
                    floor_name = a if is_floor(a) else b
                    obj_name = b if is_floor(a) else a
                    logger.info(f"  ⚠️  dedup WARNING: '{obj_name}' and '{floor_name}' have high overlap "
                          f"(IoU={iou:.2f}) — floor is likely a copy fallback, keeping both")
                    continue
                # Choose which to drop: two floor variants -> later one; otherwise the alphabetically later name
                to_remove = b if not is_floor(b) else a
                if is_floor(a) and is_floor(b):
                    to_remove = b
                logger.info(f"  dedup: removing '{to_remove}' (IoU={iou:.2f}, "
                      f"overlap_a={overlap_a:.2f}, overlap_b={overlap_b:.2f})")
                removed.add(to_remove)

    if not removed:
        return

    # Delete files
    for seg_name in removed:
        seg_png = os.path.join(seg_obj_dir, f"{seg_name}.png")
        if os.path.exists(seg_png):
            os.remove(seg_png)
        for extra in [
            os.path.join(agent_output_dir, f"{image_stem}_{seg_name}_agent_{llm_name}_pred.json"),
            os.path.join(img_output_dir, "overlay", f"{seg_name}.jpg"),
        ]:
            if os.path.exists(extra):
                os.remove(extra)
    logger.info(f"  dedup removed {removed}")


def load_cached_results(img_output_dir, agent_output_dir, image_stem, llm_name):
    """
    Try to load existing outputs to skip re-running the agent.

    Returns:
        objects: list[str] or None (None means no cache hit)
        cached_objects: set of object names that already have _pred.json
    """
    # Load the existing object list
    obj_list_path = os.path.join(img_output_dir, "scene_object_lists.txt")
    if not os.path.exists(obj_list_path):
        return None, set()

    with open(obj_list_path, "r") as f:
        objects = [line.strip() for line in f if line.strip()]
    if not objects:
        return None, set()

    # Always append "the floor" in memory (not written to disk)
    if "the floor" not in objects:
        objects.append("the floor")

    # Find objects that already have *_pred.json AND a matching PNG in segemented_obj/
    seg_obj_dir = os.path.join(img_output_dir, "segemented_obj")
    cached = set()
    for name in objects:
        prompt_safe = name.replace("/", "_").replace(" ", "_")
        base_filename = f"{image_stem}_{prompt_safe}_agent_{llm_name}"
        json_path = os.path.join(agent_output_dir, f"{base_filename}_pred.json")
        if not os.path.exists(json_path):
            continue
        # Floor is intentionally stored as the_floor.png; other objects are
        # per-instance files such as chair_000.png.
        if name == "the floor":
            has_png = os.path.exists(os.path.join(seg_obj_dir, "the_floor.png"))
        else:
            has_png = any(
                True for _ in glob(os.path.join(seg_obj_dir, f"{prompt_safe}_*.png"))
            )
        if has_png:
            cached.add(name)

    return objects, cached


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_folder",
        type=str,
        default=None,
        help="Path to a single image file, or a folder of images"
    )
    parser.add_argument(
        "--image_list",
        type=str,
        default=None,
        help="Path to a txt file where each line is either an image path or a folder path"
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="output",
        help="Root output directory. Stage-1 output lands at "
             "{output_folder}/{image_stem}/stage1/ (default: output)"
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default=None,
        help="Override the per-image output directory name for a single image. "
             "Only valid when --image_folder points to one image file."
    )
    parser.add_argument(
        "--vlm_backend",
        type=str,
        default="gemini",
        choices=["gpt4o", "gemini"],
        help="VLM backend to use: gpt4o (OpenAI API) or gemini (Google API)"
    )
    parser.add_argument(
        "--vlm_prompt_file",
        type=str,
        default="list_objects.txt",
        help="VLM prompt file. Relative paths resolve against rest3d/prompts/; "
             "absolute paths are used as-is. (default: list_objects.txt)"
    )
    parser.add_argument(
        "--vlm_max_image_edge",
        type=int,
        default=int(os.getenv("REST3D_VLM_MAX_IMAGE_EDGE", "1536")),
        help="Longest image edge uploaded to the VLM; SAM3 still uses the original "
             "resolution. Set to 0 to disable resizing. (default: 1536)"
    )
    parser.add_argument(
        "--save_debug",
        action="store_true",
        help="If set, persist the debug_scene_tree/ folder (per-object overlays + "
             "VLM input snapshots). Off by default — overlays stay in memory."
    )
    parser.add_argument(
        "--legacy_scene_tree_per_object",
        action="store_true",
        help="Disable batched scene-tree inference and query the VLM once per mask."
    )
    parser.add_argument(
        "--force_scene_tree",
        action="store_true",
        help="Rebuild scene_tree.json even when its input fingerprint is unchanged."
    )
    args = parser.parse_args()

    # List of (abs_image_path, output_name) tuples
    images_list = []

    if args.output_name:
        if args.image_list:
            parser.error("--output_name cannot be used with --image_list")
        if not args.image_folder or not os.path.isfile(args.image_folder):
            parser.error("--output_name requires --image_folder to point to one image file")
        if os.path.basename(os.path.normpath(args.output_name)) != args.output_name \
                or args.output_name in {".", ".."}:
            parser.error("--output_name must be a single directory name, not a path")

    if args.image_list:
        with open(args.image_list, "r") as f:
            lines = [l.strip() for l in f if l.strip()]
        for line in lines:
            parts = [p.strip() for p in line.split(",", 1)]
            if len(parts) != 2:
                logger.info(f"Warning: skipping malformed line (expected 'path, name'): {line}")
                continue
            img_path, output_name = parts
            if not os.path.isfile(img_path):
                logger.info(f"Warning: image not found, skipping: {img_path}")
                continue
            images_list.append((img_path, output_name))
    elif args.image_folder:
        image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}
        if os.path.isfile(args.image_folder):
            found = [args.image_folder]
        else:
            found = []
            for ext in image_extensions:
                found.extend(glob(os.path.join(args.image_folder, f"*{ext}")))
                found.extend(glob(os.path.join(args.image_folder, f"*{ext.upper()}")))
        for img_path in sorted(found):
            output_name = args.output_name or os.path.splitext(os.path.basename(img_path))[0]
            images_list.append((img_path, output_name))
    else:
        raise ValueError("Must provide either --image_folder or --image_list")

    output_root = args.output_folder
    os.makedirs(output_root, exist_ok=True)

    set_vlm_backend(args.vlm_backend)
    set_vlm_image_max_edge(args.vlm_max_image_edge)

    # Build agent components (SAM 3 segmentor + VLM backend)
    llm_config, send_generate_request, call_sam_service = build_agent_components(args)
    llm_name = llm_config["name"]

    for img_path, output_name in images_list:
        image_started = time.monotonic()
        get_vlm_stats(reset=True)
        abs_img_path = os.path.abspath(img_path)
        image_stem = os.path.splitext(os.path.basename(img_path))[0]
        # Per-image, per-stage output: {output_root}/{image_stem}/stage1/
        img_output_dir = os.path.join(output_root, output_name, "stage1")
        agent_output_dir = os.path.join(img_output_dir, "segment_agent_out")
        seg_obj_dir = os.path.join(img_output_dir, "segemented_obj")
        os.makedirs(seg_obj_dir, exist_ok=True)

        # Capture this image's log into a timestamped stage1_log_<YYYYMMDD_HHMMSS>.txt
        _ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        _log_fh = attach_file_handler(logger, os.path.join(img_output_dir, f"stage1_log_{_ts}.txt"))

        logger.info(f"\n{'='*60}")
        logger.info(f"Running stage1: scene tree construction, Processing image: {img_path}")
        logger.info(f"VLM backend set to: {args.vlm_backend}")
        logger.info(f"VLM upload max image edge: {args.vlm_max_image_edge or 'original'}")
        logger.info(
            "Scene-tree VLM mode: %s",
            "per-object" if args.legacy_scene_tree_per_object else "batch",
        )
        # Show the resolved prompt path (relative names live under rest3d/prompts/)
        _prompt_display = args.vlm_prompt_file if os.path.isabs(args.vlm_prompt_file) \
            else os.path.join("rest3d/prompts", args.vlm_prompt_file)
        logger.info(f"Stage1 prompt: {_prompt_display}")
        logger.info(f"{'='*60}")

        # Lazy load: check for cached results first
        cached_objects, cached_set = load_cached_results(
            img_output_dir, agent_output_dir, image_stem, llm_name
        )
        all_segmentations_cached = bool(
            cached_objects and len(cached_set) == len(cached_objects)
        )

        if all_segmentations_cached:
            # All objects already have agent output; skip to scene-tree analysis
            objects = cached_objects
            logger.info(f"Loaded {len(objects)} cached objects, skipping agent segmentation")
        else:
            # Step 1: VLM generates the object list
            if cached_objects:
                objects = cached_objects
                logger.info(f"Loaded {len(objects)} objects from cache, {len(cached_set)}/{len(objects)} already segmented")
            else:
                objects = analyze_scene_object_lists(abs_img_path, save_dir=img_output_dir, vlm_prompt_file=args.vlm_prompt_file)
                if not objects:
                    logger.info(f"VLM identified no objects; skipping {img_path}")
                    continue
                logger.info(f"Identified {len(objects)} objects: {objects}")

            # Load the image so we can save masks
            image = load_image(img_path, backend="cv2", image_format="bgr")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Step 2: run_single_image_inference for each object
            for obj_prompt in objects:
                if obj_prompt in cached_set:
                    logger.info(f"\n  Skipping '{obj_prompt}' (cached)")
                    continue
                logger.info(f"\n  Agent segmenting: '{obj_prompt}'")

                if obj_prompt == "the floor":
                    # Same VLM+SAM agent path as other objects (max 10 rounds)
                    prompt_safe_floor = "the_floor"
                    base_floor = f"{image_stem}_{prompt_safe_floor}_agent_{llm_name}"
                    floor_json_path = os.path.join(agent_output_dir, f"{base_floor}_pred.json")
                    floor_png_path = os.path.join(seg_obj_dir, "the_floor.png")

                    run_single_image_inference(
                        abs_img_path, obj_prompt, llm_config,
                        send_generate_request, call_sam_service,
                        output_dir=agent_output_dir, debug=args.save_debug,
                    )

                    # Check whether the agent succeeded
                    agent_succeeded = False
                    if os.path.exists(floor_json_path):
                        masks, _ = decode_agent_masks(floor_json_path)
                        if len(masks) > 0:
                            save_seg_obj(image, masks[0], out_path=floor_png_path)
                            logger.info("    floor agent succeeded: saved to the_floor.png")
                            agent_succeeded = True

                    if not agent_succeeded:
                        # Floor agent failed in 10 rounds: try SAM3 synonyms first, then fall back to copy
                        FLOOR_SYNONYMS = [
                            "carpet", "rug", "floor mat", "flooring",
                            "hardwood floor", "tile floor", "ground",
                        ]
                        synonym_succeeded = False
                        for synonym in FLOOR_SYNONYMS:
                            logger.info(f"    🔄 floor synonym retry: '{synonym}'")
                            # call_sam_service builds the save path internally; reconstruct it for the check
                            prompt_safe_syn = synonym.replace("/", "_").replace(" ", "_")
                            # output_folder_path subdir name is image_path.replace("/", "-")
                            syn_sub = abs_img_path.replace("/", "-")
                            syn_json = os.path.join(agent_output_dir, "sam_synonym",
                                                    syn_sub, f"{prompt_safe_syn}.json")
                            os.makedirs(os.path.dirname(syn_json), exist_ok=True)
                            call_sam_service(
                                abs_img_path, synonym,
                                output_folder_path=os.path.join(agent_output_dir, "sam_synonym"),
                            )
                            if os.path.exists(syn_json):
                                syn_masks, _ = decode_agent_masks(syn_json)
                                if len(syn_masks) > 0:
                                    # Write the synonym result into floor_json_path (keep format consistent)
                                    with open(syn_json) as f:
                                        result_syn = json.load(f)
                                    result_syn["text_prompt"] = "the floor"
                                    result_syn["image_path"] = abs_img_path
                                    json.dump(result_syn, open(floor_json_path, "w"), indent=4)
                                    save_seg_obj(image, syn_masks[0], out_path=floor_png_path)
                                    logger.info(f"    ✅ floor synonym '{synonym}' succeeded: saved to the_floor.png")
                                    synonym_succeeded = True
                                    break

                        if not synonym_succeeded:
                            # All synonyms failed -> copy the most floor-like object mask
                            result_json = _pick_most_floor_like(seg_obj_dir, image)
                            if result_json is not None:
                                result_json["text_prompt"] = "the floor"
                                result_json["image_path"] = abs_img_path
                                json.dump(result_json, open(floor_json_path, "w"), indent=4)
                                masks, _ = decode_agent_masks(floor_json_path)
                                if len(masks) > 0:
                                    save_seg_obj(image, masks[0], out_path=floor_png_path)
                            else:
                                logger.info("    ⚠️  floor: agent+synonyms failed and no object masks available, skipping")
                    continue  # Skip Step 3: floor already handled above
                else:
                    run_single_image_inference(
                        abs_img_path, obj_prompt, llm_config,
                        send_generate_request, call_sam_service,
                        output_dir=agent_output_dir, debug=args.save_debug,
                    )

                # Step 3: decode the RLE mask from the agent JSON and save it
                prompt_for_filename = obj_prompt.replace("/", "_").replace(" ", "_")
                base_filename = f"{image_stem}_{prompt_for_filename}_agent_{llm_name}"
                output_json_path = os.path.join(agent_output_dir, f"{base_filename}_pred.json")

                if not os.path.exists(output_json_path):
                    # Agent failed to pick a mask within 10 rounds; skip without saving a wrong result
                    logger.info(f"    Warning: agent failed to segment '{obj_prompt}'; skipping (no fallback mask written)")
                    continue

                masks, _ = decode_agent_masks(output_json_path)
                logger.info(f"    Found {len(masks)} masks")

                # Decide how to handle multiple masks based on object type:
                # - Fixed installations (cabinet/shelf/radiator...): union all into 1
                # - Everything else: trust the agent — keep all masks it selected
                _FIXED_KW = ["cabinet", "shelf", "shelving", "bookcase", "hutch",
                             "dresser", "wardrobe", "closet", "system", "radiator",
                             "fireplace", "window", "blinds", "built-in"]
                _name = obj_prompt.lower() + " "
                _is_fixed = any(kw in _name for kw in _FIXED_KW)

                if len(masks) > 1:
                    if _is_fixed:
                        # Union all masks → one complete mask for the installation
                        combined = np.zeros_like(masks[0])
                        for m in masks:
                            combined = np.logical_or(combined, m > 0.5).astype(np.float32)
                        masks = [combined]
                        logger.info("    Merged into 1 combined mask (fixed installation)")
                    else:
                        # Trust the agent: it selected N masks intentionally → keep all
                        logger.info(f"    Keeping {len(masks)} masks (agent selection)")

                prompt_safe = prompt_for_filename

                vis_overlay_masks_path = os.path.join(img_output_dir, "overlay", f"{prompt_safe}.jpg")
                os.makedirs(os.path.dirname(vis_overlay_masks_path), exist_ok=True)
                visualize_masks_on_image_cv2(image, masks, out_path=vis_overlay_masks_path)

                vis_each_seg_obj_dir = os.path.join(img_output_dir, "segemented_obj")
                os.makedirs(vis_each_seg_obj_dir, exist_ok=True)
                for mask_idx, mask in enumerate(masks):
                    vis_each_seg_obj_path = os.path.join(
                        vis_each_seg_obj_dir,
                        f"{prompt_safe}_{mask_idx:03d}.png"
                    )
                    save_seg_obj(image, mask, out_path=vis_each_seg_obj_path)

        # Step 4: pairwise-deduplicate only after segmentation has changed. A
        # fully cached run already processed this exact set on its first pass.
        if all_segmentations_cached:
            logger.info("Skipping mask deduplication: all segmentations are cached")
        else:
            dedup_seg_masks(
                seg_obj_dir, agent_output_dir, img_output_dir, image_stem, llm_name
            )

        # Step 5: analyze the scene tree (object list comes from segemented_obj/)
        logger.info(f"\n{'='*80}")
        logger.info(f"Scene tree construction: {output_name}")
        logger.info(f"{'='*80}")
        analyze_scene_tree(
            abs_img_path,
            seg_obj_dir,
            agent_output_dir,
            image_stem,
            llm_name,
            img_output_dir,
            save_debug=args.save_debug,
            batch_scene_tree=not args.legacy_scene_tree_per_object,
            force_scene_tree=args.force_scene_tree,
        )

        logger.info(f"\n{'='*40}")
        logger.info(f"Saving stage1 object masks + scene tree to {img_output_dir}")
        vlm_stats = get_vlm_stats()
        logger.info(
            "Stage1 timing: %.1fs total; VLM: %d request(s), %d image(s), %.2f MiB, %.1fs",
            time.monotonic() - image_started,
            vlm_stats["requests"],
            vlm_stats["images"],
            vlm_stats["image_bytes"] / (1024 * 1024),
            vlm_stats["elapsed_seconds"],
        )
        logger.info(
            "VLM file-image cache: %d hit(s), %d miss(es)",
            vlm_stats["file_cache_hits"], vlm_stats["file_cache_misses"]
        )
        logger.info(f"{'='*40}")

        logger.removeHandler(_log_fh)
        _log_fh.close()


if __name__ == "__main__":
    main()
