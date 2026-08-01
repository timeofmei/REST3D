"""Replay optimized scene poses in Isaac Gym with optional MP4 and viser output."""

# Isaac Gym must be imported before torch
from isaacgym import gymapi, gymtorch

import argparse
import json
import logging
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from rest3d.models.cem_opt import (
    create_sim_and_viewer,
    torch_cuda_pipeline_supported,
)
from rest3d.sim.replay_scene import gym_states_xyzw_to_rest, load_replay_scene
from rest3d.sim.stability import evaluate_replay_stability

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
_logger = logging.getLogger("replay_scene")


def config_args():
    ap = argparse.ArgumentParser(description="Replay optimized scene poses in Isaac Gym")
    ap.add_argument("--scene-tree", "--scene_tree", dest="scene_tree", required=True,
                    help="Path to scene_tree.json (e.g. output/.../stage2/scene_tree.json)")
    ap.add_argument("--scene-dir", "--scene_dir", dest="scene_dir", required=True,
                    help="Read-only Stage 3 scene containing obj_files and urdf_files")
    ap.add_argument("--output-dir", "--output_dir", dest="output_dir", required=True,
                    help="New replay result directory; fails if it already exists")
    ap.add_argument("--settle-steps", "--settle_steps", dest="settle_steps", type=int, default=120,
                    help="Physics settle steps")
    ap.add_argument("--headless", action="store_true",
                    help="Run without interactive viewer")
    ap.add_argument("--cpu", action="store_true",
                    help="Use CPU PhysX and CPU tensors (slower, but works when PyTorch does not support the GPU)")
    ap.add_argument("--save_video", action="store_true", default=True)
    ap.add_argument("--no_save_video", dest="save_video", action="store_false",
                    help="Disable MP4 saving")
    ap.add_argument("--video_path", type=str, default=None,
                    help="Output MP4 path (default: <output_dir>/replay.mp4)")
    ap.add_argument("--viser", action="store_true",
                    help="Launch viser 3D viewer for interactive replay of the physics simulation")
    ap.add_argument("--num_position_iterations", type=int, default=16)
    ap.add_argument("--max_depenetration_velocity", type=float, default=1.0)
    ap.add_argument("--no_vhacd", action="store_true",
                    help="Disable V-HACD decomposition (faster load, rougher collision)")
    ap.add_argument("--vhacd-max-hulls", type=int, default=16,
                    help="Generic V-HACD hull limit applied to every object")
    ap.add_argument("--linear_damping",  type=float, default=0.3)
    ap.add_argument("--angular_damping", type=float, default=0.3)
    ap.add_argument("--expected-object-count", type=int)
    ap.add_argument("--stability-evaluation-steps", type=int, default=60)
    ap.add_argument("--position-stability-threshold", type=float, default=0.1)
    ap.add_argument("--rotation-stability-threshold", type=float, default=0.1)
    ap.add_argument("--early-window-steps", type=int, default=10)
    ap.add_argument("--terminal-window-steps", type=int, default=10)
    ap.add_argument("--require-stable", action="store_true")
    ap.add_argument("--fps",           type=int, default=30,   help="Video FPS (default: 30)")
    ap.add_argument("--cam_width",     type=int, default=1920, help="Isaac Gym camera width")
    ap.add_argument("--cam_height",    type=int, default=1080, help="Isaac Gym camera height")
    ap.add_argument("--record_width",  type=int, default=1280, help="Viser recording width (default: 1280)")
    ap.add_argument("--record_height", type=int, default=720,  help="Viser recording height (default: 720)")
    ap.add_argument("--record_settle_ms", type=int, default=20,
                    help="ms to wait after mesh update before get_render (default: 20)")
    # viser camera params (mirror vr_demo_render.py)
    ap.add_argument("--viser_dist",       type=float, default=None,
                    help="Absolute viser camera distance (m); overrides --viser_dist_scale")
    ap.add_argument("--viser_dist_scale", type=float, default=1.5,
                    help="Viser camera dist = scale * scene radius (default: 1.5)")
    ap.add_argument("--viser_pitch_deg",  type=float, default=25.0,
                    help="Viser camera pitch above horizon in degrees (default: 25)")
    ap.add_argument("--viser_azimuth_deg",type=float, default=270.0,
                    help="Viser camera azimuth in degrees (default: 270 = front)")
    ap.add_argument("--viser_target_y_offset", type=float, default=0.5,
                    help="Y offset of look-at target above scene center (default: 0.5)")
    ap.add_argument("--viser_extra_height", type=float, default=0.0,
                    help="Extra +Y offset on camera position (default: 0)")
    ap.add_argument("--viser_wait",      type=float, default=10.0,
                    help="Seconds to wait for browser to open before loading objects (default: 10)")
    ap.add_argument("--viser_obj_delay", type=float, default=0.3,
                    help="Seconds between loading each object mesh into viser (default: 0.3)")
    args = ap.parse_args()
    args.scene_tree = str(Path(args.scene_tree).expanduser().resolve(strict=True))
    args.scene_dir = str(Path(args.scene_dir).expanduser().resolve(strict=True))
    if args.settle_steps < 1:
        ap.error("--settle-steps must be positive")
    if not 1 <= args.stability_evaluation_steps <= args.settle_steps:
        ap.error("stability evaluation must be within the simulated frame range")
    if args.position_stability_threshold <= 0.0 or args.rotation_stability_threshold <= 0.0:
        ap.error("stability thresholds must be positive")
    if args.early_window_steps < 1 or args.terminal_window_steps < 1:
        ap.error("velocity windows must be positive")
    if args.vhacd_max_hulls < 1:
        ap.error("--vhacd-max-hulls must be positive")
    if args.expected_object_count is not None and args.expected_object_count < 1:
        ap.error("--expected-object-count must be positive")
    output_dir = Path(args.output_dir).expanduser().resolve()
    scene_dir = Path(args.scene_dir)
    if output_dir == scene_dir or scene_dir in output_dir.parents:
        ap.error("--output-dir must be outside the read-only --scene-dir")
    output_dir.mkdir(parents=True, exist_ok=False)
    args.output_dir = str(output_dir)
    file_handler = logging.FileHandler(output_dir / "replay.log", mode="x")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    )
    _logger.addHandler(file_handler)
    if args.video_path is None:
        _name = "replay_viser.mp4" if args.viser else "replay.mp4"
        args.video_path = os.path.join(args.output_dir, _name)
    return args



def init_viser_server(center, radius, args):
    """Start viser server, add ground grid, register camera callback. Return server."""
    try:
        import viser
    except ImportError:
        _logger.warning("[viser] viser not installed — skipping")
        return None

    server = viser.ViserServer()
    server.scene.set_up_direction("+y")

    # ground grid on xz plane at y=0
    try:
        server.scene.add_grid(
            "/grid", width=200.0, height=200.0, plane="xz",
            cell_size=0.5, section_size=1.0,
            infinite_grid=True,
            fade_distance=8.0, fade_strength=1.0, fade_from="camera",
        )
    except Exception:
        server.scene.add_grid("/grid", width=20.0, height=20.0, cell_size=1.0, plane="xz")

    # camera pose from pitch/azimuth/dist
    cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
    gym_dist    = args.viser_dist if args.viser_dist is not None else args.viser_dist_scale * radius
    gym_pitch   = np.radians(args.viser_pitch_deg)
    gym_azimuth = np.radians(args.viser_azimuth_deg)
    gym_horiz   = gym_dist * np.cos(gym_pitch)
    gym_pos = (
        cx + gym_horiz * np.cos(gym_azimuth),
        cy + gym_dist  * np.sin(gym_pitch) + args.viser_extra_height,
        cz + gym_horiz * np.sin(gym_azimuth),
    )
    gym_target = (cx, cy + args.viser_target_y_offset, cz)
    _logger.info(f"[viser] cam pos={tuple(f'{v:.2f}' for v in gym_pos)}  "
                 f"target={tuple(f'{v:.2f}' for v in gym_target)}  dist={gym_dist:.2f}m")

    @server.on_client_connect
    def _on_connect(client):
        client.camera.position = gym_pos
        client.camera.look_at  = gym_target

    _logger.info(f"[viser] Open browser at http://localhost:{server.get_port()}")
    return server


def load_viser_objects(server, obj_paths, name_to_idx, obj_delay=0.3):
    """Add world-frame OBJ meshes to viser at identity pose (vertices already in world frame)."""
    try:
        import trimesh
    except ImportError:
        _logger.warning("[viser] trimesh not installed — skipping object load")
        return {}

    handles = {}
    for name in name_to_idx:
        obj_path = obj_paths.get(name)
        if obj_path is None or not os.path.isfile(obj_path):
            _logger.warning(f"[viser] OBJ not found: {name}")
            continue
        try:
            mesh = trimesh.load(obj_path, force="mesh", process=False)
            if isinstance(mesh, trimesh.Scene):
                if not mesh.geometry:
                    continue
                mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
        except Exception as e:
            _logger.warning(f"[viser] Load failed {name}: {e}")
            continue
        handle = server.scene.add_mesh_trimesh(
            name=f"/scene/{name}", mesh=mesh,
            position=(0.0, 0.0, 0.0), wxyz=(1.0, 0.0, 0.0, 0.0))
        handles[name] = handle
        _logger.info(f"[viser]   added {name}")
        if obj_delay > 0:
            time.sleep(obj_delay)
    _logger.info(f"[viser] {len(handles)} meshes loaded — starting simulation")
    return handles


def _record_viser_mp4(server, handles, name_to_idx, all_states, cam_pos, center, args):
    """Record viser view to MP4. Camera is forced each frame — user interaction has no effect.
    No GUI elements exist during recording (sliders/buttons are added after in _viser_replay)."""
    import cv2
    _logger.info("[viser] Waiting for browser connection to start recording ...")
    while not server.get_clients():
        time.sleep(0.5)
    client = next(iter(server.get_clients().values()))
    _logger.info("[viser] Client connected — recording started (do not interact with camera)")
    _cam_pos   = tuple(float(v) for v in cam_pos)
    _cam_target = tuple(float(v) for v in center)

    rec_w, rec_h = args.record_width, args.record_height
    settle_s = args.record_settle_ms / 1000.0
    os.makedirs(os.path.dirname(os.path.abspath(args.video_path)), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.video_path, fourcc, args.fps, (rec_w, rec_h))
    if not writer.isOpened():
        _logger.warning(f"[viser] VideoWriter failed: {args.video_path}")
        return

    _status_md = server.gui.add_markdown(
        "## ⏺ Recording simulation video...\n"
        "Please wait — camera is locked. Do not move the camera.\n\n"
        f"Saving to: `{os.path.basename(args.video_path)}`"
    )

    _logger.info(f"[viser] Recording {len(all_states)} frames at {rec_w}x{rec_h} -> {args.video_path}")
    _logger.info(f"[viser] Camera locked at pos={_cam_pos}  target={_cam_target}")
    for i, states in enumerate(all_states):
        with server.atomic():
            client.camera.position = _cam_pos
            client.camera.look_at  = _cam_target
            for name, handle in handles.items():
                actor_idx = name_to_idx.get(name)
                if actor_idx is None:
                    continue
                pos  = states[actor_idx, 0:3]
                quat = states[actor_idx, 3:7]
                handle.position = tuple(float(v) for v in pos)
                handle.wxyz = (float(quat[3]), float(quat[0]), float(quat[1]), float(quat[2]))
        server.flush()
        if settle_s > 0:
            time.sleep(settle_s)
        try:
            img = client.camera.get_render(rec_h, rec_w)
        except Exception as e:
            _logger.warning(f"[viser] get_render failed at frame {i}: {e}")
            continue
        arr = np.asarray(img)
        if arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[..., :3]
        writer.write(arr[..., [2, 1, 0]].astype(np.uint8))
        if (i + 1) % 30 == 0 or i == len(all_states) - 1:
            _logger.info(f"[viser]   frame {i+1}/{len(all_states)}")
    writer.release()
    _status_md.remove()
    server.gui.add_markdown(
        "## ✅ Recording complete!\n"
        f"`{os.path.basename(args.video_path)}` saved.\n\n"
        "You can now interact with the replay using the controls below."
    )
    _logger.info(f"[viser] MP4 saved: {args.video_path} — entering interactive mode")


def _viser_replay(server, handles, name_to_idx, all_states, default_fps):
    """Interactive viser replay with Play/Pause, Frame slider, FPS slider."""
    n_frames = len(all_states)

    def set_frame(idx):
        states = all_states[int(idx)]
        for name, handle in handles.items():
            actor_idx = name_to_idx.get(name)
            if actor_idx is None:
                continue
            pos  = states[actor_idx, 0:3]
            quat = states[actor_idx, 3:7]   # x,y,z,w
            handle.position = tuple(pos.tolist())
            handle.wxyz = (float(quat[3]), float(quat[0]), float(quat[1]), float(quat[2]))

    frame_slider = server.gui.add_slider(
        "Frame", min=0, max=n_frames - 1, step=1, initial_value=0)
    fps_slider = server.gui.add_slider(
        "FPS", min=1, max=120, step=1, initial_value=default_fps)
    play_cb = server.gui.add_checkbox("▶ Play", initial_value=False)

    set_frame(0)

    @frame_slider.on_update
    def _on_frame(_):
        set_frame(frame_slider.value)

    _logger.info(f"[viser] Replay ready: {n_frames} frames. Use GUI controls to play.")
    try:
        while True:
            if play_cb.value:
                cur = int(frame_slider.value)
                nxt = cur + 1
                if nxt >= n_frames:
                    frame_slider.value = 0.0
                    set_frame(0)
                    play_cb.value = False
                    continue
                frame_slider.value = float(nxt)
                set_frame(nxt)
                time.sleep(1.0 / max(1, fps_slider.value))
            else:
                time.sleep(0.02)
    except KeyboardInterrupt:
        pass


def _gpu_memory_used_mib():
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return int(completed.stdout.strip().splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def _nvidia_environment():
    result = {"gpu": None, "driver": None, "memory_total_mib": None}
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        fields = [value.strip() for value in completed.stdout.splitlines()[0].split(",")]
        result.update(
            gpu=fields[0],
            driver=fields[1],
            memory_total_mib=int(fields[2]),
        )
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        pass
    return result


def replay(args):
    total_started = time.perf_counter()
    scene_tree_path = args.scene_tree

    _logger.info(f"[replay] scene_dir={args.scene_dir}  output_dir={args.output_dir}  settle_steps={args.settle_steps}"
                 f"  headless={args.headless}  save_video={args.save_video}")

    variant_dir = args.scene_dir
    scene = load_replay_scene(scene_tree_path, variant_dir)
    all_names = list(scene.names)
    object_specs = {spec.name: spec for spec in scene.objects}
    obj_paths = {name: str(spec.obj_path) for name, spec in object_specs.items()}
    _logger.info(f"[replay] {len(all_names)} objects from {scene.scene_dir / 'obj_files'}")

    fixed_set = set(scene.fixed_names)
    _logger.info(f"[replay] fixed={len(fixed_set & set(all_names))}  "
                 f"movable={len(set(all_names) - fixed_set)}")

    # --viser implies headless (no gym popup window)
    headless = args.headless or args.viser

    # Isaac Gym exposes two independent CUDA switches.  Its bundled PhysX can
    # run on a Blackwell GPU even though the Python 3.8-compatible PyTorch
    # wheel cannot execute sm_120 kernels.  In that case keep GPU PhysX and use
    # host tensors for the state pipeline.
    use_gpu_physics = not args.cpu and torch.cuda.is_available()
    pipeline_supported, pipeline_reason = torch_cuda_pipeline_supported()
    use_gpu_pipeline = use_gpu_physics and pipeline_supported

    if use_gpu_physics and not use_gpu_pipeline:
        _logger.warning(
            "[replay] %s; using GPU PhysX with the CPU tensor pipeline",
            pipeline_reason,
        )
    elif not use_gpu_physics:
        _logger.info("[replay] using CPU PhysX and the CPU tensor pipeline")

    ig = gymapi.acquire_gym()
    sim, viewer = create_sim_and_viewer(
        ig,
        headless=headless,
        num_position_iterations=args.num_position_iterations,
        max_depenetration_velocity=args.max_depenetration_velocity,
        use_gpu_pipeline=use_gpu_pipeline,
        use_gpu_physics=use_gpu_physics,
    )
    gpu_memory_samples = [_gpu_memory_used_mib()]

    # ---- load assets ------------------------------------------------
    asset_started = time.perf_counter()
    assets = {}
    for name in all_names:
        urdf_path = object_specs[name].urdf_path
        asset_dir = urdf_path.parent
        urdf_fname = urdf_path.name
        is_fixed = name in fixed_set
        opt = gymapi.AssetOptions()
        opt.fix_base_link         = is_fixed
        opt.disable_gravity       = is_fixed
        opt.collapse_fixed_joints = True
        opt.override_com          = True
        opt.override_inertia      = True
        opt.linear_damping        = args.linear_damping
        opt.angular_damping       = args.angular_damping
        opt.vhacd_enabled         = not args.no_vhacd
        if opt.vhacd_enabled:
            opt.vhacd_params.max_convex_hulls = args.vhacd_max_hulls
            opt.vhacd_params.resolution = 100_000
        asset = ig.load_asset(sim, str(asset_dir), urdf_fname, opt)
        assets[name] = (asset, is_fixed)
        _logger.info(f"  loaded {name}  fixed={is_fixed}  ({asset_dir})")
    if tuple(assets) != tuple(all_names):
        raise RuntimeError("loaded asset set/order differs from the validated scene")
    if args.expected_object_count is not None and len(assets) != args.expected_object_count:
        raise RuntimeError(
            "validated scene contains %d objects; expected %d"
            % (len(assets), args.expected_object_count)
        )

    # ---- create single env + actors ---------------------------------
    env_handle = ig.create_env(
        sim,
        gymapi.Vec3(-10.0, 0.0, -10.0),
        gymapi.Vec3( 10.0, 10.0,  10.0),
        1,
    )
    actor_handles = {}
    for name, (asset, _) in assets.items():
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, 100.0, 0.0)   # stage above sim floor
        pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        actor_handles[name] = ig.create_actor(env_handle, asset, pose, name, 0, 0)

    ig.prepare_sim(sim)
    asset_load_seconds = time.perf_counter() - asset_started
    gpu_memory_samples.append(_gpu_memory_used_mib())

    _rt = ig.acquire_actor_root_state_tensor(sim)
    root_states = gymtorch.wrap_tensor(_rt)   # (n_sim_actors, 13) on GPU or CPU
    n_actors = root_states.shape[0]

    name_to_idx = {
        name: ig.get_actor_index(env_handle, ah, gymapi.DOMAIN_SIM)
        for name, ah in actor_handles.items()
    }
    if set(name_to_idx) != set(all_names):
        raise RuntimeError("created actor set differs from the validated scene")

    _device = root_states.device
    ordered_indices = torch.tensor(
        [name_to_idx[name] for name in all_names], dtype=torch.long, device=_device
    )

    def push_poses():
        # OBJs are world-frame: actor origin at identity, mesh vertices define world positions
        root_states[:, 0:3] = 0.0
        root_states[:, 3:6] = 0.0
        root_states[:, 6]   = 1.0
        root_states[:, 7:]  = 0.0
        all_idx = torch.arange(n_actors, dtype=torch.int32, device=_device)
        ig.set_actor_root_state_tensor_indexed(
            sim,
            gymtorch.unwrap_tensor(root_states),
            gymtorch.unwrap_tensor(all_idx),
            n_actors,
        )

    push_poses()
    ig.refresh_actor_root_state_tensor(sim)
    initial_states = root_states[ordered_indices].detach().cpu().numpy().copy()
    state_frames_xyzw = [initial_states]

    # ---- camera framing: AABB from stage3 world-frame OBJs (no transform needed) ----
    from rest3d.utils.mesh import load_trimesh_any

    _world_mins, _world_maxs = [], []
    for name in all_names:
        obj_path = obj_paths[name]
        try:
            mesh = load_trimesh_any(obj_path)
        except Exception:
            continue
        lo, hi = mesh.bounds  # stage3 OBJs are world-frame
        _world_mins.append(lo)
        _world_maxs.append(hi)
    if _world_mins:
        bbox_min = np.array(_world_mins).min(axis=0).astype(np.float32)
        bbox_max = np.array(_world_maxs).max(axis=0).astype(np.float32)
        center   = 0.5 * (bbox_min + bbox_max)
        radius   = float(np.linalg.norm(bbox_max - bbox_min)) * 0.5 + 0.5
    else:
        center = np.zeros(3, dtype=np.float32)
        radius = 3.0
    k       = 1.5 * radius
    cam_pos = np.array([center[0], center[1] + 0.4 * k, center[2] - k], dtype=np.float32)
    _logger.info(f"[replay] Scene center={center.tolist()}  radius={radius:.2f}  k={k:.2f}")

    if viewer is not None:
        ig.viewer_camera_look_at(
            viewer, None,
            gymapi.Vec3(float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2])),
            gymapi.Vec3(float(center[0]),  float(center[1]),  float(center[2])),
        )

    # ---- camera sensor for MP4 capture (Isaac Gym, only when not using viser) ----
    cam_handle = None
    frames = []
    if args.save_video and not args.viser:
        cam_props = gymapi.CameraProperties()
        cam_props.width  = args.cam_width
        cam_props.height = args.cam_height
        cam_handle = ig.create_camera_sensor(env_handle, cam_props)
        ig.set_camera_location(
            cam_handle, env_handle,
            gymapi.Vec3(float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2])),
            gymapi.Vec3(float(center[0]),  float(center[1]),  float(center[2])),
        )
        _logger.info(f"[replay] Camera at {cam_pos.tolist()}  target={center.tolist()}")

    # ---- settle loop: physics + state recording -------
    all_states = []   # list of (n_actors, 7) float32 numpy arrays for viser replay
    _logger.info(f"[replay] Settling {args.settle_steps} steps ...")
    simulation_started = time.perf_counter()
    completed_steps = 0
    for step in range(args.settle_steps):
        ig.simulate(sim)
        ig.fetch_results(sim, True)
        ig.refresh_actor_root_state_tensor(sim)
        if viewer is not None or cam_handle is not None:
            ig.step_graphics(sim)
        state_frames_xyzw.append(
            root_states[ordered_indices].detach().cpu().numpy().copy()
        )
        completed_steps = step + 1
        if cam_handle is not None:
            ig.render_all_camera_sensors(sim)
            rgba = ig.get_camera_image(sim, env_handle, cam_handle, gymapi.IMAGE_COLOR)
            rgb = np.frombuffer(rgba, dtype=np.uint8).reshape(
                args.cam_height, args.cam_width, 4)[:, :, :3]
            frames.append(rgb.copy())
        if args.viser:
            all_states.append(root_states[:, :7].cpu().numpy().copy())
        if (step + 1) % 30 == 0:
            gpu_memory_samples.append(_gpu_memory_used_mib())
        if viewer is not None:
            ig.draw_viewer(viewer, sim, True)
            if ig.query_viewer_has_closed(viewer):
                _logger.info("[replay] Viewer closed early.")
                break
    simulation_seconds = time.perf_counter() - simulation_started
    gpu_memory_samples.append(_gpu_memory_used_mib())
    ig.refresh_actor_root_state_tensor(sim)
    final_states = root_states[ordered_indices].detach().cpu().numpy().copy()

    # ---- save Isaac Gym MP4 (non-viser mode) -------------------------
    if args.save_video and frames:
        import cv2
        os.makedirs(os.path.dirname(os.path.abspath(args.video_path)), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            args.video_path, fourcc, args.fps,
            (args.cam_width, args.cam_height))
        if not writer.isOpened():
            _logger.warning(f"[replay] VideoWriter failed to open: {args.video_path}")
        else:
            _logger.info(f"[replay] Writing {len(frames)} frames -> {args.video_path}")
            for frame in frames:
                writer.write(frame[:, :, [2, 1, 0]])  # RGB → BGR for cv2
            writer.release()
            _logger.info(f"[replay] Video saved: {args.video_path}")

    # ---- keep interactive viewer open --------------------------------
    if viewer is not None:
        _logger.info("[replay] Viewer open — close window to exit.")
        while not ig.query_viewer_has_closed(viewer):
            ig.simulate(sim)
            ig.fetch_results(sim, True)
            ig.step_graphics(sim)
            ig.draw_viewer(viewer, sim, True)
        ig.destroy_viewer(viewer)

    states_xyzw = np.stack(state_frames_xyzw, axis=0)
    states_rest = gym_states_xyzw_to_rest(states_xyzw)
    stability = evaluate_replay_stability(
        states_rest,
        evaluation_step=args.stability_evaluation_steps,
        position_threshold_m=args.position_stability_threshold,
        rotation_threshold_rad=args.rotation_stability_threshold,
        early_window_steps=args.early_window_steps,
        terminal_window_steps=args.terminal_window_steps,
    )
    valid_gpu_memory = [value for value in gpu_memory_samples if value is not None]
    nvidia_environment = _nvidia_environment()
    compute_capability = (
        list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None
    )
    collision_geometry = {
        name: {
            "body_count": int(ig.get_asset_rigid_body_count(assets[name][0])),
            "shape_count": int(ig.get_asset_rigid_shape_count(assets[name][0])),
        }
        for name in all_names
    }
    checks = {
        "all_scene_objects_loaded": tuple(assets) == tuple(all_names),
        "expected_object_count_matches": args.expected_object_count is None
        or len(all_names) == args.expected_object_count,
        "all_objects_have_rigid_bodies": all(
            values["body_count"] > 0 for values in collision_geometry.values()
        ),
        "all_objects_have_collision_shapes": all(
            values["shape_count"] > 0 for values in collision_geometry.values()
        ),
        "physics_uses_gpu_sim": use_gpu_physics,
        "physics_uses_expected_cpu_tensor_pipeline": not use_gpu_pipeline
        and root_states.device.type == "cpu",
        "state_shape_is_complete": states_rest.shape
        == (args.settle_steps + 1, len(all_names), 13),
        "all_objects_evaluated_for_stability": stability.stable.shape
        == (len(all_names),),
        "states_are_finite": bool(np.isfinite(states_rest).all()),
    }
    if args.require_stable:
        checks["scene_stable_at_evaluation_step"] = stability.scene_stable

    np.save(os.path.join(args.output_dir, "replay_states_initial.npy"), states_rest[0])
    np.save(os.path.join(args.output_dir, "replay_states_final.npy"), states_rest[-1])
    np.save(os.path.join(args.output_dir, "replay_states_rest.npy"), states_rest)
    np.save(os.path.join(args.output_dir, "replay_states_gym_xyzw.npy"), states_xyzw)
    np.savez_compressed(
        os.path.join(args.output_dir, "stability_metrics.npz"),
        object_names=np.asarray(all_names),
        fixed=np.asarray([name in fixed_set for name in all_names]),
        evaluation_step=np.asarray(stability.evaluation_step),
        position_threshold_m=np.asarray(stability.position_threshold_m),
        rotation_threshold_rad=np.asarray(stability.rotation_threshold_rad),
        stable=stability.stable,
        displacement_at_evaluation_m=stability.displacement_at_evaluation_m,
        rotation_at_evaluation_rad=stability.rotation_at_evaluation_rad,
        final_displacement_m=stability.final_displacement_m,
        final_rotation_rad=stability.final_rotation_rad,
        maximum_excursion_m=stability.maximum_excursion_m,
        maximum_rotation_excursion_rad=stability.maximum_rotation_excursion_rad,
        early_max_linear_speed_m_s=stability.early_max_linear_speed_m_s,
        early_max_angular_speed_rad_s=stability.early_max_angular_speed_rad_s,
        terminal_max_linear_speed_m_s=stability.terminal_max_linear_speed_m_s,
        terminal_max_angular_speed_rad_s=stability.terminal_max_angular_speed_rad_s,
        terminal_mean_linear_speed_m_s=stability.terminal_mean_linear_speed_m_s,
        terminal_mean_angular_speed_rad_s=stability.terminal_mean_angular_speed_rad_s,
    )

    object_results = {}
    for index, name in enumerate(all_names):
        unstable_reasons = []
        if stability.displacement_at_evaluation_m[index] > stability.position_threshold_m:
            unstable_reasons.append("translation")
        if stability.rotation_at_evaluation_rad[index] > stability.rotation_threshold_rad:
            unstable_reasons.append("rotation")
        object_results[name] = {
            "fixed": name in fixed_set,
            "collision": collision_geometry[name],
            "initial_state_rest": states_rest[0, index].tolist(),
            "evaluation_state_rest": states_rest[stability.evaluation_step, index].tolist(),
            "final_state_rest": states_rest[-1, index].tolist(),
            "stable": bool(stability.stable[index]),
            "unstable_reasons": unstable_reasons,
            "displacement_at_evaluation_m": float(
                stability.displacement_at_evaluation_m[index]
            ),
            "rotation_at_evaluation_rad": float(
                stability.rotation_at_evaluation_rad[index]
            ),
            "final_displacement_m": float(stability.final_displacement_m[index]),
            "final_rotation_rad": float(stability.final_rotation_rad[index]),
            "maximum_excursion_m": float(stability.maximum_excursion_m[index]),
            "maximum_rotation_excursion_rad": float(
                stability.maximum_rotation_excursion_rad[index]
            ),
            "early_max_linear_speed_m_s": float(
                stability.early_max_linear_speed_m_s[index]
            ),
            "early_max_angular_speed_rad_s": float(
                stability.early_max_angular_speed_rad_s[index]
            ),
            "terminal_max_linear_speed_m_s": float(
                stability.terminal_max_linear_speed_m_s[index]
            ),
            "terminal_max_angular_speed_rad_s": float(
                stability.terminal_max_angular_speed_rad_s[index]
            ),
            "terminal_mean_linear_speed_m_s": float(
                stability.terminal_mean_linear_speed_m_s[index]
            ),
            "terminal_mean_angular_speed_rad_s": float(
                stability.terminal_mean_angular_speed_rad_s[index]
            ),
        }

    replay_result = {
        "passed": all(checks.values()),
        "runtime_passed": all(
            value
            for key, value in checks.items()
            if key != "scene_stable_at_evaluation_step"
        ),
        "backend": "isaac-gym",
        "scene_stable": stability.scene_stable,
        "checks": checks,
        "versions": {
            "python": platform.python_version(),
            "isaac_gym": "Preview 4",
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "driver": nvidia_environment["driver"],
        },
        "environment": {
            "platform": platform.platform(),
            "conda_prefix": os.environ.get("CONDA_PREFIX"),
            "gpu": nvidia_environment["gpu"],
            "compute_capability": compute_capability,
            "gpu_memory_total_mib": nvidia_environment["memory_total_mib"],
        },
        "scene": {
            "scene_tree": scene_tree_path,
            "scene_dir": variant_dir,
            "object_names": all_names,
            "fixed_names": sorted(fixed_set & set(all_names)),
            "movable_names": sorted(set(all_names) - fixed_set),
            "object_count": len(all_names),
        },
        "simulation": {
            "device": "cuda:0" if use_gpu_physics else "cpu",
            "physics_uses_gpu_sim": use_gpu_physics,
            "physics_uses_gpu_pipeline": use_gpu_pipeline,
            "physics_broadphase_type": "GPU" if use_gpu_physics else "CPU",
            "state_tensor_device": str(root_states.device),
            "state_tensor_shape": list(root_states.shape),
            "recorded_state_shape": list(states_rest.shape),
            "steps": completed_steps,
            "dt_seconds": 1.0 / 60.0,
            "startup_seconds": asset_started - total_started,
            "asset_load_seconds": asset_load_seconds,
            "simulation_seconds": simulation_seconds,
            "steps_per_second": completed_steps / simulation_seconds,
            "total_seconds": time.perf_counter() - total_started,
            "system_gpu_memory_used_mib_peak": max(valid_gpu_memory)
            if valid_gpu_memory
            else None,
            "collision_approximation": "convex_hull"
            if args.no_vhacd
            else "convex_decomposition",
            "vhacd_max_convex_hulls": None
            if args.no_vhacd
            else args.vhacd_max_hulls,
            "data_collection": "root_states_each_step",
        },
        "stability": {
            "evaluation_step": stability.evaluation_step,
            "evaluation_time_seconds": stability.evaluation_step / 60.0,
            "position_threshold_m": stability.position_threshold_m,
            "rotation_threshold_rad": stability.rotation_threshold_rad,
            "require_stable": args.require_stable,
            "scene_stable": stability.scene_stable,
            "stable_object_count": int(np.count_nonzero(stability.stable)),
            "unstable_object_names": [
                name for index, name in enumerate(all_names) if not stability.stable[index]
            ],
            "metrics_file": os.path.join(args.output_dir, "stability_metrics.npz"),
        },
        "objects": object_results,
    }
    with open(os.path.join(args.output_dir, "replay_results.json"), "x") as file:
        json.dump(replay_result, file, indent=2)
        file.write("\n")

    ig.destroy_sim(sim)
    _logger.info("[replay] Physics done. Results: %s", os.path.join(args.output_dir, "replay_results.json"))
    if not replay_result["passed"]:
        raise RuntimeError("Isaac Gym replay checks failed: %s" % checks)

    # Save states for future interactive replay without re-running physics
    if all_states:
        _states_path = os.path.join(args.output_dir, "replay_states.npy")
        np.save(_states_path, np.stack(all_states, axis=0))
        _logger.info(f"[replay] States saved: {_states_path}")

    # ---- viser replay (after physics) --------------------------------
    if args.viser and all_states:
        viser_server = init_viser_server(center, radius, args)
        if viser_server is not None:
            handles = load_viser_objects(
                viser_server, obj_paths, name_to_idx,
                obj_delay=args.viser_obj_delay)
            if handles:
                if args.save_video:
                    _record_viser_mp4(viser_server, handles, name_to_idx, all_states,
                                      cam_pos, center, args)
                _viser_replay(viser_server, handles, name_to_idx, all_states, args.fps)

    _logger.info("[replay] Done.")


def main():
    args = config_args()
    replay(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    # Isaac Gym P4 segfaults during Python interpreter teardown (known upstream bug).
    # sys.exit(0) terminates before the C++ module destructors fire.
    sys.exit(0)
