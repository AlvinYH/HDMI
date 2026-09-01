#!/usr/bin/env python3
"""Render direct-input ArtHOI4D HDMI rollouts as frames and/or an MP4."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import torch
import numpy as np
from omegaconf import OmegaConf

from isaaclab.app import AppLauncher
from torchrl.envs.utils import ExplorationType, set_exploration_type


def parse_resolution(value: str) -> list[int]:
    width, height = (int(v) for v in value.lower().split("x", 1))
    return [width, height]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Direct ArtHOI4D HDMI input bundle (hdmi_input.json).",
    )
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument(
        "--output-video",
        type=Path,
        default=None,
        help="Optional MP4 assembled from the rendered PNG frame sequence.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Video frame rate; defaults to 1 / task.sim.step_dt from the run config.",
    )
    parser.add_argument("--num-frames", type=int, default=-1)
    parser.add_argument("--resolution", default="640x360")
    parser.add_argument("--render-mode", default="rgb_array")
    parser.add_argument("--rendering-mode", default=None)
    parser.add_argument("--pt-spp", type=int, default=32)
    parser.add_argument("--pt-total-spp", type=int, default=None)
    parser.add_argument("--antialiasing-mode", default=None)
    parser.add_argument("--carb-settings-json", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--gravity", nargs=3, type=float, default=None)
    parser.add_argument("--root-body-name", default=None)
    parser.add_argument("--print-reset-state", action="store_true")
    parser.add_argument("--ignore-done", action="store_true")
    parser.add_argument(
        "--reference-playback",
        action="store_true",
        help="Render the prepared HDMI reference motion directly instead of rolling out the policy.",
    )
    parser.add_argument("--init-joint-pos-noise", type=float, default=None)
    parser.add_argument("--init-joint-vel-noise", type=float, default=None)
    parser.add_argument("--action-scaling", type=float, default=None)
    parser.add_argument("--min-delay", type=int, default=None)
    parser.add_argument("--max-delay", type=int, default=None)
    parser.add_argument("--kp-scale", type=float, default=None)
    parser.add_argument("--kd-scale", type=float, default=None)
    parser.add_argument("--effort-limit-scale", type=float, default=None)
    parser.add_argument("--eye", nargs=3, type=float, default=[3.5, -4.5, 2.4])
    parser.add_argument("--lookat", nargs=3, type=float, default=[0.0, 1.9, 0.8])
    return parser.parse_args()


def load_motion_length(motion_dir: Path) -> int:
    motion = np.load(motion_dir / "motion.npz", allow_pickle=True)
    return int(motion["joint_pos"].shape[0])


def zero_pose_range() -> dict[str, list[float]]:
    return {key: [0.0, 0.0] for key in ("x", "y", "z", "roll", "pitch", "yaw")}


def clear_existing_frames(frames_dir: Path) -> int:
    removed = 0
    for path in frames_dir.glob("frame_*.png"):
        if path.is_file():
            path.unlink()
            removed += 1
    return removed


def resolve_video_fps(cfg, requested_fps: float | None) -> float:
    if requested_fps is not None:
        if requested_fps <= 0:
            raise ValueError("--fps must be positive")
        return float(requested_fps)
    try:
        step_dt = float(cfg.task.sim.step_dt)
    except (AttributeError, TypeError, ValueError):
        step_dt = 1.0 / 30.0
    if step_dt <= 0:
        raise ValueError(f"task.sim.step_dt must be positive, got {step_dt}")
    return 1.0 / step_dt


def encode_video_from_frames(
    frames_dir: Path,
    num_frames: int,
    output_video: Path,
    fps: float,
    imageio,
) -> None:
    output_video.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(
        output_video,
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
    ) as writer:
        for frame_idx in range(num_frames):
            frame_path = frames_dir / f"frame_{frame_idx:05d}.png"
            if not frame_path.is_file():
                raise FileNotFoundError(f"Missing rendered frame: {frame_path}")
            writer.append_data(imageio.imread(frame_path))
    print(f"saved video ({fps:g} fps): {output_video}", flush=True)


def summarize_scale(value) -> str:
    if value is None:
        return "None"
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.numel() <= 12:
            return str(value.tolist())
        flat = value.reshape(-1, value.shape[-1]) if value.ndim > 1 else value.reshape(-1)
        return (
            f"tensor(shape={tuple(value.shape)}, "
            f"min={float(value.min()):.6g}, max={float(value.max()):.6g}, "
            f"first={flat[0].tolist() if flat.ndim > 1 else float(flat[0]):})"
        )
    return repr(value)


def prepare_video_frame(frame):
    frame = np.asarray(frame)
    if frame.ndim == 4 and frame.shape[0] == 1:
        frame = frame[0]
    if frame.ndim == 3 and frame.shape[-1] > 3:
        frame = frame[..., :3]
    if np.issubdtype(frame.dtype, np.floating):
        finite = np.isfinite(frame)
        max_value = frame[finite].max() if finite.any() else 0.0
        if max_value <= 1.0:
            frame = frame * 255.0
        frame = np.nan_to_num(frame, nan=0.0, posinf=255.0, neginf=0.0)
        frame = np.clip(frame, 0.0, 255.0).astype(np.uint8)
    elif frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frame)


def normalize_rendering_mode(value: str | None) -> str | None:
    if value is None:
        return None
    return value.strip().lower().replace("_", "").replace("-", "")


def apply_runtime_render_settings(
    rendering_mode: str | None,
    antialiasing_mode: str | None,
    carb_settings_json: str | None,
    pt_spp: int,
    pt_total_spp: int | None,
    label: str,
) -> None:
    import carb
    from isaacsim.core.utils.carb import set_carb_setting

    carb_settings = carb.settings.get_settings()
    mode = normalize_rendering_mode(rendering_mode)

    if mode in {"pathtracing", "pathtrace", "pt"}:
        total_spp = pt_spp if pt_total_spp is None else pt_total_spp
        set_carb_setting(carb_settings, "/rtx/rendermode", "PathTracing")
        set_carb_setting(carb_settings, "/rtx/pathtracing/spp", pt_spp)
        set_carb_setting(carb_settings, "/rtx/pathtracing/totalSpp", total_spp)
        set_carb_setting(carb_settings, "/rtx/pathtracing/clampSpp", total_spp)
    elif mode in {"raytracedlighting", "raytracing", "realtime", "rtxrealtime", "rt"}:
        set_carb_setting(carb_settings, "/rtx/rendermode", "RaytracedLighting")

    if antialiasing_mode:
        try:
            import omni.replicator.core as rep

            rep.settings.set_render_rtx_realtime(antialiasing=antialiasing_mode)
        except Exception as exc:
            print(f"[WARN] failed to set antialiasing mode {antialiasing_mode!r}: {exc}", flush=True)

    if carb_settings_json:
        carb_overrides = json.loads(Path(carb_settings_json).read_text() if Path(carb_settings_json).is_file() else carb_settings_json)
        for key, value in carb_overrides.items():
            set_carb_setting(carb_settings, key, value)

    print(
        "[render-settings:{}] /rtx/rendermode={!r} /rtx/pathtracing/spp={!r} "
        "/rtx/pathtracing/totalSpp={!r}".format(
            label,
            carb_settings.get("/rtx/rendermode"),
            carb_settings.get("/rtx/pathtracing/spp"),
            carb_settings.get("/rtx/pathtracing/totalSpp"),
        ),
        flush=True,
    )


def write_reference_frame(base_env, command, frame_idx: int) -> None:
    env_ids = torch.zeros(1, dtype=torch.long, device=base_env.device)
    env_origin = base_env.scene.env_origins[env_ids]
    start = int(command.dataset.starts[0].item())
    motion = command.dataset.data[start + frame_idx]

    robot = command.asset
    root_pos = motion.body_pos_w[command.root_body_idx_motion].unsqueeze(0) + env_origin
    root_quat = motion.body_quat_w[command.root_body_idx_motion].unsqueeze(0)
    root_lin_vel = motion.body_lin_vel_w[command.root_body_idx_motion].unsqueeze(0)
    root_ang_vel = motion.body_ang_vel_w[command.root_body_idx_motion].unsqueeze(0)
    robot.write_root_link_pose_to_sim(torch.cat([root_pos, root_quat], dim=-1), env_ids=env_ids)
    robot.write_root_com_velocity_to_sim(torch.cat([root_lin_vel, root_ang_vel], dim=-1), env_ids=env_ids)
    robot.write_joint_state_to_sim(
        motion.joint_pos[command.asset_joint_idx_motion].unsqueeze(0),
        motion.joint_vel[command.asset_joint_idx_motion].unsqueeze(0),
        env_ids=env_ids,
    )

    if hasattr(command, "object"):
        obj = command.object
        object_pos = motion.body_pos_w[command.object_body_id_motion].unsqueeze(0) + env_origin
        object_quat = motion.body_quat_w[command.object_body_id_motion].unsqueeze(0)
        object_lin_vel = motion.body_lin_vel_w[command.object_body_id_motion].unsqueeze(0)
        object_ang_vel = motion.body_ang_vel_w[command.object_body_id_motion].unsqueeze(0)
        obj.write_root_link_pose_to_sim(torch.cat([object_pos, object_quat], dim=-1), env_ids=env_ids)
        obj.write_root_com_velocity_to_sim(torch.cat([object_lin_vel, object_ang_vel], dim=-1), env_ids=env_ids)
        if command.object_joint_indices_asset is not None:
            obj.write_joint_state_to_sim(
                motion.joint_pos[command.object_joint_indices_motion].unsqueeze(0),
                motion.joint_vel[command.object_joint_indices_motion].unsqueeze(0),
                env_ids=env_ids,
                joint_ids=command.object_joint_indices_asset,
            )

    base_env.sim.forward()
    base_env.scene.update(0.0)


def render_reference_frames(env, frames_dir: Path, num_frames: int, render_mode: str, imageio) -> None:
    base_env = env.base_env
    command = base_env.command_manager
    with torch.inference_mode():
        for frame_idx in range(num_frames):
            write_reference_frame(base_env, command, frame_idx)
            frame = prepare_video_frame(env.render(mode=render_mode))
            imageio.imwrite(frames_dir / f"frame_{frame_idx:05d}.png", frame)
            if frame_idx == 0 or frame_idx % 25 == 0:
                print(f"saved frame {frame_idx}/{num_frames}", flush=True)


def main() -> None:
    args = parse_args()
    hdmi_root = Path(__file__).resolve().parents[1]
    run_dir = args.run_dir.resolve()
    checkpoint = args.checkpoint.resolve()
    input_path = args.input.resolve()
    frames_dir = args.frames_dir.resolve()
    frames_dir.mkdir(parents=True, exist_ok=True)

    input_data = json.loads(input_path.read_text(encoding="utf-8"))
    motion_dir = (input_path.parent / input_data["motion_dir"]).resolve()
    motion_len = load_motion_length(motion_dir)
    num_frames = motion_len if args.num_frames is None or args.num_frames < 0 else min(args.num_frames, motion_len)
    removed_frames = clear_existing_frames(frames_dir)
    print(
        f"[render] motion_len={motion_len} requested_num_frames={args.num_frames} "
        f"render_num_frames={num_frames} frames_dir={frames_dir} "
        f"removed_stale_frames={removed_frames}",
        flush=True,
    )

    # Use the fork's direct runtime integration.  This renderer must never
    # depend on the historical apply-adapter/manifest protocol.
    os.environ["ARTHOI4D_HDMI_INPUT_PATH"] = str(input_path)
    os.environ["ARTHOI4D_HDMI_MOTION_DIR"] = str(motion_dir)
    os.environ["ARTHOI4D_HDMI_OBJECT_BODY"] = input_data["object_contact_body_name"]
    os.environ["ARTHOI4D_HDMI_CONTACT_TARGET_BODY_NAMES"] = json.dumps(
        input_data["contact_target_body_names"]
    )
    os.environ["ARTHOI4D_HDMI_CONTACT_TARGET_POS_OFFSETS"] = json.dumps(
        input_data["contact_target_pos_offsets"]
    )
    os.environ["ARTHOI4D_HDMI_OBJECT_JOINT_NAMES"] = json.dumps(
        input_data["object_joint_names"]
    )
    os.environ["ARTHOI4D_HDMI_CONTACT_REGION_LINK_NAMES"] = json.dumps(
        input_data["contact_region_link_names"]
    )
    os.environ["ARTHOI4D_HDMI_SUPPORT_NAMES"] = json.dumps(
        [f"arthoi4d_support_{index}" for index in range(len(input_data["support_boxes"]))]
    )
    if args.rendering_mode:
        os.environ["ARTHOI4D_HDMI_RENDERING_MODE"] = args.rendering_mode
    if args.antialiasing_mode:
        os.environ["ARTHOI4D_HDMI_ANTIALIASING_MODE"] = args.antialiasing_mode
    if args.carb_settings_json:
        os.environ["ARTHOI4D_HDMI_CARB_SETTINGS_JSON"] = args.carb_settings_json

    if str(hdmi_root) not in sys.path:
        sys.path.insert(0, str(hdmi_root))

    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    OmegaConf.set_struct(cfg, False)
    cfg.checkpoint_path = str(checkpoint)
    cfg.vecnorm = "eval"
    cfg.eval_render = True
    cfg.render_mode = args.render_mode
    cfg.app.headless = True
    cfg.app.enable_cameras = True
    if normalize_rendering_mode(args.rendering_mode) in {"pathtracing", "pathtrace", "pt"}:
        cfg.app.rendering_mode = "quality"
    elif normalize_rendering_mode(args.rendering_mode) in {"performance", "balanced", "quality"}:
        cfg.app.rendering_mode = args.rendering_mode
    cfg.task.num_envs = 1
    cfg.task.max_episode_length = num_frames
    cfg.task.viewer.resolution = parse_resolution(args.resolution)
    cfg.task.viewer.eye = list(args.eye)
    cfg.task.viewer.lookat = list(args.lookat)
    cfg.task.command.data_path = str(motion_dir)
    cfg.task.command.object_body_name = input_data["object_contact_body_name"]
    cfg.task.command.contact_target_body_names = input_data["contact_target_body_names"]
    cfg.task.command.contact_target_pos_offset = input_data["contact_target_pos_offsets"]
    cfg.task.command.object_joint_names = input_data["object_joint_names"]
    if args.root_body_name is not None:
        cfg.task.command.root_body_name = args.root_body_name
    if args.init_joint_pos_noise is not None:
        cfg.task.command.init_joint_pos_noise = args.init_joint_pos_noise
    if args.init_joint_vel_noise is not None:
        cfg.task.command.init_joint_vel_noise = args.init_joint_vel_noise
    if args.action_scaling is not None:
        cfg.task.action.action_scaling_replace = args.action_scaling
    if args.min_delay is not None:
        cfg.task.action.min_delay = args.min_delay
    if args.max_delay is not None:
        cfg.task.action.max_delay = args.max_delay
    if args.seed is not None:
        cfg.seed = args.seed
    if args.gravity is not None:
        cfg.task.sim.gravity = list(args.gravity)
    cfg.task.max_episode_length = num_frames
    video_fps = resolve_video_fps(cfg, args.fps)

    # Rendering must be a deterministic checkpoint rollout. Never inject the
    # training reset randomization into the human or object at evaluation time.
    cfg.task.command.pose_range = zero_pose_range()
    cfg.task.command.velocity_range = zero_pose_range()
    cfg.task.command.init_joint_pos_noise = 0.0
    cfg.task.command.init_joint_vel_noise = 0.0
    cfg.task.command.object_pose_range = zero_pose_range()
    cfg.task.command.object_init_joint_pos_noise = 0.0
    cfg.task.command.object_init_joint_vel_noise = 0.0
    if args.kp_scale is not None or args.kd_scale is not None or args.effort_limit_scale is not None:
        robot_override = dict(getattr(cfg.task.robot, "override_params", {}))
        actuators = dict(robot_override.get("actuators", {}))
        smplx = dict(actuators.get("smplx", {}))
        if args.kp_scale is not None:
            smplx["stiffness"] = float(args.kp_scale) * 80.0
        if args.kd_scale is not None:
            smplx["damping"] = float(args.kd_scale) * 4.0
        if args.effort_limit_scale is not None:
            smplx["effort_limit_sim"] = float(args.effort_limit_scale) * 300.0
        actuators["smplx"] = smplx
        robot_override["actuators"] = actuators
        cfg.task.robot.override_params = robot_override

    app_launcher = AppLauncher(OmegaConf.to_container(cfg.app))
    simulation_app = app_launcher.app
    apply_runtime_render_settings(
        args.rendering_mode,
        args.antialiasing_mode,
        args.carb_settings_json,
        args.pt_spp,
        args.pt_total_spp,
        "after-app-launch",
    )

    from scripts.helpers import make_env_policy
    import imageio.v2 as imageio

    env, policy, _ = make_env_policy(cfg)
    object_asset_name = cfg.task.command.object_asset_name
    try:
        object_asset = env.scene[object_asset_name]
        print(
            f"[render] object_asset={object_asset_name} "
            f"spawn.scale={summarize_scale(getattr(object_asset.cfg.spawn, 'scale', None))} "
            f"spawn.scale_range={summarize_scale(getattr(object_asset.cfg.spawn, 'scale_range', None))}",
            flush=True,
        )
    except Exception as exc:
        print(f"[WARN] could not inspect object scale for {object_asset_name}: {exc}", flush=True)
    apply_runtime_render_settings(
        args.rendering_mode,
        args.antialiasing_mode,
        args.carb_settings_json,
        args.pt_spp,
        args.pt_total_spp,
        "after-env-create",
    )
    rollout_policy = policy.get_rollout_policy("eval")

    env.base_env.eval()
    env.eval()
    env.set_seed(cfg.seed)
    tensordict = env.reset()

    if args.print_reset_state:
        robot = env.scene["robot"]
        body_names = robot.body_names
        for name in ["Pelvis", "Chest", "L_Ankle", "R_Ankle", "L_Toe", "R_Toe"]:
            if name in body_names:
                idx = body_names.index(name)
                pos = robot.data.body_link_pos_w[0, idx].tolist()
                print(f"[reset-state] {name} pos_w={pos}", flush=True)
        print(f"[reset-state] root_link_pos_w={robot.data.root_link_pos_w[0].tolist()}", flush=True)

    if args.reference_playback:
        render_reference_frames(env, frames_dir, num_frames, args.render_mode, imageio)
        (frames_dir / "frames_manifest.json").write_text(
            json.dumps(
                {
                    "motion_len": motion_len,
                    "requested_num_frames": args.num_frames,
                    "num_frames": num_frames,
                    "frame_pattern": "frame_%05d.png",
                    "input": str(input_path),
                    "mode": "reference_playback",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        if args.output_video is not None:
            encode_video_from_frames(
                frames_dir,
                num_frames,
                args.output_video.resolve(),
                video_fps,
                imageio,
            )
        print(f"saved {num_frames} frames to {frames_dir}", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)

    with torch.inference_mode(), set_exploration_type(ExplorationType.MODE):
        torch.compiler.cudagraph_mark_step_begin()
        frame = prepare_video_frame(env.render(mode=args.render_mode))
        imageio.imwrite(frames_dir / "frame_00000.png", frame)
        print(f"saved frame 0/{num_frames}", flush=True)

        for frame_idx in range(1, num_frames):
            tensordict = rollout_policy(tensordict)
            step_td = env.step(tensordict)
            frame = prepare_video_frame(env.render(mode=args.render_mode))
            imageio.imwrite(frames_dir / f"frame_{frame_idx:05d}.png", frame)
            next_td = step_td.get("next", step_td)
            done = bool(next_td.get("done").any().item())
            if done and not args.ignore_done:
                num_frames = frame_idx + 1
                print(f"[render] rollout done at frame {frame_idx}; stop without reset", flush=True)
                break
            tensordict = next_td
            if frame_idx % 25 == 0:
                print(f"saved frame {frame_idx}/{num_frames}", flush=True)

    (frames_dir / "frames_manifest.json").write_text(
        json.dumps(
            {
                "motion_len": motion_len,
                "requested_num_frames": args.num_frames,
                "num_frames": num_frames,
                "frame_pattern": "frame_%05d.png",
                "input": str(input_path),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if args.output_video is not None:
        encode_video_from_frames(
            frames_dir,
            num_frames,
            args.output_video.resolve(),
            video_fps,
            imageio,
        )
    print(f"saved {num_frames} frames to {frames_dir}", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
