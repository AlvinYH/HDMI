"""Export one headless ArtHOI4D HDMI rollout from native Isaac Lab state."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import traceback

import hydra
import numpy as np
import torch
from isaaclab.app import AppLauncher
from isaacsim import SimulationApp
from omegaconf import DictConfig, OmegaConf
from torchrl.envs import ExplorationType, set_exploration_type

HDMI_ROOT = Path(__file__).resolve().parents[1]
if str(HDMI_ROOT) not in sys.path:
    sys.path.insert(0, str(HDMI_ROOT))

from scripts.helpers import make_env_policy


def _input_bundle() -> dict:
    path_value = os.environ.get("ARTHOI4D_HDMI_INPUT_PATH")
    if not path_value:
        raise RuntimeError("ARTHOI4D_HDMI_INPUT_PATH is required")
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"HDMI input bundle is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _seed_hdmi_environment(bundle: dict) -> None:
    os.environ.setdefault("ARTHOI4D_HDMI_MOTION_DIR", str(Path(os.environ["ARTHOI4D_HDMI_INPUT_PATH"]).parent / bundle["motion_dir"]))
    os.environ.setdefault("ARTHOI4D_HDMI_OBJECT_BODY", str(bundle["object_body_names"][0]))
    os.environ.setdefault("ARTHOI4D_HDMI_CONTACT_TARGET_BODY_NAMES", json.dumps(bundle["contact_target_body_names"]))
    os.environ.setdefault("ARTHOI4D_HDMI_CONTACT_TARGET_POS_OFFSETS", json.dumps(bundle["contact_target_pos_offsets"]))
    os.environ.setdefault("ARTHOI4D_HDMI_CONTACT_REGION_LINK_NAMES", json.dumps(bundle["contact_region_link_names"]))
    os.environ.setdefault("ARTHOI4D_HDMI_OBJECT_JOINT_NAMES", json.dumps(bundle["object_joint_names"]))
    os.environ.setdefault("ARTHOI4D_HDMI_OBJECT_INITIAL_QPOS", json.dumps(bundle["object_initial_joint_qpos"]))
    os.environ.setdefault(
        "ARTHOI4D_HDMI_SUPPORT_NAMES",
        json.dumps([f"arthoi4d_support_{index}" for index in range(len(bundle["support_boxes"]))]),
    )


def _indices(names: tuple[str, ...], actual_names: list[str], label: str) -> list[int]:
    missing = [name for name in names if name not in actual_names]
    if missing:
        raise RuntimeError(f"HDMI {label} is missing required names: {missing}")
    return [actual_names.index(name) for name in names]


def _xyzw(wxyz: torch.Tensor) -> torch.Tensor:
    """Convert native Isaac Lab quaternions from wxyz to Studio xyzw."""

    return wxyz[..., (1, 2, 3, 0)]


def _state(
    position: torch.Tensor,
    quaternion_wxyz: torch.Tensor,
    linear_velocity: torch.Tensor,
    angular_velocity: torch.Tensor,
    ground_height: float,
) -> torch.Tensor:
    """Return actual Studio-world state vectors in ``[pos, xyzw, vel, omega]`` order."""

    position = position.clone()
    position[..., 2] += ground_height
    return torch.cat(
        (position, _xyzw(quaternion_wxyz), linear_velocity, angular_velocity), dim=-1
    )


def _snapshot(command, bundle: dict, human_body_ids: list[int], human_joint_ids: list[int]):
    """Read only current simulator state and selected-link contact telemetry."""

    robot = command.asset
    object_asset = command.object
    ground_height = float(bundle["ground_height_studio"])

    human_body = _state(
        robot.data.body_link_pos_w[0, human_body_ids],
        robot.data.body_link_quat_w[0, human_body_ids],
        robot.data.body_com_lin_vel_w[0, human_body_ids],
        robot.data.body_com_ang_vel_w[0, human_body_ids],
        ground_height,
    )
    object_root = _state(
        object_asset.data.root_state_w[0, :3],
        object_asset.data.root_state_w[0, 3:7],
        object_asset.data.root_state_w[0, 7:10],
        object_asset.data.root_state_w[0, 10:13],
        ground_height,
    )
    region_body_ids = command.contact_region_body_indices_asset
    region_link_state = _state(
        object_asset.data.body_link_pos_w[0, region_body_ids],
        object_asset.data.body_link_quat_w[0, region_body_ids],
        object_asset.data.body_com_lin_vel_w[0, region_body_ids],
        object_asset.data.body_com_ang_vel_w[0, region_body_ids],
        ground_height,
    )
    if command.object_joint_indices_asset is None:
        object_joint_qpos = torch.empty(0, device=command.device)
    else:
        object_joint_qpos = object_asset.data.joint_pos[
            0, command.object_joint_indices_asset
        ]
    return {
        "human_root_state": human_body[0],
        "human_dof_pos": robot.data.joint_pos[0, human_joint_ids],
        "human_body_state": human_body,
        "object_root_state": object_root,
        "object_joint_qpos": object_joint_qpos,
        "hand_force_n": command.hand_force_n[0],
        "region_force_n": command.region_force_n[0],
        "region_link_state": region_link_state,
    }


def _stack(frames: list[dict[str, torch.Tensor]]) -> dict[str, np.ndarray]:
    if not frames:
        raise RuntimeError("HDMI rollout ended before producing a native frame")
    payload = {
        name: torch.stack([frame[name] for frame in frames]).detach().cpu().numpy()
        for name in frames[0]
    }
    for name, value in payload.items():
        if value.dtype != np.float32:
            payload[name] = value.astype(np.float32, copy=False)
        if not np.isfinite(payload[name]).all():
            raise RuntimeError(f"Native HDMI rollout contains non-finite {name}")
    payload["frame_indices"] = np.arange(len(frames), dtype=np.int64)
    return payload


def _float_rows(value) -> list[list[float]]:
    return value.detach().cpu().tolist()


def _collision_group(stage, path: str) -> dict[str, object]:
    from pxr import UsdPhysics

    group = UsdPhysics.CollisionGroup.Get(stage, path)
    if not group or not group.GetPrim().IsValid():
        raise RuntimeError(f"Missing ARCTIC collision group: {path}")
    return {
        "path": path,
        "includes": [str(target) for target in group.GetIncludesRel().GetTargets()],
        "filters": [str(target) for target in group.GetFilteredGroupsRel().GetTargets()],
    }


def _cuboid_size(stage, root_path: str) -> list[float]:
    from pxr import Gf, Usd, UsdGeom

    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        raise RuntimeError(f"Missing ARCTIC support root: {root_path}")
    cubes = [prim for prim in Usd.PrimRange(root) if prim.IsA(UsdGeom.Cube)]
    if len(cubes) != 1:
        raise RuntimeError(f"ARCTIC support must contain one USD cube: {root_path}")
    cube = UsdGeom.Cube(cubes[0])
    side = float(cube.GetSizeAttr().Get())
    transform = UsdGeom.Xformable(cube).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    return [
        side * transform.TransformDir(Gf.Vec3d(*axis)).GetLength()
        for axis in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    ]


def _write_support_readback(env, output_value: str) -> None:
    from active_adaptation.assets.arthoi4d import (
        ARTHOI4D_HAND_COLLISION_GROUP_PATH,
        ARTHOI4D_TABLE_COLLISION_GROUP_PATH,
    )

    base_env = env.base_env
    stage = base_env.scene.stage
    supports: dict[str, object] = {}
    for name in base_env.cfg.static_support_names:
        asset = base_env.scene.rigid_objects[str(name)]
        supports[str(name)] = {
            "position": asset.data.root_pos_w[0].detach().cpu().tolist(),
            "size": _cuboid_size(stage, f"{base_env.scene.env_prim_paths[0]}/{name}"),
            "material": _float_rows(asset.root_physx_view.get_material_properties()),
        }
    report = {
        "support": supports,
        "collision_groups": {
            "table": _collision_group(stage, ARTHOI4D_TABLE_COLLISION_GROUP_PATH),
            "hands": _collision_group(stage, ARTHOI4D_HAND_COLLISION_GROUP_PATH),
        },
    }
    output_path = Path(output_value).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"[arthoi4d-export] wrote support readback: {output_path}", flush=True)


@hydra.main(config_path="../cfg", config_name="eval", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run one checkpoint headlessly and save its actual native trajectory."""

    OmegaConf.set_struct(cfg, False)
    bundle = _input_bundle()
    _seed_hdmi_environment(bundle)
    output_value = cfg.get(
        "native_rollout_path", os.environ.get("ARTHOI4D_HDMI_NATIVE_ROLLOUT_PATH")
    )
    if not output_value:
        raise RuntimeError(
            "Set +native_rollout_path=/absolute/path/native_rollout.npz or "
            "ARTHOI4D_HDMI_NATIVE_ROLLOUT_PATH"
        )
    output_path = Path(str(output_value)).expanduser().resolve()
    if int(cfg.task.num_envs) != 1:
        raise ValueError("ArtHOI4D native rollout export requires task.num_envs=1")
    if not bool(cfg.headless):
        raise ValueError("ArtHOI4D native rollout export must run headless")
    if bool(cfg.get("eval_render", False)):
        raise ValueError("ArtHOI4D native rollout export must not enable rendering")

    kit_args = os.environ.get("ARTHOI4D_HDMI_KIT_ARGS")
    if kit_args:
        cfg.app.kit_args = kit_args

    print("[arthoi4d-export] stage=app-launch", flush=True)

    # AppLauncher pins active_gpu/physics_gpu during config resolution.  On
    # headless machines without a Vulkan ICD that can fail before PhysX is
    # created, while the direct SimulationApp path is sufficient for the
    # physics-only exporter used here.
    if os.environ.get("ARTHOI4D_HDMI_DIRECT_SIM_APP", "1").lower() not in {"0", "false", "no"}:
        device = str(cfg.app.get("device", os.environ.get("ARTHOI4D_HDMI_APP_DEVICE", "cuda:0")))
        gpu_id = int(device.split(":", 1)[1]) if device.startswith("cuda:") else 0
        simulation_app = SimulationApp(
            {
                "headless": True,
                "enable_cameras": False,
                "renderer": "RaytracedLighting",
                "disable_viewport_updates": True,
                "active_gpu": gpu_id,
                "physics_gpu": gpu_id,
            },
            experience=os.environ.get("ARTHOI4D_HDMI_EXPERIENCE", ""),
        )
        asset_root = os.environ.get("ARTHOI4D_HDMI_ASSET_ROOT")
        if asset_root:
            import carb

            carb.settings.get_settings().set(
                "/persistent/isaac/asset_root/cloud", asset_root
            )
    else:
        app_launcher = AppLauncher(OmegaConf.to_container(cfg.app, resolve=True))
        simulation_app = app_launcher.app
    try:
        print("[arthoi4d-export] stage=make_env_policy", flush=True)
        env, policy, _ = make_env_policy(cfg)
        print("[arthoi4d-export] stage=env-ready", flush=True)
        support_readback_path = os.environ.get("ARTHOI4D_HDMI_SUPPORT_READBACK_PATH")
        if support_readback_path:
            _write_support_readback(env, support_readback_path)
        command = env.base_env.command_manager
        human_body_names = tuple(str(name) for name in bundle["human_body_names"])
        human_joint_names = tuple(str(name) for name in bundle["human_joint_names"])
        human_body_ids = _indices(human_body_names, command.asset.body_names, "body")
        human_joint_ids = _indices(human_joint_names, command.asset.joint_names, "joint")
        if tuple(command.contact_region_link_names) != tuple(
            bundle["contact_region_link_names"]
        ):
            raise RuntimeError("HDMI native contact-region links differ from the input bundle")

        requested_steps = cfg.get("rollout_steps", None)
        max_steps = (
            int(command.dataset.lengths[0])
            if requested_steps is None
            else int(requested_steps)
        )
        if max_steps <= 0:
            raise ValueError("rollout_steps must be positive")
        policy_eval = policy.get_rollout_policy("eval")
        env.base_env.eval()
        env.eval()
        env.set_seed(int(cfg.seed))
        print(f"[arthoi4d-export] stage=rollout max_steps={max_steps}", flush=True)
        tensordict = env.reset()
        print("[arthoi4d-export] stage=reset-ok", flush=True)
        frames: list[dict[str, torch.Tensor]] = []
        with torch.inference_mode(), set_exploration_type(ExplorationType.MODE):
            for _ in range(max_steps):
                action_tensordict = policy_eval(tensordict)
                transition = env.step(action_tensordict)
                frames.append(_snapshot(command, bundle, human_body_ids, human_joint_ids))
                if bool(transition["next", "done"][0].item()):
                    break
                tensordict = transition["next"]

        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, **_stack(frames))
        print(f"Saved ArtHOI4D native rollout: {output_path}")
        print("[arthoi4d-export] stage=done", flush=True)
        os._exit(0)
    except BaseException:
        traceback.print_exc()
        os._exit(1)
        raise
    finally:
        if "env" in locals():
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
