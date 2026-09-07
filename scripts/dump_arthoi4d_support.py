"""Build one HDMI ARCTIC environment and print its table actor readback."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import hydra
from omegaconf import OmegaConf

from isaaclab.app import AppLauncher


def _input_bundle() -> dict:
    path_value = os.environ.get("ARTHOI4D_HDMI_INPUT_PATH")
    if not path_value:
        raise RuntimeError("ARTHOI4D_HDMI_INPUT_PATH is required")
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"HDMI input bundle is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _seed_hdmi_environment(bundle: dict) -> None:
    input_root = Path(os.environ["ARTHOI4D_HDMI_INPUT_PATH"]).expanduser().resolve().parent
    os.environ.setdefault("ARTHOI4D_HDMI_MOTION_DIR", str(input_root / bundle["motion_dir"]))
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


def _float_rows(value) -> list[list[float]]:
    """Convert a simulator material tensor to JSON-safe rows."""

    return value.detach().cpu().tolist()


def _collision_group(stage, path: str) -> dict[str, object]:
    """Read one authored USD collision group without modifying the stage."""

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
    """Measure the spawned USD cuboid after all authored scales are applied."""

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


@hydra.main(config_path="../cfg", config_name="eval", version_base=None)
def main(cfg) -> None:
    """Reuse HDMI's standard task configuration with exactly one environment."""

    OmegaConf.set_struct(cfg, False)
    _seed_hdmi_environment(_input_bundle())
    cfg.headless = True
    cfg.eval_render = False
    cfg.app.enable_cameras = False
    cfg.task.num_envs = 1
    print("[arthoi4d-support] stage=app-launch", flush=True)
    app_launcher = AppLauncher(OmegaConf.to_container(cfg.app, resolve=True))
    simulation_app = app_launcher.app
    env = None
    try:
        from active_adaptation.assets.arthoi4d import (
            ARTHOI4D_HAND_COLLISION_GROUP_PATH,
            ARTHOI4D_TABLE_COLLISION_GROUP_PATH,
        )
        from scripts.helpers import make_env_policy

        print("[arthoi4d-support] stage=make_env", flush=True)
        env, _, _ = make_env_policy(cfg)
        print("[arthoi4d-support] stage=env-ready", flush=True)
        stage = env.scene.stage
        supports: dict[str, object] = {}
        for name in cfg.task.static_support_names:
            asset = env.scene.rigid_objects[str(name)]
            supports[str(name)] = {
                "position": asset.data.root_pos_w[0].detach().cpu().tolist(),
                "size": _cuboid_size(
                    stage,
                    f"{env.scene.env_prim_paths[0]}/{name}",
                ),
                "material": _float_rows(asset.root_physx_view.get_material_properties()),
            }
        report = {
            "support": supports,
            "collision_groups": {
                "table": _collision_group(stage, ARTHOI4D_TABLE_COLLISION_GROUP_PATH),
                "hands": _collision_group(stage, ARTHOI4D_HAND_COLLISION_GROUP_PATH),
            },
        }
        print("[arthoi4d-support] stage=report-ready", flush=True)
        print(json.dumps(report, indent=2, sort_keys=True))
        sys.stdout.flush()
        os._exit(0)
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
