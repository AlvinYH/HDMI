"""Runtime assets for the direct ArtHOI4D HDMI fork integration."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
import xml.etree.ElementTree as ET

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.assets.rigid_object import RigidObjectCfg
from isaaclab.terrains import TerrainImporterCfg
from isaacsim.core.utils.extensions import enable_extension
from isaaclab.sim.converters import (
    MjcfConverter,
    MjcfConverterCfg,
    UrdfConverter,
    UrdfConverterCfg,
)
from pxr import Sdf, Usd, UsdPhysics


def _input() -> tuple[dict, Path]:
    path = Path(os.environ["ARTHOI4D_HDMI_INPUT_PATH"]).expanduser().resolve()
    values = json.loads(path.read_text(encoding="utf-8"))
    return values, path.parent


def _safe_identifier(value: str, *, fallback: str = "asset") -> str:
    value = re.sub(r"[^A-Za-z0-9_]", "_", value).strip("_")
    value = re.sub(r"_+", "_", value)
    if not value:
        value = fallback
    if value[0].isdigit():
        value = f"{fallback}_{value}"
    return value


def _dedupe_name(value: str, counts: dict[str, int]) -> str:
    count = counts.get(value, 0)
    counts[value] = count + 1
    return value if count == 0 else f"{value}_{count}"


def _sanitize_mtl(source: Path, destination: Path) -> None:
    if not source.is_file():
        return
    lines: list[str] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if line.startswith("newmtl "):
            lines.append(f"newmtl {_safe_identifier(line[7:].strip(), fallback='material')}")
        else:
            lines.append(line)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _sanitize_obj(source: Path, destination: Path) -> None:
    lines: list[str] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if line.startswith("mtllib "):
            safe_mtls: list[str] = []
            for raw_name in line.split()[1:]:
                mtl_source = (source.parent / raw_name).resolve()
                mtl_destination = source.parent / (
                    "arthoi4d_" + _safe_identifier(Path(raw_name).stem, fallback="mtl") + Path(raw_name).suffix
                )
                _sanitize_mtl(mtl_source, mtl_destination)
                safe_mtls.append(mtl_destination.name)
            lines.append("mtllib " + " ".join(safe_mtls))
        elif line.startswith("usemtl "):
            lines.append(f"usemtl {_safe_identifier(line[7:].strip(), fallback='material')}")
        elif line.startswith("o ") or line.startswith("g "):
            prefix, names = line[:2], line[2:].split()
            safe_names = [_safe_identifier(name, fallback="mesh") for name in names]
            lines.append(prefix + " ".join(safe_names))
        else:
            lines.append(line)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _sanitize_mesh_reference(urdf_path: Path, filename: str) -> str:
    source = (urdf_path.parent / filename).resolve()
    if not source.is_file() or source.suffix.lower() != ".obj":
        return filename
    destination = source.parent / (
        "arthoi4d_" + _safe_identifier(source.stem, fallback="mesh") + source.suffix
    )
    if source == destination:
        return filename
    _sanitize_obj(source, destination)
    return os.path.relpath(destination, urdf_path.parent).replace(os.sep, "/")


def _patch_object_urdf(urdf_path: Path) -> Path:
    patched = urdf_path.with_name(f"{urdf_path.stem}_arthoi4d.urdf")
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    child_links: set[str] = set()
    root_has_child = False
    first_joint_parent: str | None = None
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        parent_link = parent.get("link") if parent is not None else None
        child_link = child.get("link") if child is not None else None
        if parent_link == "base":
            root_has_child = True
        if child_link:
            child_links.add(child_link)
        if first_joint_parent is None and parent_link and parent_link != "base":
            first_joint_parent = parent_link

    if not root_has_child and first_joint_parent and first_joint_parent not in child_links:
        joint = ET.Element("joint", {"name": f"base_to_{first_joint_parent}", "type": "fixed"})
        ET.SubElement(joint, "parent", {"link": "base"})
        ET.SubElement(joint, "child", {"link": first_joint_parent})
        root.insert(1, joint)

    scoped_counts: dict[tuple[str, str], dict[str, int]] = {}
    for link in root.findall("link"):
        link_name = link.get("name", "link")
        for tag in ("visual", "collision"):
            counts = scoped_counts.setdefault((link_name, tag), {})
            for index, element in enumerate(link.findall(tag)):
                raw_name = element.get("name", f"{tag}_{index}")
                element.set(
                    "name",
                    _dedupe_name(_safe_identifier(raw_name, fallback=tag), counts),
                )

    for mesh in root.findall(".//mesh"):
        filename = mesh.get("filename")
        if filename:
            mesh.set("filename", _sanitize_mesh_reference(urdf_path, filename))

    ET.indent(tree, space="  ")
    tree.write(patched, encoding="utf-8", xml_declaration=True)
    return patched


INPUT, INPUT_ROOT = _input()
CACHE = (INPUT_ROOT / INPUT["usd_cache_dir"]).resolve()
CACHE.mkdir(parents=True, exist_ok=True)
enable_extension("isaacsim.asset.importer.mjcf")
WORLD = INPUT["world"]
GROUND = WORLD["ground"]
MATERIALS = WORLD["materials"]
PHYSX = WORLD["physx"]
OBJECT = WORLD["object"]

human_usd = MjcfConverter(
    MjcfConverterCfg(
        asset_path=str((INPUT_ROOT / INPUT["human_mjcf"]).resolve()),
        usd_dir=str(CACHE / "human"),
        usd_file_name="smplx.usd",
        fix_base=False,
        self_collision=False,
        make_instanceable=True,
    )
).usd_path

object_urdf = _patch_object_urdf((INPUT_ROOT / INPUT["object_urdf"]).resolve())

object_usd = UrdfConverter(
    UrdfConverterCfg(
        asset_path=str(object_urdf),
        usd_dir=str(CACHE / "object"),
        usd_file_name="object.usd",
        fix_base=bool(INPUT["object_fix_base"]),
        root_link_name=INPUT["object_body_names"][0],
        merge_fixed_joints=False,
        self_collision=False,
        make_instanceable=True,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            target_type="none",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=0.0,
                damping=0.0,
            ),
        ),
    )
).usd_path


def _rigid_body_prim_paths(
    usd_path: str,
    requested_names: list[str],
) -> dict[str, str]:
    """Map simulator body names to paths below the converted asset root."""

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"Could not open converted USD: {usd_path}")
    default_prim = stage.GetDefaultPrim()
    default_prefix = ""
    if default_prim and default_prim.IsValid():
        default_prefix = default_prim.GetPath().pathString.rstrip("/") + "/"
    requested = set(requested_names)
    matches: dict[str, list[str]] = {name: [] for name in requested_names}
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        name = prim.GetName()
        if name not in requested:
            continue

        body_prim = prim
        while body_prim and not body_prim.HasAPI(UsdPhysics.RigidBodyAPI):
            body_prim = body_prim.GetParent()
        if body_prim is None:
            body_path = str(prim.GetPath())
        else:
            body_path = str(body_prim.GetPath())
        matches[name].append(body_path)
    selected: dict[str, str] = {}
    for name, paths in matches.items():
        unique_paths = [path for path in dict.fromkeys(paths) if path]
        if not unique_paths:
            raise RuntimeError(
                f"Converted USD rigid-body paths are not unique/complete: {matches}"
            )
        selected[name] = max(unique_paths, key=lambda path: (path.count("/"), len(path)))
    return {
        name: path[len(default_prefix) :] if default_prefix and path.startswith(default_prefix) else path.lstrip("/")
        for name, path in selected.items()
    }


ARTHOI4D_ROBOT_BODY_PRIM_PATHS = _rigid_body_prim_paths(
    human_usd,
    list(INPUT["human_body_names"]),
)
ARTHOI4D_OBJECT_BODY_PRIM_PATHS = _rigid_body_prim_paths(
    object_usd,
    list(INPUT["object_body_names"]),
)

_rigid_props = sim_utils.RigidBodyPropertiesCfg(
    disable_gravity=False,
    retain_accelerations=False,
    linear_damping=0.0,
    angular_damping=0.0,
    max_linear_velocity=1000.0,
    max_angular_velocity=1000.0,
    max_depenetration_velocity=float(PHYSX["max_depenetration_velocity"]),
)

# The object has canonical URDF damping.  Human actuator and rigid-body
# damping remain native-control settings; shared contact solver values are
# set explicitly on both articulations below.
_object_rigid_props = sim_utils.RigidBodyPropertiesCfg(
    disable_gravity=bool(OBJECT["disable_gravity"]),
    retain_accelerations=False,
    linear_damping=float(OBJECT["linear_damping"]),
    angular_damping=float(OBJECT["angular_damping"]),
    max_linear_velocity=1000.0,
    max_angular_velocity=1000.0,
    max_depenetration_velocity=float(PHYSX["max_depenetration_velocity"]),
)


def _object_joint_dynamics(field: str) -> dict[str, float]:
    names = list(INPUT["object_joint_names"])
    values = list(INPUT[field])
    if len(names) != len(values):
        raise ValueError(f"{field} must have one value per object joint")
    if len(set(names)) != len(names):
        raise ValueError("object joint names must be unique")
    return {name: float(value) for name, value in zip(names, values, strict=True)}


_object_joint_damping = _object_joint_dynamics("object_joint_damping")
_object_joint_friction = _object_joint_dynamics("object_joint_friction")

_matched_material = sim_utils.RigidBodyMaterialCfg(
    friction_combine_mode="multiply",
    restitution_combine_mode="multiply",
    static_friction=float(MATERIALS["shape_friction"]),
    dynamic_friction=float(MATERIALS["shape_friction"]),
    restitution=float(MATERIALS["restitution"]),
)

_matched_collision = sim_utils.CollisionPropertiesCfg(
    collision_enabled=True,
    contact_offset=float(PHYSX["contact_offset"]),
    rest_offset=float(PHYSX["rest_offset"]),
)


ARTHOI4D_TERRAIN = TerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="plane",
    physics_material=sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=float(GROUND["static_friction"]),
        dynamic_friction=float(GROUND["dynamic_friction"]),
        restitution=float(GROUND["restitution"]),
    ),
    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
)


def _support_cfg(box: dict, index: int) -> RigidObjectCfg:
    """Build one fixed cuboid from the existing Studio case JSON box."""

    center = tuple(float(value) for value in box["center"])
    size = tuple(float(value) for value in box["size"])
    color = tuple(float(value) for value in box["color"])
    if len(center) != 3 or len(size) != 3 or len(color) != 3:
        raise ValueError("ARCTIC support center, size, and color must each have three values")
    if min(size) <= 0.0:
        raise ValueError("ARCTIC support size must be positive")
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/arthoi4d_support_{index}",
        spawn=sim_utils.CuboidCfg(
            size=size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
                linear_damping=0.0,
                angular_damping=0.0,
                max_depenetration_velocity=float(PHYSX["max_depenetration_velocity"]),
            ),
            collision_props=_matched_collision,
            physics_material=_matched_material,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=center),
    )


ARTHOI4D_SUPPORT_CFGS = {
    f"arthoi4d_support_{index}": _support_cfg(box, index)
    for index, box in enumerate(INPUT["support_boxes"])
}


# CoDA uses filter value 10 on its table actor.  Its SMPL-X shape filters use
# that value only for both wrists and their finger segments, so these are the
# only human bodies that must not collide with the shared table.
_CODA_TABLE_FILTER = 10
_CODA_FILTERED_HAND_BODIES = (
    "L_Wrist",
    "L_Index1",
    "L_Index2",
    "L_Index3",
    "L_Middle1",
    "L_Middle2",
    "L_Middle3",
    "L_Pinky1",
    "L_Pinky2",
    "L_Pinky3",
    "L_Ring1",
    "L_Ring2",
    "L_Ring3",
    "L_Thumb1",
    "L_Thumb2",
    "L_Thumb3",
    "R_Wrist",
    "R_Index1",
    "R_Index2",
    "R_Index3",
    "R_Middle1",
    "R_Middle2",
    "R_Middle3",
    "R_Pinky1",
    "R_Pinky2",
    "R_Pinky3",
    "R_Ring1",
    "R_Ring2",
    "R_Ring3",
    "R_Thumb1",
    "R_Thumb2",
    "R_Thumb3",
)
_COLLISION_GROUP_ROOT = "/World/arthoi4d_collision_groups"
ARTHOI4D_TABLE_COLLISION_GROUP_PATH = (
    f"{_COLLISION_GROUP_ROOT}/table_filter_{_CODA_TABLE_FILTER}"
)
ARTHOI4D_HAND_COLLISION_GROUP_PATH = (
    f"{_COLLISION_GROUP_ROOT}/hands_filter_{_CODA_TABLE_FILTER}"
)


def _collision_prim_paths(stage: Usd.Stage, root_path: str) -> list[Sdf.Path]:
    """Return every collider below one spawned rigid body."""

    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        raise RuntimeError(f"Missing ARCTIC collision root: {root_path}")
    paths = [
        prim.GetPath()
        for prim in Usd.PrimRange(root)
        if prim.HasAPI(UsdPhysics.CollisionAPI)
    ]
    if not paths:
        raise RuntimeError(f"ARCTIC collision root has no collider: {root_path}")
    return paths


def _set_collision_group_targets(
    group: UsdPhysics.CollisionGroup,
    targets: list[Sdf.Path],
) -> None:
    group.CreateIncludesRel().SetTargets(list(dict.fromkeys(targets)))


def apply_arthoi4d_table_collision_filter(scene) -> None:
    """Install CoDA's table filter before HDMI advances the simulator."""

    if not ARTHOI4D_SUPPORT_CFGS:
        return
    if int(WORLD["table_collision_filter"]) != _CODA_TABLE_FILTER:
        raise ValueError(
            "ARCTIC table collision filter must match CoDA value "
            f"{_CODA_TABLE_FILTER}"
        )

    import omni.usd

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("Isaac Sim USD stage is unavailable for ARCTIC collision filtering")
    missing = sorted(set(_CODA_FILTERED_HAND_BODIES) - set(ARTHOI4D_ROBOT_BODY_PRIM_PATHS))
    if missing:
        raise RuntimeError(f"HDMI SMPL-X conversion is missing CoDA hand bodies: {missing}")

    stage.DefinePrim(_COLLISION_GROUP_ROOT, "Scope")
    table_group = UsdPhysics.CollisionGroup.Define(
        stage, ARTHOI4D_TABLE_COLLISION_GROUP_PATH
    )
    hand_group = UsdPhysics.CollisionGroup.Define(
        stage, ARTHOI4D_HAND_COLLISION_GROUP_PATH
    )
    table_targets: list[Sdf.Path] = []
    hand_targets: list[Sdf.Path] = []
    for env_path in scene.env_prim_paths:
        for support_name in ARTHOI4D_SUPPORT_CFGS:
            table_targets.extend(
                _collision_prim_paths(stage, f"{env_path}/{support_name}")
            )
        for body_name in _CODA_FILTERED_HAND_BODIES:
            body_path = ARTHOI4D_ROBOT_BODY_PRIM_PATHS[body_name]
            hand_targets.extend(
                _collision_prim_paths(stage, f"{env_path}/Robot/{body_path}")
            )
    _set_collision_group_targets(table_group, table_targets)
    _set_collision_group_targets(hand_group, hand_targets)
    table_group.CreateFilteredGroupsRel().SetTargets([hand_group.GetPath()])
    hand_group.CreateFilteredGroupsRel().SetTargets([table_group.GetPath()])

ARTHOI4D_SMPLX_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    articulation_root_prim_path=f"/{INPUT['human_body_names'][0]}",
    spawn=sim_utils.UsdFileCfg(
        usd_path=human_usd,
        activate_contact_sensors=True,
        rigid_props=_rigid_props,
        collision_props=_matched_collision,
        physics_material=_matched_material,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=int(PHYSX["num_position_iterations"]),
            solver_velocity_iteration_count=int(PHYSX["num_velocity_iterations"]),
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.95,
    actuators={
        "smplx": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit_sim=300.0,
            velocity_limit_sim=100.0,
            stiffness=80.0,
            damping=4.0,
            armature=0.01,
        )
    },
)

ARTHOI4D_OBJECT_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/arthoi4d_object",
    spawn=sim_utils.UsdFileCfg(
        usd_path=object_usd,
        activate_contact_sensors=True,
        rigid_props=_object_rigid_props,
        collision_props=_matched_collision,
        physics_material=_matched_material,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=int(PHYSX["num_position_iterations"]),
            solver_velocity_iteration_count=int(PHYSX["num_velocity_iterations"]),
            fix_root_link=bool(INPUT["object_fix_base"]),
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=tuple(float(value) for value in INPUT["object_initial_root_pos"]),
        rot=tuple(float(value) for value in INPUT["object_initial_root_quat_wxyz"]),
        # The command manager writes the complete q(t), qdot(t) state before
        # every simulation reset.  This static spawn state is deliberately
        # reference-phase independent.
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    actuators={
        "passive": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit_sim=1000.0,
            velocity_limit_sim=100.0,
            stiffness=0.0,
            # These per-joint values come directly from the common URDF;
            # do not replace them with an HDMI-wide damping/friction value.
            damping=_object_joint_damping,
            friction=_object_joint_friction,
        )
    },
)
