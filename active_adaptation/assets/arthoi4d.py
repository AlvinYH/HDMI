"""Runtime assets for the direct ArtHOI4D HDMI fork integration."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.assets.rigid_object import RigidObjectCfg
from isaaclab.terrains import TerrainImporterCfg
from pxr import PhysxSchema, Sdf, Usd, UsdPhysics


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
# The source XML used for conversion is deliberately different from the
# authored MJCF limits (see ``convert_smplx_mjcf_to_usd`` below).  Keep this
# cache namespace separate so ``REUSE=1`` can never load a pre-policy USD.
CACHE = CACHE / "principal_rotvec_limits_pelvis_root_v2"
# A torchrun job starts one Isaac process per rank.  Keep generated USD files
# rank-local on first conversion so the workers never race while creating the
# same cache.  Single-process HDMI retains the existing cache location.
if os.environ.get("ARTHOI4D_HDMI_USD_CACHE_PER_RANK", "0").lower() in {
    "1",
    "true",
    "yes",
}:
    CACHE = CACHE / f"rank{int(os.environ.get('LOCAL_RANK', '0'))}"
CACHE.mkdir(parents=True, exist_ok=True)


def _articulation_root_prim_path(
    usd_path: str | Path,
    root_body_name: str,
) -> str:
    """Validate the single Pelvis rigid articulation root and return its path."""

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"Cannot open SMPL-X USD: {usd_path}")

    articulation_roots = [
        prim
        for prim in stage.TraverseAll()
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    ]

    if len(articulation_roots) != 1:
        raise RuntimeError(
            "SMPL-X USD must contain exactly one articulation root; "
            f"found {[str(prim.GetPath()) for prim in articulation_roots]}"
        )

    articulation_root = articulation_roots[0]

    if articulation_root.GetName() != root_body_name:
        raise RuntimeError(
            f"Expected articulation root named {root_body_name!r}, "
            f"got {articulation_root.GetName()!r} at {articulation_root.GetPath()}"
        )

    if not articulation_root.HasAPI(UsdPhysics.RigidBodyAPI):
        raise RuntimeError(
            f"SMPL-X articulation root must be the {root_body_name} rigid body, got "
            f"{articulation_root.GetPath()}"
        )

    if not articulation_root.HasAPI(PhysxSchema.PhysxArticulationAPI):
        raise RuntimeError(
            "SMPL-X Pelvis root is missing PhysxArticulationAPI: "
            f"{articulation_root.GetPath()}"
        )

    # PhysX expects the articulation API on the importer wrapper, while
    # IsaacLab's ArticulationCfg path is relative to the USD default prim.
    default_prim = stage.GetDefaultPrim()
    prefix = (default_prim.GetPath().pathString.rstrip("/") + "/") if default_prim else ""
    relative_path = str(articulation_root.GetPath())[len(prefix):].lstrip("/")
    # ArticulationCfg concatenates this value to cfg.prim_path, and IsaacLab
    # requires the explicit child path to begin with a slash.
    return "/" + relative_path


def convert_smplx_mjcf_to_usd(
    *,
    input_xml: str | Path,
    output_usd: str | Path,
    root_body_name: str,
) -> tuple[str, str]:
    """Convert one SMPL-X MJCF into one valid Isaac Lab articulation."""

    import omni.kit.app
    from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg

    app = omni.kit.app.get_app()
    extension_manager = app.get_extension_manager()
    extension_manager.set_extension_enabled_immediate(
        "isaacsim.asset.importer.mjcf",
        True,
    )
    app.update()

    input_xml = Path(input_xml).expanduser().resolve()
    output_usd = Path(output_usd).expanduser().resolve()

    if not input_xml.is_file():
        raise FileNotFoundError(input_xml)

    output_usd.parent.mkdir(parents=True, exist_ok=True)

    tree = ET.parse(input_xml)
    root = tree.getroot()
    worldbody = root.find("worldbody")

    if worldbody is None:
        raise ValueError(f"MJCF has no worldbody: {input_xml}")

    # Human asset must not contain floor, lights, tables, or other scene items.
    for child in list(worldbody):
        if child.tag in {"geom", "light"}:
            worldbody.remove(child)

    root_bodies = worldbody.findall("body")
    if len(root_bodies) != 1:
        raise ValueError(
            f"SMPL-X MJCF must have exactly one root body, "
            f"got {len(root_bodies)}"
        )

    actual_root_name = root_bodies[0].get("name")
    if actual_root_name != root_body_name:
        raise ValueError(
            f"Expected root body {root_body_name!r}, "
            f"got {actual_root_name!r}"
        )

    bodies = list(root.iter("body"))
    joints = [
        joint
        for joint in root.iter("joint")
        if joint.get("name")
        and joint.get("type", "hinge") != "free"
    ]

    if len(bodies) != 52:
        raise RuntimeError(
            f"Expected 52 SMPL-X bodies, got {len(bodies)}"
        )

    if len(joints) != 153:
        raise RuntimeError(
            f"Expected 153 SMPL-X joints, got {len(joints)}"
        )

    body_names = [body.get("name") for body in bodies]
    joint_names = [joint.get("name") for joint in joints]

    if len(set(body_names)) != len(body_names):
        raise RuntimeError("SMPL-X MJCF contains duplicate body names")

    if len(set(joint_names)) != len(joint_names):
        raise RuntimeError("SMPL-X MJCF contains duplicate joint names")

    # MJCF defaults to degrees unless compiler angle="radian".
    compiler = root.find("compiler")
    angle_unit = (
        "degree"
        if compiler is None
        else compiler.get("angle", "degree")
    )
    if angle_unit not in {"degree", "radian"}:
        raise ValueError(
            f"Unsupported MJCF angle unit: {angle_unit!r}"
        )

    full_rotation_limit = np.pi if angle_unit == "radian" else 180.0
    widened_joint_names: list[str] = []

    for joint in joints:
        raw_range = joint.get("range")
        if raw_range is None:
            raise ValueError(
                f"SMPL-X joint has no range: {joint.get('name')}"
            )

        values = raw_range.split()
        if len(values) != 2:
            raise ValueError(
                f"Invalid range for {joint.get('name')}: {raw_range!r}"
            )

        lower, upper = map(float, values)
        lower = max(lower, -full_rotation_limit)
        upper = min(upper, full_rotation_limit)

        if lower >= upper:
            raise ValueError(
                f"Invalid joint range after clamping: "
                f"{joint.get('name')} [{lower}, {upper}]"
            )

        # HDMI's 153 joint references are principal axis-angle components,
        # not MuJoCo Euler hinge coordinates.  The authored 5.625 degree
        # finger ranges therefore reject valid source motion after the MJCF
        # importer creates scalar USD limits.  Keep the authored XML intact;
        # only this conversion-only copy uses the finite [-pi, pi] envelope
        # guaranteed by _quat_rotvec() for each component.
        if lower > -full_rotation_limit or upper < full_rotation_limit:
            widened_joint_names.append(str(joint.get("name")))
            lower, upper = -full_rotation_limit, full_rotation_limit

        joint.set("range", f"{lower:.9g} {upper:.9g}")

    print(
        "[SMPL-X MJCF source] "
        f"source={input_xml} "
        f"policy=principal_rotvec_finite_envelope "
        f"widened_joint_ranges={len(widened_joint_names)}",
        flush=True,
    )

    with tempfile.TemporaryDirectory(
        prefix="smplx_mjcf_",
        dir=output_usd.parent,
    ) as temporary_dir:
        prepared_xml = Path(temporary_dir) / input_xml.name
        tree.write(
            prepared_xml,
            encoding="utf-8",
            xml_declaration=True,
        )

        converter = MjcfConverter(
            MjcfConverterCfg(
                asset_path=str(prepared_xml),
                usd_dir=str(output_usd.parent),
                usd_file_name=output_usd.name,
                force_usd_conversion=True,
                make_instanceable=True,
                fix_base=False,
                import_sites=True,

                # Keep the current HDMI benchmark collision setting.
                # Self-collision is unrelated to articulation ordering.
                self_collision=False,
            )
        )

    result = Path(converter.usd_path).expanduser().resolve()

    if result != output_usd:
        raise RuntimeError(
            f"Isaac Lab returned {result}, expected {output_usd}"
        )

    if not result.is_file():
        raise RuntimeError(
            f"Isaac Lab did not generate the expected USD: {result}"
        )

    stage = Usd.Stage.Open(str(result))
    if stage is None:
        raise RuntimeError(f"Cannot open generated USD: {result}")

    articulation_roots = [
        prim
        for prim in stage.TraverseAll()
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    ]

    wrappers = [
        prim
        for prim in articulation_roots
        if prim.GetName() == root_body_name
        and stage.GetPrimAtPath(prim.GetPath().AppendChild(root_body_name)).IsValid()
        and stage.GetPrimAtPath(prim.GetPath().AppendChild(root_body_name)).HasAPI(UsdPhysics.RigidBodyAPI)
    ]

    if len(wrappers) != 1:
        raise RuntimeError(
            f"Expected one importer articulation wrapper named "
            f"{root_body_name!r}; found "
            f"{[str(prim.GetPath()) for prim in wrappers]}; "
            f"all articulation roots are "
            f"{[str(prim.GetPath()) for prim in articulation_roots]}"
        )

    importer_wrapper = wrappers[0]
    physical_root = stage.GetPrimAtPath(
        importer_wrapper.GetPath().AppendChild(root_body_name)
    )
    physical_root_path = str(physical_root.GetPath())
    default_prim = stage.GetDefaultPrim()
    default_prefix = (
        default_prim.GetPath().pathString.rstrip("/") + "/"
        if default_prim and default_prim.IsValid()
        else ""
    )
    physical_root_relative_path = (
        "/"
        + physical_root_path[len(default_prefix) :].lstrip("/")
        if default_prefix and physical_root_path.startswith(default_prefix)
        else "/" + physical_root_path.lstrip("/")
    )

    # The MJCF importer puts the root API on an Xform wrapper.  PhysX then
    # selects an arbitrary descendant (Chest here) as its actor root even
    # though the joint graph is Pelvis-rooted.  Author the complete API pair
    # on the actual free-joint rigid body instead; the runtime API can then
    # consistently reset and observe the ARCTIC Pelvis root.
    for prim in articulation_roots:
        prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        if prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
            prim.RemoveAPI(PhysxSchema.PhysxArticulationAPI)

    UsdPhysics.ArticulationRootAPI.Apply(physical_root)
    PhysxSchema.PhysxArticulationAPI.Apply(physical_root)

    stage.GetRootLayer().Save()

    # Reopen the saved result instead of trusting the in-memory stage.
    articulation_root_prim_path = _articulation_root_prim_path(
        result,
        root_body_name,
    )

    if articulation_root_prim_path != physical_root_relative_path:
        raise RuntimeError(
            f"SMPL-X articulation root changed after saving: "
            f"before={physical_root_relative_path}, "
            f"after={articulation_root_prim_path}"
        )

    print(
        "[SMPL-X USD] "
        f"source={input_xml} "
        f"output={result} "
        f"bodies={len(bodies)} "
        f"joints={len(joints)} "
        f"articulation_root={articulation_root_prim_path}",
        flush=True,
    )

    return str(result), articulation_root_prim_path


WORLD = INPUT["world"]
GROUND = WORLD["ground"]
MATERIALS = WORLD["materials"]
PHYSX = WORLD["physx"]
OBJECT = WORLD["object"]

human_usd_cache = CACHE / "human" / "smplx.usd"
object_usd_cache = CACHE / "object" / "object.usd"

reuse_usd_cache = (
    os.environ.get("ARTHOI4D_HDMI_REUSE_USD_CACHE", "1").lower()
    not in {"0", "false", "no"}
)

if reuse_usd_cache:
    missing = [
        str(path)
        for path in (human_usd_cache, object_usd_cache)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "USD cache reuse was requested, but files are missing: "
            + ", ".join(missing)
        )

    human_usd = str(human_usd_cache)
    object_usd = str(object_usd_cache)

    # Cached USD must obey the same articulation contract.
    articulation_root_prim_path = _articulation_root_prim_path(
        human_usd,
        INPUT["human_body_names"][0],
    )
else:
    human_usd, articulation_root_prim_path = (
        convert_smplx_mjcf_to_usd(
            input_xml=(
                INPUT_ROOT / INPUT["human_mjcf"]
            ).resolve(),
            output_usd=human_usd_cache,
            root_body_name=INPUT["human_body_names"][0],
        )
    )

    from isaaclab.sim.converters import (
        UrdfConverter,
        UrdfConverterCfg,
    )

    object_urdf = _patch_object_urdf(
        (INPUT_ROOT / INPUT["object_urdf"]).resolve()
    )

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
    collision_enabled=os.environ.get("ARTHOI4D_DIAG_DISABLE_ASSET_COLLISION", "") != "1",
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


def _collision_prim_paths(
    stage: Usd.Stage,
    root_path: str,
    *,
    allow_empty: bool = False,
) -> list[Sdf.Path]:
    """Return every collider below one spawned rigid body."""

    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        raise RuntimeError(f"Missing ARCTIC collision root: {root_path}")
    paths = [
        prim.GetPath()
        for prim in Usd.PrimRange(root)
        if prim.HasAPI(UsdPhysics.CollisionAPI)
    ]
    if not paths and not allow_empty:
        raise RuntimeError(f"ARCTIC collision root has no collider: {root_path}")
    return paths


def _set_collision_group_targets(
    group: UsdPhysics.CollisionGroup,
    targets: list[Sdf.Path],
) -> None:
    prim = group.GetPrim()
    includes = prim.GetRelationship("includes")
    if not includes:
        includes = prim.CreateRelationship("includes")
    includes.SetTargets(list(dict.fromkeys(targets)))


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
                _collision_prim_paths(
                    stage,
                    f"{env_path}/Robot/{body_path}",
                    allow_empty=True,
                )
            )
    _set_collision_group_targets(table_group, table_targets)
    _set_collision_group_targets(hand_group, hand_targets)
    table_group.CreateFilteredGroupsRel().SetTargets([hand_group.GetPath()])
    hand_group.CreateFilteredGroupsRel().SetTargets([table_group.GetPath()])

    def _relationship_target_count(group: UsdPhysics.CollisionGroup, name: str) -> int:
        if name == "includes":
            # ``includes`` is a regular USD relationship; the Isaac Sim
            # ``UsdPhysics.CollisionGroup`` wrapper does not expose a
            # ``GetIncludesRel`` convenience method in all supported
            # versions.  Fetch it from the underlying prim instead.
            relationship = group.GetPrim().GetRelationship("includes")
        elif name == "filteredGroups":
            relationship = group.GetFilteredGroupsRel()
        else:
            relationship = group.GetPrim().GetRelationship(name)
        return len(relationship.GetTargets()) if relationship else 0

    print(
        "[COLLISION_GROUP] "
        f"table.includes={_relationship_target_count(table_group, 'includes')} "
        f"table.filtered={_relationship_target_count(table_group, 'filteredGroups')} "
        f"hands.includes={_relationship_target_count(hand_group, 'includes')} "
        f"hands.filtered={_relationship_target_count(hand_group, 'filteredGroups')}",
        flush=True,
    )

    # Diagnostic-only collision ablations.  The default is unchanged; these
    # groups are enabled only for short A/B smoke tests via an environment
    # variable and are intentionally not part of the training configuration.
    ablation = os.environ.get("ARTHOI4D_COLLISION_ABLATION", "").strip()
    if ablation in {"object_table_off", "human_object_off", "human_table_off", "human_ground_off", "both_off"}:
        object_group_path = f"{_COLLISION_GROUP_ROOT}/diagnostic_object"
        human_group_path = f"{_COLLISION_GROUP_ROOT}/diagnostic_human"
        object_group = UsdPhysics.CollisionGroup.Define(stage, object_group_path)
        human_group = UsdPhysics.CollisionGroup.Define(stage, human_group_path)
        object_targets: list[Sdf.Path] = []
        human_targets: list[Sdf.Path] = []
        for env_path in scene.env_prim_paths:
            for body_path in ARTHOI4D_OBJECT_BODY_PRIM_PATHS.values():
                object_targets.extend(
                    _collision_prim_paths(stage, f"{env_path}/arthoi4d_object/{body_path}", allow_empty=True)
                )
            for body_path in ARTHOI4D_ROBOT_BODY_PRIM_PATHS.values():
                human_targets.extend(
                    _collision_prim_paths(stage, f"{env_path}/Robot/{body_path}", allow_empty=True)
                )
        _set_collision_group_targets(object_group, object_targets)
        _set_collision_group_targets(human_group, human_targets)
        object_filters: list[Sdf.Path] = []
        if ablation in {"object_table_off", "both_off"}:
            object_filters.append(table_group.GetPath())
            table_group.CreateFilteredGroupsRel().AddTarget(object_group.GetPath())
        if ablation in {"human_object_off", "both_off"}:
            object_filters.append(human_group.GetPath())
            human_group.CreateFilteredGroupsRel().AddTarget(object_group.GetPath())
        if ablation == "human_table_off":
            table_group.CreateFilteredGroupsRel().AddTarget(human_group.GetPath())
            human_group.CreateFilteredGroupsRel().AddTarget(table_group.GetPath())
        if ablation == "human_ground_off":
            ground_group = UsdPhysics.CollisionGroup.Define(
                stage, f"{_COLLISION_GROUP_ROOT}/diagnostic_ground"
            )
            ground_targets = _collision_prim_paths(stage, "/World/ground", allow_empty=True)
            ground_includes = ground_group.GetPrim().GetRelationship("includes")
            ground_includes.SetTargets(ground_targets)
            ground_group.CreateFilteredGroupsRel().SetTargets([human_group.GetPath()])
            human_group.CreateFilteredGroupsRel().SetTargets([ground_group.GetPath()])
        object_group.CreateFilteredGroupsRel().SetTargets(object_filters)
        print(
            "[COLLISION_GROUP] "
            f"diagnostic_object.includes={_relationship_target_count(object_group, 'includes')} "
            f"diagnostic_object.filtered={_relationship_target_count(object_group, 'filteredGroups')} "
            f"diagnostic_human.includes={_relationship_target_count(human_group, 'includes')} "
            f"diagnostic_human.filtered={_relationship_target_count(human_group, 'filteredGroups')}",
            flush=True,
        )
        print(
            f"[COLLISION_ABLATION] {ablation}: object_colliders={len(object_targets)} "
            f"human_colliders={len(human_targets)} table_colliders={len(table_targets)} "
            f"ground_colliders={len(ground_targets) if ablation == 'human_ground_off' else 0}",
            flush=True,
        )

def _human_effort_limit():
    if "ARTHOI4D_HDMI_EVAL_EFFORT_LIMITS" not in os.environ:
        return 300.0
    limits = json.loads(os.environ["ARTHOI4D_HDMI_EVAL_EFFORT_LIMITS"])
    if limits is None:
        return 300.0
    result = {}
    matched = {key: 0 for key in limits}
    for name in INPUT["human_joint_names"]:
        keys = [key for key in limits if key in name]
        if len(keys) > 1:
            raise ValueError(f"Multiple evaluation effort limits match {name}: {keys}")
        if keys:
            matched[keys[0]] += 1
            result[name] = min(300.0, float(limits[keys[0]]))
        else:
            result[name] = 300.0
    if any(value == 0 for value in matched.values()):
        raise ValueError(f"Evaluation effort limits do not match humanoid joints: {matched}")
    return result


SMPLX_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    articulation_root_prim_path=articulation_root_prim_path,
    spawn=sim_utils.UsdFileCfg(
        usd_path=human_usd,
        activate_contact_sensors=True,
        rigid_props=_rigid_props,
        collision_props=_matched_collision,
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
    actuators={
        "smplx": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit_sim=_human_effort_limit(),
            velocity_limit_sim=100.0,
            stiffness=0.0 if os.environ.get("ARTHOI4D_DIAG_DISABLE_HUMAN_PD", "") == "1" else 80.0,
            damping=0.0 if os.environ.get("ARTHOI4D_DIAG_DISABLE_HUMAN_PD", "") == "1" else 4.0,
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
        joint_pos={
            name: float(value)
            for name, value in zip(
                INPUT["object_joint_names"],
                INPUT["object_initial_joint_qpos"],
            )
        },
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
