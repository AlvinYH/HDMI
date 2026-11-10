def hand_region_bodies(body_paths, eef_names):
    """Select each distinct physical wrist/finger body once per hand."""
    if not body_paths:
        return [[name] for name in eef_names]
    if len(eef_names) != 2:
        raise ValueError("ArtHOI4D contact telemetry requires two hands")
    groups = []
    for prefix in ("L_", "R_"):
        selected = {}
        for name, path in body_paths.items():
            if name.startswith(prefix) and any(
                token in name for token in ("Wrist", "Hand", "Index", "Middle", "Pinky", "Ring", "Thumb")
            ):
                selected.setdefault(path, name)
        if not selected:
            raise ValueError(f"No physical hand bodies found for {prefix}")
        groups.append(selected)
    if set(groups[0]).intersection(groups[1]):
        raise ValueError("A physical body cannot belong to both hands")
    return [list(group.values()) for group in groups]


def reduce_net_contact_forces(hand_forces, region_forces):
    """Maximum body net-force magnitude per hand and net force per object link."""
    import torch

    hand_force = torch.stack([
        torch.linalg.vector_norm(hand, dim=-1).amax(dim=1)
        for hand in hand_forces
    ], dim=1)
    return hand_force, torch.linalg.vector_norm(region_forces, dim=-1)
