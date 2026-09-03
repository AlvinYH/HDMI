import active_adaptation
from collections.abc import Callable


if active_adaptation.get_backend() == "isaac":
    from isaaclab.sim import SimulationContext, SimulationCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    import builtins

    def create_isaaclab_sim_and_scene(
        sim_cfg: SimulationCfg,
        scene_cfg: InteractiveSceneCfg,
        before_first_step: Callable[[InteractiveScene], None] | None = None,
    ):
        # create a simulation context to control the simulator
        if SimulationContext.instance() is None:
            sim = SimulationContext(sim_cfg)
        else:
            raise RuntimeError("Simulation context already exists. Cannot create a new one.")
        scene = InteractiveScene(scene_cfg)
        # USD cloning does not install PhysX collision groups.  Keep every
        # cloned environment isolated while preserving collisions with the
        # shared ground plane.
        if not scene_cfg.replicate_physics and scene_cfg.filter_collisions:
            scene.filter_collisions(global_prim_paths=["/World/ground"])
        if before_first_step is not None:
            before_first_step(scene)
        if builtins.ISAAC_LAUNCHED_FROM_TERMINAL is False:
            sim.reset()
        sim.step(render=sim.has_gui())
        return sim, scene

elif active_adaptation.get_backend() == "mujoco":
    pass
else:
    raise NotImplementedError
