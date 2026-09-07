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
        # Original HDMI uses USD cloning (replicate_physics=False), for which
        # IsaacLab requires an explicit per-environment PhysX collision filter.
        # Keep the shared ground enabled after isolating env_i from env_j.
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
