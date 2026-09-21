import torch
# import warp
import hydra
import numpy as np
import einops
import wandb
import logging
import os
import time
import datetime
import random

from omegaconf import OmegaConf, DictConfig
from collections import OrderedDict
from collections.abc import Mapping
from tqdm import tqdm
from setproctitle import setproctitle

import active_adaptation as aa
from isaaclab.app import AppLauncher
# from active_adaptation.utils.torchrl import SyncDataCollector
from torchrl.envs.utils import set_exploration_type, ExplorationType
from tensordict.nn import TensorDictModuleBase
from tensordict import TensorDict

# local import
from scripts.helpers import make_env_policy, EpisodeStats, evaluate

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(FILE_PATH, "..", "cfg")

@hydra.main(config_path=CONFIG_PATH, config_name="train", version_base=None)
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    os.environ.setdefault("ARTHOI4D_HDMI_VALIDATE_RESET_LIMITS", "0")
    os.environ.setdefault("ARTHOI4D_HDMI_CONTACT_REGION_TELEMETRY", "0")

    print(f"is_distributed: {aa.is_distributed()}, local_rank: {aa.get_local_rank()}/{aa.get_world_size()}")
    app_cfg = OmegaConf.to_container(cfg.app, resolve=True)
    asset_root = os.environ.get("ARTHOI4D_HDMI_ASSET_ROOT")
    if asset_root:
        # IsaacLab snapshots the asset-root setting when its asset utilities
        # are imported. Pass it to Kit before AppLauncher starts, rather than
        # mutating carb settings after the snapshot may already exist.
        kit_args = str(app_cfg.get("kit_args", "")).strip()
        app_cfg["kit_args"] = (
            f"{kit_args} --/persistent/isaac/asset_root/cloud={asset_root}"
        ).strip()
        print(f"Using HDMI Isaac asset root: {asset_root}")
    app_launcher = AppLauncher(
        app_cfg,
        distributed=aa.is_distributed(),
        device=f"cuda:{aa.get_local_rank()}"
    )
    simulation_app = app_launcher.app

    if asset_root:
        import carb

        carb.settings.get_settings().set(
            "/persistent/isaac/asset_root/cloud", asset_root
        )

    run = wandb.init(
        job_type=cfg.wandb.job_type,
        project=cfg.wandb.project,
        mode=cfg.wandb.mode,
        tags=cfg.wandb.tags,
    )
    run.config.update(OmegaConf.to_container(cfg))
    
    default_run_name = f"{cfg.exp_name}-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    run_idx = run.name.split("-")[-1]
    run.name = f"{run_idx}-{default_run_name}"
    setproctitle(run.name)

    # W&B disabled mode uses a temporary run directory.  Keep all durable
    # artifacts in Hydra's explicitly requested output directory instead.
    run_dir = os.getcwd()
    os.makedirs(run_dir, exist_ok=True)
    heartbeat_path = os.environ.get("ARTHOI4D_HDMI_HEARTBEAT_PATH")

    def heartbeat(stage, iteration=-1, step=-1):
        if not heartbeat_path or not aa.is_main_process():
            return
        tmp_path = f"{heartbeat_path}.tmp.{os.getpid()}"
        with open(tmp_path, "w", encoding="utf-8") as stream:
            stream.write(
                f"timestamp={time.time():.6f} iteration={iteration} "
                f"step={step} stage={stage}\n"
            )
        os.replace(tmp_path, heartbeat_path)

    heartbeat("initializing")
    cfg_save_path = os.path.join(run_dir, "cfg.yaml")
    OmegaConf.save(cfg, cfg_save_path)
    run.save(cfg_save_path, policy="now")
    run.save(os.path.join(run_dir, "config.yaml"), policy="now")

    env, policy, vecnorm = make_env_policy(cfg)
    heartbeat("initialized")

    import inspect
    import shutil
    source_path = inspect.getfile(policy.__class__)
    target_path = os.path.join(run_dir, source_path.split("/")[-1])
    shutil.copy(source_path, target_path)
    wandb.save(target_path, policy="now")

    frames_per_batch = env.num_envs * cfg.algo.train_every
    total_frames = cfg.get("total_frames", -1) // aa.get_world_size()
    total_frames = total_frames // frames_per_batch * frames_per_batch
    total_iters = total_frames // frames_per_batch
    save_interval = cfg.get("save_interval", -1)

    configured_log_interval = cfg.get("train_stats_interval", 20)
    if configured_log_interval is None:
        configured_log_interval = 20
    log_interval = max(
        1,
        int(os.environ.get(
            "ARTHOI4D_HDMI_LOG_INTERVAL",
            configured_log_interval,
        )),
    )
    logging.info(f"Log interval: {log_interval} steps")

    def _nonfinite_state_entries(value, prefix):
        """Return tensor paths containing NaN/Inf for low-frequency checkpoints."""
        entries = []
        if isinstance(value, torch.Tensor):
            if value.numel() and not bool(torch.isfinite(value).all().item()):
                nonfinite = int((~torch.isfinite(value)).sum().item())
                entries.append((prefix, nonfinite, list(value.shape), str(value.dtype)))
        elif isinstance(value, Mapping):
            for key, child in value.items():
                child_prefix = f"{prefix}/{key}" if prefix else str(key)
                entries.extend(_nonfinite_state_entries(child, child_prefix))
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                entries.extend(_nonfinite_state_entries(child, f"{prefix}[{index}]"))
        return entries

    def _check_checkpoint_finite(state_dict, checkpoint_name):
        """Check saved model/runtime state only when a checkpoint is emitted."""
        entries = []
        for key in ("policy", "env", "vecnorm"):
            if key in state_dict:
                entries.extend(_nonfinite_state_entries(state_dict[key], key))
        if entries:
            details = "; ".join(
                f"{path}: nonfinite={count}, shape={shape}, dtype={dtype}"
                for path, count, shape, dtype in entries[:20]
            )
            logging.error(
                "Non-finite checkpoint state at %s (%d tensor paths): %s",
                checkpoint_name,
                len(entries),
                details,
            )
            raise FloatingPointError(
                f"Non-finite checkpoint state at {checkpoint_name}; see log for paths"
            )
        logging.info("Checkpoint finite check passed: %s", checkpoint_name)

    stats_keys = [
        k for k in env.reward_spec.keys(True, True) 
        if isinstance(k, tuple) and k[0] == "stats"
    ]
    episode_stats = EpisodeStats(stats_keys, device=env.device)

    checkpoint_path = cfg.get("checkpoint_path")
    resume_state = None
    if checkpoint_path:
        resume_state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        for name in ("opt_policy", "opt_critic", "opt_adapt", "opt_adapt_actor", "opt_estimator", "opt_dr_estimator"):
            if name in resume_state.get("optimizers", {}) and hasattr(policy, name):
                getattr(policy, name).load_state_dict(resume_state["optimizers"][name])
        if "rng" in resume_state:
            random.setstate(resume_state["rng"]["python"])
            np.random.set_state(resume_state["rng"]["numpy"])
            torch.set_rng_state(resume_state["rng"]["torch"])
            torch.cuda.set_rng_state_all(resume_state["rng"]["cuda"])
        else:
            logging.warning("Legacy checkpoint has no optimizer/RNG state; resuming weights and iteration only")

    def save(policy, checkpoint_name: str, artifact: bool=False):
        ckpt_path = os.path.join(run_dir, f"{checkpoint_name}.pt")
        state_dict = OrderedDict()
        state_dict["wandb"] = {"name": run.name, "id": run.id}
        state_dict["policy"] = policy.state_dict()
        state_dict["env"] = env.state_dict()
        state_dict["cfg"] = cfg
        state_dict["optimizers"] = {
            name: getattr(policy, name).state_dict()
            for name in ("opt_policy", "opt_critic", "opt_adapt", "opt_adapt_actor", "opt_estimator", "opt_dr_estimator")
            if hasattr(policy, name)
        }
        state_dict["rng"] = {
            "python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(),
        }
        if vecnorm is not None:
            state_dict["vecnorm"] = vecnorm.state_dict()
        _check_checkpoint_finite(state_dict, checkpoint_name)
        tmp_path = ckpt_path + f".tmp.{os.getpid()}"
        try:
            torch.save(state_dict, tmp_path)
            os.replace(tmp_path, ckpt_path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        if artifact:
            artifact = wandb.Artifact(
                f"{type(env).__name__}-{type(policy).__name__}",
                type="model"
            )
            artifact.add_file(ckpt_path)
            run.log_artifact(artifact)
        run.save(ckpt_path, policy="now", base_path=run_dir)
        logging.info(f"Saved checkpoint to {str(ckpt_path)}")

    assert env.training
    def should_save(i):
        if not aa.is_main_process():
            return False
        return i > 0 and save_interval > 0 and i % save_interval == 0

    # 4. --- Training Loop ---
    carry = env.reset()
    rollout_policy: TensorDictModuleBase = policy.get_rollout_policy("train")

    with torch.inference_mode():
        tmp_carry = rollout_policy(carry.clone(False))
        tmp_td, _ = env.step_and_maybe_reset(tmp_carry.clone(False))
        tmp_td["next"] = tmp_td["next"].select("done", "terminated", "discount", "reward", "stats", "is_init", "adapt_hx", strict=False)

    N = env.num_envs
    T = cfg.algo.train_every
    device = env.device

    data_buf = TensorDict({}, batch_size=[N, T], device=device)
    for key, value in tmp_td.items(include_nested=True, leaves_only=True):
        shape_tail = value.shape[1:]
        buf = torch.empty((N, T, *shape_tail), dtype=value.dtype, device=device)
        data_buf.set(key, buf)
    logging.info(f"Data buffer size: {data_buf.bytes() / 1e6:.2f} MB")

    start_iter = int(env.current_iter) + (1 if resume_state is not None else 0)
    progress = tqdm(range(start_iter, total_iters)) if aa.is_main_process() else range(start_iter, total_iters)
    env_frames = start_iter * frames_per_batch
    trace_rollout_iteration = os.environ.get("ARTHOI4D_HDMI_PPO_TRACE_ROLLOUT_ITER")
    if trace_rollout_iteration is not None:
        trace_rollout_iteration = int(trace_rollout_iteration)
    for i in progress:
        heartbeat("rollout", i, 0)
        rollout_start = time.perf_counter()
        with torch.inference_mode(), set_exploration_type(ExplorationType.RANDOM):
            torch.compiler.cudagraph_mark_step_begin() # for compiled policy
            env.set_progress(i)
            for step in range(cfg.algo.train_every):
                policy.set_trace_rollout_context(i, step)
                trace_policy_io = step == 0 or i == trace_rollout_iteration
                if trace_policy_io:
                    policy.trace_policy_io("pre", carry)
                carry = rollout_policy(carry)
                if trace_policy_io:
                    policy.trace_policy_io("post", carry)
                # Opt-in no-op unless ARTHOI4D_HDMI_FINITE_TRACE_DIR is set.
                env.base_env.trace_policy_action(carry["action"])
                td, carry = env.step_and_maybe_reset(carry)
                td["next"] = td["next"].select("done", "terminated", "discount", "reward", "stats", "is_init", "adapt_hx", strict=False)
                data_buf[:, step] = td
            policy.critic(data_buf)
            values = data_buf["state_value"]
            data_buf["next", "state_value"] = torch.where(
                data_buf["next", "done"],
                values, # a walkaround to avoid storing the next states
                torch.cat([values[:, 1:], policy.critic(carry.copy())["state_value"].unsqueeze(1)], dim=1)
            )
        rollout_time = time.perf_counter() - rollout_start

        episode_stats.add(data_buf)
        env_frames += data_buf.numel()

        should_log = i % log_interval == 0 or i == total_iters - 1
        if hasattr(policy, "log_detailed_metrics"):
            policy.log_detailed_metrics = should_log
        info = {}
        if should_log and len(episode_stats):
            for k, v in sorted(episode_stats.pop().items(True, True)):
                key = "train/" + ("/".join(k) if isinstance(k, tuple) else k)
                info[key] = torch.mean(v.float()).item()
        training_start = time.perf_counter()
        heartbeat("train_op", i)
        policy.trace_rollout("pre_train", data_buf, vecnorm)
        info.update(policy.train_op(data_buf))
        policy.trace_rollout("post_train", data_buf, vecnorm)
        training_time = time.perf_counter() - training_start
        if should_log:
            info.update(env.extra)
            info.update(env.stats_ema)

        if hasattr(policy, "step_schedule"):
            policy.step_schedule(i / total_iters)

        info["env_frames"] = env_frames
        info["rollout_fps"] = data_buf.numel() / rollout_time
        info["training_time"] = training_time

        if should_save(i) and not os.path.exists(os.path.join(run_dir, f"checkpoint_{i}.pt")):
            heartbeat("checkpoint", i)
            save(policy, f"checkpoint_{i}")

        if aa.is_main_process() and should_log:
            # print(OmegaConf.to_yaml({k: v for k, v in info.items() if (isinstance(v, (float, int)) and not k.startswith("performance_reward"))}))
            run.log(info)

    # 5. --- Finalization and Cleanup ---
    if aa.is_main_process():
        save(policy, "checkpoint_final", artifact=True)

    policy_eval = policy.get_rollout_policy("eval")
    info, trajs, stats, policy_trajs = evaluate(env, policy_eval, render=cfg.eval_render, seed=cfg.seed)
    run.log(info)

    wandb.finish()
    os._exit(0)
    env.close()
    simulation_app.close()

    run_id = run.id
    project = run.project
    entity = run.entity
    run_path = f"{entity}/{project}/{run_id}"
    
    return run_path



if __name__ == "__main__":
    main()
