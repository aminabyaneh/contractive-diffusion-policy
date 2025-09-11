import os
import pathlib
import time
import contextlib

from copy import deepcopy

# suppress d4rl import warnings
with open(os.devnull, 'w') as devnull, \
    contextlib.redirect_stdout(devnull), \
    contextlib.redirect_stderr(devnull):
    import d4rl

import gym
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from cleandiffuser.dataset.d4rl_kitchen_dataset import D4RLKitchenTDDataset
from cleandiffuser.dataset.d4rl_antmaze_dataset import D4RLAntmazeTDDataset
from cleandiffuser.dataset.d4rl_mujoco_dataset import D4RLMuJoCoTDDataset

from cleandiffuser.dataset.dataset_utils import loop_dataloader
from cleandiffuser.diffusion import DiscreteDiffusionSDE, ContinuousDiffusionSDE
from cleandiffuser.nn_condition import IdentityCondition
from cleandiffuser.nn_diffusion import DQLMlp
from cleandiffuser.utils import report_parameters, DQLCritic, FreezeModules, at_least_ndim

from src.contraction_loss import compute_contractive_loss
from src.sim_eval import eval
from src.utils import Logger, set_seed

# NEW: Optuna
import optuna


def run(args, trial: optuna.Trial):
    """
    Main pipeline for training and evaluating contractive EDP on D4RL datasets.
    Returns the best evaluation reward encountered (used by Optuna).
    """
    # ---------------------- Configurations ----------------------
    set_seed(args.seed)

    # ---------------------- Create Dataset/Env ------------------
    env = gym.make(args.task.env_name)

    try:
        # dataset handler
        if args.env_name == "antmaze":
            dataset = D4RLAntmazeTDDataset(d4rl.qlearning_dataset(env))
        elif args.env_name == "kitchen":
            dataset = D4RLKitchenTDDataset(d4rl.qlearning_dataset(env))
        elif args.env_name == "mujoco":
            dataset = D4RLMuJoCoTDDataset(d4rl.qlearning_dataset(env), args.normalize_reward)
        else:
            raise ValueError(f"Unknown environment: {args.env_name}")

        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                num_workers=4, pin_memory=True, drop_last=True,
                                persistent_workers=True)

        obs_dim, act_dim = dataset.o_dim, dataset.a_dim
        # --------------- Network Architecture -----------------
        nn_diffusion = DQLMlp(obs_dim, act_dim, emb_dim=64, timestep_emb_type="positional").to(args.device)
        nn_condition = IdentityCondition(dropout=0.0).to(args.device)

        # ----------------- Diffusion Actor -------------------
        actor = DiscreteDiffusionSDE(
            nn_diffusion, nn_condition, predict_noise=args.predict_noise,
            optim_params={"lr": args.actor_learning_rate},
            x_max=+1. * torch.ones((1, act_dim), device=args.device),
            x_min=-1. * torch.ones((1, act_dim), device=args.device),
            diffusion_steps=args.diffusion_steps,
            device=args.device)

        # ---------------------- Critic ------------------------
        critic = DQLCritic(obs_dim, act_dim, hidden_dim=args.hidden_dim).to(args.device)

        # target copy for stability
        critic_target = deepcopy(critic).requires_grad_(False).eval()
        critic_optim = torch.optim.Adam(critic.parameters(), lr=args.critic_learning_rate)

        actor_lr_scheduler = CosineAnnealingLR(actor.optimizer, T_max=args.gradient_steps)
        critic_lr_scheduler = CosineAnnealingLR(critic_optim, T_max=args.gradient_steps)

        actor.train()
        critic.train()

        n_gradient_step = 0
        best_reward = -float("inf")

        prior = torch.zeros((args.batch_size, act_dim), device=args.device)

        for batch in loop_dataloader(dataloader):
            # load batch of data
            obs, next_obs = batch["obs"]["state"].to(args.device), batch["next_obs"]["state"].to(args.device)
            act = batch["act"].to(args.device)
            rew = batch["rew"].to(args.device)
            tml = batch["tml"].to(args.device)

            # ---------------------- Critic Training ----------------------
            current_q1, current_q2 = critic(obs, act)

            # sample next action
            if args.env_name == "antmaze":
                next_obs_rpt = next_obs.unsqueeze(1).repeat(1, 10, 1).view(-1, obs_dim)

                next_act, _ = actor.sample(prior.repeat(10, 1), solver=args.solver, n_samples=args.batch_size * 10,
                                            sample_steps=args.sampling_steps, use_ema=True, temperature=1.0, condition_cfg=next_obs_rpt, w_cfg=1.0, requires_grad=False)

                target_q1, target_q2 = critic_target(next_obs_rpt, next_act)
                target_q1 = target_q1.view(-1, 10, 1).max(1)[0]
                target_q2 = target_q2.view(-1, 10, 1).max(1)[0]
                target_q = torch.min(target_q1, target_q2)
                target_q = (rew + (1 - tml) * args.discount * target_q).detach()

            elif args.env_name == "kitchen" or args.env_name == "mujoco":
                next_act, _ = actor.sample(prior, solver=args.solver, n_samples=args.batch_size,
                                            sample_steps=args.sampling_steps, use_ema=True, temperature=1.0,
                                            condition_cfg=next_obs, w_cfg=1.0, requires_grad=False)

                with torch.no_grad():
                    target_q = torch.min(*critic_target(next_obs, next_act))

                    # TD‐target: reward + γ*(1−done)*min_Q′
                    target_q = (rew + (1 - tml) * args.discount * target_q).detach()

            # form critic loss
            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

            # update critic
            critic_optim.zero_grad()
            critic_loss.backward()
            critic_optim.step()

            # ---------------------- Actor Training ----------------------
            bc_loss = actor.loss(act, obs)

            # EDP action approximation
            t = torch.randint(args.diffusion_steps, (args.batch_size,), device=args.device)
            eps = torch.randn_like(act)

            alpha, sigma = at_least_ndim(actor.alpha[t], act.dim()), at_least_ndim(actor.sigma[t], act.dim())
            noisy_act = alpha * act + sigma * eps

            condition_vec_cfg = actor.model["condition"](obs, None)
            pred_act = actor.model["diffusion"](noisy_act, t, condition_vec_cfg)

            with FreezeModules([critic, ]):
                q1_new_action, q2_new_action = critic(obs, pred_act)
            if np.random.uniform() > 0.5:
                q_loss = - q1_new_action.mean() / q2_new_action.abs().mean().detach()
            else:
                q_loss = - q2_new_action.mean() / q1_new_action.abs().mean().detach()

            # ---------------------- Contraction Loss ----------------------
            t_threshold = int(args.diffusion_steps * args.contraction_threshold)  # e.g., 10% of total steps
            low_t_mask = t < t_threshold

            if low_t_mask.any():
                loss_dict = compute_contractive_loss(
                    model=actor.model["diffusion"],
                    xt=noisy_act[low_t_mask],
                    t=t[low_t_mask],
                    condition=condition_vec_cfg[low_t_mask],
                    lambda_contr=args.lambda_contr,
                    loss_type=args.loss_type,
                    num_power_iters=args.num_power_iters
                )

                contraction_loss = (
                    float(args.loss_weights.jacobian) * loss_dict["jacobian_loss"] +
                    float(args.loss_weights.eigen_max) * loss_dict["eigen_max"] +
                    float(args.loss_weights.eigen_avg) * loss_dict["eigen_avg"]
                )
            else:
                contraction_loss = torch.tensor(0.0, device=act.device)

            # compute the total actor loss
            actor_loss = bc_loss + args.task.eta * q_loss + contraction_loss

            actor.optimizer.zero_grad()
            actor_loss.backward()
            actor.optimizer.step()

            actor_lr_scheduler.step()
            critic_lr_scheduler.step()

            # EMA update
            if n_gradient_step % args.ema_update_interval == 0:
                if n_gradient_step >= 1000:
                    actor.ema_update()
                # Polyak‐average update for the ensemble critic targets
                for param, target_param in zip(critic.parameters(), critic_target.parameters()):
                    target_param.data.copy_(0.995 * param.data + (1 - 0.995) * target_param.data)

            # evaluation
            if (n_gradient_step + 1) % args.eval_interval == 0:
                eval_log = eval(env, actor, critic, critic_target, dataset, args, obs_dim, act_dim)
                reward = eval_log["mean_rew"]  # the signal for hyperparameter optimization
                best_reward = max(best_reward, float(reward))

                # Report to Optuna & allow pruning
                if trial is not None:
                    trial.report(best_reward, step=n_gradient_step + 1)
                    if trial.should_prune():
                        raise optuna.TrialPruned()

            n_gradient_step += 1
            if n_gradient_step >= args.gradient_steps:
                break

        return best_reward
    finally:
        try:
            env.close()
        except Exception:
            pass


@hydra.main(config_path="configs/edp", config_name="optuna", version_base=None)
def main(args):
    """
    Launch Optuna HPO for:
      - args.loss_weights.jacobian
      - args.lambda_contr
    """
    base_args = args  # keep hydra-managed args as the base

    # --- HPO knobs (safe defaults if not in your YAML) ---
    # You can also put these under `optuna:` in Hydra config and they’ll be picked up.
    n_trials = 100
    sampler_seed = 0
    storage = None  # e.g., "sqlite:///study.db"
    study_name = "cd_edp_hpo"
    direction = "maximize"

    # Pruner warms up a few evals before pruning. If evals are sparse, keep it small.
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=3)
    sampler = optuna.samplers.TPESampler(seed=sampler_seed, n_startup_trials=min(5, max(2, n_trials // 5)))

    if storage is not None:
        study = optuna.create_study(direction=direction, sampler=sampler, pruner=pruner,
                                    study_name=study_name, storage=storage, load_if_exists=True)
    else:
        study = optuna.create_study(direction=direction, sampler=sampler, pruner=pruner, study_name=study_name)

    def optuna_objective(trial: optuna.Trial):
        # Clone Hydra args so each trial is isolated
        local_args = deepcopy(base_args)

        # Suggest hyperparameters
        # Feel free to tighten ranges later if you have priors.
        local_args.loss_weights.jacobian = trial.suggest_float(
            "loss_weights.jacobian", 1e-4, 1e10, log=True
        )
        local_args.lambda_contr = trial.suggest_float(
            "lambda_contr", 1e-4, 10.0, log=True
        )

        # Make trials reproducible but different: offset seed per trial
        if hasattr(local_args, "seed"):
            try:
                local_args.seed = int(local_args.seed) + int(trial.number)
            except Exception:
                pass

        # Run training/eval loop and return the metric to maximize
        return run(local_args, trial=trial)

    try:
        study.optimize(optuna_objective, n_trials=n_trials, gc_after_trial=True, n_jobs=10)
    except KeyboardInterrupt:
        pass  # allow graceful stop

    # Print best result; Hydra logs will capture this too.
    best = study.best_trial
    print("\n=== Optuna Best Trial ===")
    print(f"Value (best mean_rew): {best.value}")
    print("Params:")
    for k, v in best.params.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    # use optuna to optimize hyperparameters
    main()
