import sys
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
import torch
import numpy as np
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

# add source to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from source.contraction_loss import compute_contractive_loss
from source.sim_eval import eval
from source.utils import Logger, set_seed


@hydra.main(config_path="../configs/cdp-rl", config_name="main", version_base=None)
def pipeline(args):
    """
    Main pipeline for training and evaluating contractive EDP on D4RL datasets.
    This function initializes the environment, dataset, models, and starts the training or evaluation process.

    Args:
        args: Configurations for the pipeline. See confgs/cd_edp/main.yaml for details.

    Raises:
        ValueError: Unknown config or training mode.
    """

    # ---------------------- Configurations ----------------------
    set_seed(args.seed)

    save_path = f'{args.log_dir}/{args.pipeline_name}/{args.task.env_name}/'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    # ---------------------- Wandb Init ----------------------
    logger = Logger(pathlib.Path(save_path), args)

    # ---------------------- Create Dataset ----------------------
    env = gym.make(args.task.env_name)

    # dataset handler
    if args.env_name == "antmaze":
        full_dataset = D4RLAntmazeTDDataset(d4rl.qlearning_dataset(env))
    elif args.env_name == "kitchen":
        full_dataset = D4RLKitchenTDDataset(d4rl.qlearning_dataset(env))
    elif args.env_name == "mujoco":
        full_dataset = D4RLMuJoCoTDDataset(d4rl.qlearning_dataset(env), args.normalize_reward)
    else:
        raise ValueError(f"Unknown environment: {args.env_name}")

    # use only a portion of the dataset
    data_portion = getattr(args, 'data_portion', 1.0)  # Default to full dataset if not specified
    if data_portion < 1.0:
        total_size = len(full_dataset)
        subset_size = int(total_size * data_portion)
        indices = torch.randperm(total_size)[:subset_size]
        dataset = torch.utils.data.Subset(full_dataset, indices)
    else:
        dataset = full_dataset

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=4, pin_memory=True, drop_last=True,
                            persistent_workers=True)

    obs_dim, act_dim = full_dataset.o_dim, full_dataset.a_dim

    print(f"\n================================ Dataset report ==================================")
    print(f"env_name: {args.task.env_name}")
    print(f"num_episodes: {len(dataset)}")
    print(f"batch_size: {args.batch_size}")
    print(f"obs_dim: {obs_dim}, act_dim: {act_dim}")
    print(f"===================================================================================\n")

    # --------------- Network Architecture -----------------
    nn_diffusion = DQLMlp(obs_dim, act_dim, emb_dim=64, timestep_emb_type="positional").to(args.device)
    nn_condition = IdentityCondition(dropout=0.0).to(args.device)

    print(f"================================ Diffusion Model ==================================")
    report_parameters(nn_diffusion)
    print(f"===================================================================================\n")

    # ----------------- Diffusion Actor -------------------
    if args.diffusion_type == "continuous":
        actor = ContinuousDiffusionSDE(
            nn_diffusion, nn_condition, predict_noise=args.predict_noise,
            optim_params={"lr": args.actor_learning_rate},
            x_max=+1. * torch.ones((1, act_dim), device=args.device),
            x_min=-1. * torch.ones((1, act_dim), device=args.device),
            device=args.device)

    elif args.diffusion_type == "discrete":
        actor = DiscreteDiffusionSDE(
            nn_diffusion, nn_condition, predict_noise=args.predict_noise,
            optim_params={"lr": args.actor_learning_rate},
            x_max=+1. * torch.ones((1, act_dim), device=args.device),
            x_min=-1. * torch.ones((1, act_dim), device=args.device),
            diffusion_steps=args.diffusion_steps,
            device=args.device)
    else:
        raise ValueError(f"Unknown diffusion type: {args.diffusion_type}")

    # ---------------------- Critic ------------------------
    critic = DQLCritic(obs_dim, act_dim, hidden_dim=args.hidden_dim).to(args.device)

    # target copy for stability
    critic_target = deepcopy(critic).requires_grad_(False).eval()
    critic_optim = torch.optim.Adam(critic.parameters(), lr=args.critic_learning_rate)

    # ---------------------- Training ----------------------
    if args.mode == "train":
        start_time = time.time()

        actor_lr_scheduler = CosineAnnealingLR(actor.optimizer, T_max=args.gradient_steps)
        critic_lr_scheduler = CosineAnnealingLR(critic_optim, T_max=args.gradient_steps)

        actor.train()
        critic.train()

        n_gradient_step = 0
        log = {"step": 0, "time": 0, "bc_loss": 0., "q_loss": 0., "critic_loss": 0., "target_q_mean": 0.,
               "target_q_std": 0., "eigen_max": 0., "eigen_avg": 0., "eigen_std": 0.,
               "jacobian_loss": 0., "jacobian_norm": 0.}

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

            # ---------------------- Logging ----------------------
            log["bc_loss"] += bc_loss.item()
            log["q_loss"] += q_loss.item()
            log["critic_loss"] += critic_loss.item()
            log["target_q_mean"] += target_q.mean().item()
            log["target_q_std"] += target_q.std().item()

            log["eigen_max"] += loss_dict["eigen_max"].item()
            log["eigen_avg"] += loss_dict["eigen_avg"].item()
            log["eigen_std"] += loss_dict["eigen_std"].item()
            log["jacobian_loss"] += loss_dict["jacobian_loss"].item()
            log["jacobian_norm"] += loss_dict["jacobian_norm"].item()

            if (n_gradient_step + 1) % args.log_interval == 0:
                log["bc_loss"] /= args.log_interval
                log["q_loss"] /= args.log_interval
                log["critic_loss"] /= args.log_interval
                log["target_q_mean"] /= args.log_interval
                log["time"] = (time.time() - start_time) / 60
                log["step"] = n_gradient_step + 1
                log["eigen_max"] /= args.log_interval
                log["eigen_avg"] /= args.log_interval
                log["eigen_std"] /= args.log_interval
                log["jacobian_loss"] /= args.log_interval
                log["jacobian_norm"] /= args.log_interval
                log["target_q_std"] /= args.log_interval

                # log to console and wandb
                logger.log(log, category='train')
                log = {"step": 0, "time": 0, "bc_loss": 0., "q_loss": 0., "critic_loss": 0.,
                       "target_q_mean": 0., "target_q_std": 0., "eigen_max": 0., "eigen_avg": 0.,
                       "eigen_std": 0., "jacobian_loss": 0., "jacobian_norm": 0.}

            # evaluation
            eval_time = time.time()
            if (n_gradient_step + 1) % args.eval_interval == 0:
                eval_log = eval(env, actor, critic, critic_target, full_dataset, args, obs_dim, act_dim)
                eval_time = (time.time() - eval_time)
                eval_log["step"] = n_gradient_step + 1
                logger.log(eval_log, category='eval')

            # saving
            if (n_gradient_step + 1) % args.save_interval == 0:
                # save diffusion actor checkpoints
                save_path = f"{save_path}/models/{args.exp_name}/"
                os.makedirs(save_path, exist_ok=True)

                actor.save(save_path + f"diffusion_ckpt_{n_gradient_step + 1}.pt")
                actor.save(save_path + f"diffusion_ckpt_latest.pt")
                torch.save({
                    "critic": critic.state_dict(),
                    "critic_target": critic_target.state_dict(),
                }, save_path + f"critic_ckpt_{n_gradient_step + 1}.pt")
                torch.save({
                    "critic": critic.state_dict(),
                    "critic_target": critic_target.state_dict(),
                }, save_path + f"critic_ckpt_latest.pt")

            n_gradient_step += 1
            if n_gradient_step >= args.gradient_steps:
                break

    # ---------------------- Evaluation ----------------------
    elif args.mode == "eval":
        save_path = f"{save_path}/models/{args.exp_name}/"

        # load actor
        actor.load(save_path + f"diffusion_ckpt_{args.ckpt}.pt")

        # load critic and target critic
        critic_ckpt = torch.load(save_path + f"critic_ckpt_{args.ckpt}.pt")
        critic.load_state_dict(critic_ckpt["critic"])
        critic_target.load_state_dict(critic_ckpt["critic_target"])

        eval_log = eval(env, actor, critic, critic_target, dataset, args, obs_dim, act_dim)
        print(f"Eval mean rewards: {eval_log['mean_rewards']:.4f} | "
              f"Eval std rewards: {eval_log['std_rewards']:.4f} |")

    else:
        raise ValueError(f"Invalid mode: {args.mode}")


if __name__ == "__main__":
    pipeline()
