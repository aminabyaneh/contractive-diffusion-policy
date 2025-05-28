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
import torch.nn.functional as F

from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from cleandiffuser.dataset.d4rl_kitchen_dataset import D4RLKitchenTDDataset
from cleandiffuser.dataset.dataset_utils import loop_dataloader
from cleandiffuser.diffusion import DiscreteDiffusionSDE
from cleandiffuser.nn_condition import IdentityCondition
from cleandiffuser.nn_diffusion import DQLMlp
from cleandiffuser.utils import report_parameters, DQLCritic, FreezeModules, at_least_ndim

from src.contraction_loss import compute_contractive_loss
from src.sim_eval import eval
from src.utils import Logger, set_seed


@hydra.main(config_path="configs/cd_edp/kitchen", config_name="kitchen", version_base=None)
def pipeline(args):
    """
    Main pipeline for training and evaluating contractive diffuser on D4RL Kitchen dataset.

    Code originally adapted from cleandiffuser projects.
    Args:
        args: Configurations for the pipeline.

    Raises:
        ValueError: Unknown config or training mode.
    """
    # ---------------------- Configurations ----------------------
    set_seed(args.seed) # set random seed

    save_path = f'{args.log_dir}/{args.pipeline_name}/{args.task.env_name}/'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    # ---------------------- Wandb Init ----------------------
    logger = Logger(pathlib.Path(save_path), args)

    # ---------------------- Create Dataset ----------------------
    env = gym.make(args.task.env_name)
    dataset = D4RLKitchenTDDataset(d4rl.qlearning_dataset(env))
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    obs_dim, act_dim = dataset.o_dim, dataset.a_dim

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

    # --------------- Diffusion Model Actor --------------------
    actor = DiscreteDiffusionSDE(
        nn_diffusion, nn_condition, predict_noise=False,
        optim_params={"lr": args.actor_learning_rate},
        x_max=+1. * torch.ones((1, act_dim), device=args.device),
        x_min=-1. * torch.ones((1, act_dim), device=args.device),
        diffusion_steps=args.diffusion_steps, ema_rate=args.ema_rate,
        device=args.device)

    # ------------------ Critic ---------------------
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
            # get batch data
            obs, next_obs = batch["obs"]["state"].to(args.device), batch["next_obs"]["state"].to(args.device)
            act = batch["act"].to(args.device)
            rew = batch["rew"].to(args.device)
            tml = batch["tml"].to(args.device)

            # critic training
            current_q1, current_q2 = critic(obs, act)

            # sample next action
            next_act, _ = actor.sample(
                prior, solver=args.solver,
                n_samples=args.batch_size, sample_steps=args.sampling_steps, use_ema=True,
                temperature=1.0, condition_cfg=next_obs, w_cfg=1.0, requires_grad=False)

            with torch.no_grad():
                # get min over next‐Q’s
                target_q = torch.min(*critic_target(next_obs, next_act))

                # TD‐target: reward + γ*(1−done)*min_Q′
                target_q = (rew + (1 - tml) * args.discount * target_q).detach()

            # form critic loss
            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

            # update critic
            critic_optim.zero_grad()
            critic_loss.backward()
            critic_optim.step()

            # policy training
            bc_loss = actor.loss(act, obs)

            # edp action approximation
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


            # contractivity loss
            loss_dict = compute_contractive_loss(model=actor.model["diffusion"],
                                                 xt=noisy_act, t=t,
                                                 condition=condition_vec_cfg,
                                                 lambda_contr=args.lambda_contr,
                                                 loss_type=args.loss_type,
                                                 num_power_iters=args.num_power_iters)

            contraction_loss = float(args.loss_weights.jacobian) * loss_dict["jacobian_loss"] + \
                               float(args.loss_weights.eigen_max) * loss_dict["eigen_max"] + \
                               float(args.loss_weights.eigen_avg) * loss_dict["eigen_avg"]

            # compute the total actor loss
            actor_loss = bc_loss + args.task.eta * q_loss + contraction_loss

            actor.optimizer.zero_grad()
            actor_loss.backward()
            actor.optimizer.step()

            actor_lr_scheduler.step()
            critic_lr_scheduler.step()

            # ema
            if n_gradient_step % args.ema_update_interval == 0:
                if n_gradient_step >= 1000:
                    actor.ema_update()
                # polyak‐average update for the ensemble critic targets
                for param, target_param in zip(critic.parameters(), critic_target.parameters()):
                    target_param.data.copy_(0.995 * param.data + (1 - 0.995) * target_param.data)

            # logging
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
                eval_log = eval(env, actor, critic, critic_target, dataset, args, obs_dim, act_dim)
                eval_time = (time.time() - eval_time)
                eval_log["step"] = n_gradient_step + 1
                logger.log(eval_log, category='eval')

            # saving
            if (n_gradient_step + 1) % args.save_interval == 0:
                # save diffusion actor checkpoints
                save_path = f"{save_path}/models/{args.exp_name}/"
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
    # direct evaluation
    elif args.mode == "eval":
        save_path = f"{save_path}/models/{args.exp_name}/"

        # load actor as before
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
