"""
Implicit Diffusion Q-Learning (IDQL) for D4RL datasets.

IDQL learns a conditional diffusion model as the policy and uses
an expectile regression value function to weigh the sampled actions.
"""
import contextlib
import os
import pathlib
import time

from copy import deepcopy

import gym
import hydra
import numpy as np

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

# suppress d4rl import warnings
with open(os.devnull, 'w') as devnull, \
    contextlib.redirect_stdout(devnull), \
    contextlib.redirect_stderr(devnull):
    import d4rl

from cleandiffuser.dataset.d4rl_antmaze_dataset import D4RLAntmazeTDDataset
from cleandiffuser.dataset.d4rl_kitchen_dataset import D4RLKitchenTDDataset
from cleandiffuser.dataset.d4rl_mujoco_dataset import D4RLMuJoCoTDDataset
from cleandiffuser.dataset.dataset_utils import loop_dataloader
from cleandiffuser.diffusion import ContinuousDiffusionSDE, DiscreteDiffusionSDE
from cleandiffuser.nn_condition import IdentityCondition
from cleandiffuser.nn_diffusion import IDQLMlp
from cleandiffuser.utils import DQLCritic, FreezeModules, report_parameters, IDQLQNet, IDQLVNet

from src.utils import Logger, set_seed


@hydra.main(config_path="../configs/idql", config_name="main", version_base=None)
def pipeline(args):
    """
    Main pipeline for training and evaluating contractive IDQL on D4RL datasets.
    This function initializes the environment, dataset, models, and starts the training or evaluation process.

    Args:
        args: Configurations for the pipeline. See configs/idql/main.yaml for details.

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

    print(f"\n================================ Dataset report ==================================")
    print(f"env_name: {args.task.env_name}")
    print(f"num_episodes: {len(dataset)}")
    print(f"batch_size: {args.batch_size}")
    print(f"obs_dim: {obs_dim}, act_dim: {act_dim}")
    print(f"===================================================================================\n")

    # --------------- Network Architecture -----------------
    nn_diffusion = IDQLMlp(obs_dim, act_dim, emb_dim=64, hidden_dim=args.actor_hidden_dim,
                           n_blocks=args.actor_n_blocks, dropout=args.actor_dropout, timestep_emb_type="positional").to(args.device)
    nn_condition = IdentityCondition(dropout=0.0).to(args.device)

    print(f"================================ Diffusion Model ==================================")
    report_parameters(nn_diffusion)
    print(f"===================================================================================\n")

    # --------------- Diffusion Actor --------------------
    actor = DiscreteDiffusionSDE(
        nn_diffusion, nn_condition, predict_noise=args.predict_noise, optim_params={"lr": args.actor_learning_rate},
        x_max=+1. * torch.ones((1, act_dim), device=args.device),
        x_min=-1. * torch.ones((1, act_dim), device=args.device),
        diffusion_steps=args.diffusion_steps, ema_rate=args.ema_rate)

    # ------------------ Critic ---------------------
    iql_q = IDQLQNet(obs_dim, act_dim, hidden_dim=args.critic_hidden_dim).to(args.device)
    iql_q_target = deepcopy(iql_q).requires_grad_(False).eval()
    iql_v = IDQLVNet(obs_dim, hidden_dim=args.critic_hidden_dim).to(args.device)

    q_optim = torch.optim.Adam(iql_q.parameters(), lr=args.critic_learning_rate)
    v_optim = torch.optim.Adam(iql_v.parameters(), lr=args.critic_learning_rate)

    # ---------------------- Training ----------------------
    if args.mode == "train":
        start_time = time.time()

        actor_lr_scheduler = CosineAnnealingLR(actor.optimizer, T_max=args.gradient_steps)
        q_lr_scheduler = CosineAnnealingLR(q_optim, T_max=args.gradient_steps)
        v_lr_scheduler = CosineAnnealingLR(v_optim, T_max=args.gradient_steps)

        actor.train()
        iql_q.train()
        iql_v.train()

        n_gradient_step = 0
        log = {"step": 0, "time": 0, "bc_loss": 0., "q_loss": 0., "v_loss": 0., "target_q_mean": 0.,
               "target_q_std": 0., "eigen_max": 0., "eigen_avg": 0., "eigen_std": 0.,
               "jacobian_loss": 0., "jacobian_norm": 0.}

        prior = torch.zeros((args.batch_size, act_dim), device=args.device)

        for batch in loop_dataloader(dataloader):
            # load batch of data
            obs, next_obs = batch["obs"]["state"].to(args.device), batch["next_obs"]["state"].to(args.device)
            act = batch["act"].to(args.device)
            rew = batch["rew"].to(args.device)
            tml = batch["tml"].to(args.device)

            # -- IQL Training
            if n_gradient_step % 2 == 0:

                q = iql_q_target(obs, act)
                v = iql_v(obs)
                v_loss = (torch.abs(args.iql_tau - ((q - v) < 0).float()) * (q - v) ** 2).mean()

                v_optim.zero_grad()
                v_loss.backward()
                v_optim.step()

                with torch.no_grad():
                    td_target = rew + args.discount * (1 - tml) * iql_v(next_obs)
                q1, q2 = iql_q.both(obs, act)
                q_loss = ((q1 - td_target) ** 2 + (q2 - td_target) ** 2).mean()
                q_optim.zero_grad()
                q_loss.backward()
                q_optim.step()

                q_lr_scheduler.step()
                v_lr_scheduler.step()

                for param, target_param in zip(iql_q.parameters(), iql_q_target.parameters()):
                    target_param.data.copy_(0.995 * param.data + (1 - 0.995) * target_param.data)

            # -- Policy Training
            bc_loss = actor.update(act, obs)["loss"]
            actor_lr_scheduler.step()

            # # ----------- Logging ------------
            log["bc_loss"] += bc_loss.item()
            log["q_loss"] += q_loss.item()
            log["v_loss"] += v_loss.item()

            if (n_gradient_step + 1) % args.log_interval == 0:
                log["bc_loss"] /= args.log_interval
                log["q_loss"] /= args.log_interval
                log["v_loss"] /= args.log_interval
                # log to console and wandb
                logger.log(log, category='train')
                log = {"step": 0, "time": 0, "bc_loss": 0., "q_loss": 0., "v_loss": 0.,
                       "target_q_mean": 0., "target_q_std": 0., "eigen_max": 0., "eigen_avg": 0.,
                       "eigen_std": 0., "jacobian_loss": 0., "jacobian_norm": 0.}

            # evaluation
            eval_time = time.time()
            if (n_gradient_step + 1) % args.eval_interval == 0:
                eval_log = eval(env, actor, iql_q, iql_v, iql_q_target, dataset, args, obs_dim, act_dim) # double check
                eval_time = (time.time() - eval_time)
                eval_log["step"] = n_gradient_step + 1
                logger.log(eval_log, category='eval')

            # saving
            if (n_gradient_step + 1) % args.save_interval == 0:
                actor.save(save_path + f"diffusion_ckpt_{n_gradient_step + 1}.pt")
                actor.save(save_path + f"diffusion_ckpt_latest.pt")
                torch.save({
                    "iql_q": iql_q.state_dict(),
                    "iql_q_target": iql_q_target.state_dict(),
                    "iql_v": iql_v.state_dict(),
                }, save_path + f"iql_ckpt_{n_gradient_step + 1}.pt")
                torch.save({
                    "iql_q": iql_q.state_dict(),
                    "iql_q_target": iql_q_target.state_dict(),
                    "iql_v": iql_v.state_dict(),
                }, save_path + f"iql_ckpt_latest.pt")

            n_gradient_step += 1
            if n_gradient_step >= args.gradient_steps:
                break

    # ---------------------- Evaluation ----------------------
    elif args.mode == "eval":
        save_path = f"{save_path}/models/{args.exp_name}/"

        # load actor
        actor.load(save_path + f"diffusion_ckpt_{args.ckpt}.pt")
        critic_ckpt = torch.load(save_path + f"iql_ckpt_{args.ckpt}.pt")
        iql_q.load_state_dict(critic_ckpt["iql_q"])
        iql_q_target.load_state_dict(critic_ckpt["iql_q_target"])
        iql_v.load_state_dict(critic_ckpt["iql_v"])


    else:
        raise ValueError(f"Invalid mode: {args.mode}")


def eval_idql(env, actor, iql_q, iql_v, iql_q_target, dataset, args, obs_dim, act_dim):
    """
    Evaluate the IDQL agent on the given environment.

    Args:
        env: The environment to evaluate on.
        actor: The diffusion policy model.
        iql_q: The IQL Q-value network.
        iql_v: The IQL V-value network.
        dataset: The dataset used for evaluation.
        args: Configuration arguments.
        obs_dim: Dimension of the observation space.
        act_dim: Dimension of the action space.

    Returns:
        A dictionary containing evaluation metrics such as mean rewards and episode lengths.
    """
    actor.eval()
    iql_q.eval()
    iql_v.eval()

    env_eval = gym.vector.make(args.task.env_name, args.num_envs)
    normalizer = dataset.get_normalizer()
    episode_rewards = []

    prior = torch.zeros((args.num_envs * args.num_candidates, act_dim), device=args.device)
    for i in range(args.num_episodes):

        obs, ep_reward, cum_done, t = env_eval.reset(), 0., 0., 0

        while not np.all(cum_done) and t < 1000 + 1:
            # normalize obs
            obs = torch.tensor(normalizer.normalize(obs), device=args.device)
            obs = obs.unsqueeze(1).repeat(1, args.num_candidates, 1).view(-1, obs_dim)

            # sample actions
            act, log = actor.sample(
                prior,
                solver=args.solver,
                n_samples=args.num_envs * args.num_candidates,
                sample_steps=args.sampling_steps,
                condition_cfg=obs, w_cfg=1.0,
                use_ema=args.use_ema, temperature=args.temperature)

            # resample
            with torch.no_grad():
                q = iql_q_target(obs, act)
                v = iql_v(obs)
                adv = (q - v)
                adv = adv.view(-1, args.num_candidates, 1)

                w = torch.softmax(adv * args.weight_temperature, 1)
                act = act.view(-1, args.num_candidates, act_dim)

                p = w / w.sum(1, keepdim=True)

                indices = torch.multinomial(p.squeeze(-1), 1).squeeze(-1)
                sampled_act = act[torch.arange(act.shape[0]), indices].cpu().numpy()

            # step
            obs, rew, done, info = env_eval.step(sampled_act)

            t += 1
            cum_done = done if cum_done is None else np.logical_or(cum_done, done)
            ep_reward += rew
            print(f'[t={t}] rew: {ep_reward}')

            if np.all(cum_done):
                break

        episode_rewards.append(np.clip(ep_reward, 0., 4.))

    episode_rewards = [list(map(lambda x: env.get_normalized_score(x), r)) for r in episode_rewards]
    episode_rewards = np.array(episode_rewards)
    print(np.mean(episode_rewards, -1), np.std(episode_rewards, -1))

    eval_log = {
        "mean_rewards": float(np.mean(episode_rewards)),
        "std_rewards": float(np.std(episode_rewards)),
        "mean_ep_len": float(np.mean([len(r) for r in episode_rewards])),
    }

    return eval_log


if __name__ == "__main__":
    pipeline()
