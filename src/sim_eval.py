"""
Evaluation script for the contractive diffusion models in simulation.

The modules in this script evaluate a trained actor and critic in a mujoco reinforcement learning environment.
"""

import os
import contextlib
import torch
import gym
import numpy as np

from gym.vector import SyncVectorEnv


def eval(env, actor, critic, critic_target, dataset, args, obs_dim, act_dim):
    """
    Synchronous inference using a trained actor and critic.
    Avoids BrokenPipeError in parallel runs on a GPU server using SyncVectorEnv (no multiprocessing).

    Args:
        env (gym.Env): The environment to evaluate the model on.
        actor (DiscreteDiffusionSDE): The diffusion model.
        critic (DQLCritic): The critic model.
        critic_target (DQLCritic): The target critic model.
        dataset (D4RLKitchenTDDataset): The dataset used for training.
        args (Namespace): The arguments used for training.
        obs_dim (int): The dimension of the observation space.
        act_dim (int): The dimension of the action space.

    Returns:
        dict: A dictionary containing the mean and std of the rewards.
            {
                "mean_rew": float,  # mean of the rewards
                "std_rew": float    # std of the rewards
            }
    """
    actor.eval()
    critic.eval()
    critic_target.eval()

    # suppress stdout/stderr during env setup
    with open(os.devnull, 'w') as devnull, \
         contextlib.redirect_stdout(devnull), \
         contextlib.redirect_stderr(devnull):
        # vectorized gym environment (Running in parallel demands high RAM usage)
        env_eval = gym.vector.make(args.task.env_name, num_envs=args.num_envs)

    normalizer = dataset.get_normalizer()
    episode_rewards = []

    prior = torch.zeros((args.num_envs * args.num_candidates, act_dim), device=args.device)

    for _ in range(args.num_episodes):
        obs, ep_reward, cum_done, t = env_eval.reset(), 0., 0., 0

        while not np.all(cum_done) and t < args.max_episode_steps + 1:
            # normalize obs
            obs_tensor = torch.tensor(normalizer.normalize(obs), device=args.device, dtype=torch.float32)
            obs_tensor = obs_tensor.unsqueeze(1).repeat(1, args.num_candidates, 1).view(-1, obs_dim)

            # sample actions
            act, _ = actor.sample(
                prior,
                solver=args.solver,
                n_samples=args.num_envs * args.num_candidates,
                sample_steps=args.sampling_steps,
                condition_cfg=obs_tensor,
                w_cfg=1.0,
                use_ema=args.use_ema,
                temperature=args.temperature
            )

            # resample
            with torch.no_grad():
                q = critic_target.q_min(obs_tensor, act)
                q = q.view(-1, args.num_candidates, 1)
                w = torch.softmax(q * args.task.weight_temperature, dim=1)
                act = act.view(-1, args.num_candidates, act_dim)

                indices = torch.multinomial(w.squeeze(-1), 1).squeeze(-1)
                sampled_act = act[torch.arange(act.shape[0]), indices].cpu().numpy()

            # step
            obs, rew, done, _ = env_eval.step(sampled_act)

            t += 1
            ep_reward += (rew * (1 - cum_done)) if t < args.max_episode_steps else rew
            cum_done = done if cum_done is None else np.logical_or(cum_done, done)

        if args.env_name == "kitchen":
            ep_reward = np.clip(ep_reward, 0.0, 4.0)
        elif args.env_name == "antmaze":
            ep_reward = np.clip(ep_reward, 0.0, 1.0)

        episode_rewards.append(ep_reward)

    # normalize rewards
    episode_rewards = [list(map(lambda x: env.get_normalized_score(x), r)) for r in episode_rewards]
    episode_rewards = np.array(episode_rewards)

    # back to train mode
    actor.train()
    critic.train()
    critic_target.train()

    return {
        "mean_rew": np.mean(np.mean(episode_rewards, -1), -1),
        "std_rew": np.mean(np.std(episode_rewards, -1), -1)
    }
