"""
Evaluation script for the contractive diffusion models in simulation.

The modules in this script evaluate a trained actor and critic in a mujoco reinforcement learning environment.
"""

import os
import torch
import gym
import contextlib
import numpy as np


def eval(env, actor, critic, critic_target, dataset, args, obs_dim, act_dim):
    """
    Inference function using a trained actor and critic. Works on dql, iql, and edp.
    This is used to evaluate the performance of the model during training.

    Args:
        actor: DiscreteDiffusionSDE or similar diffusion actor,
            The diffusion model.
        critic: DQLCritic or similar critic,
            The critic model.
        critic_target: DQLCritic or similar critic,
            The target critic model.
        dataset: D4RLKitchenTDDataset,
            The dataset used for training.
        args: Namespace,
            The arguments used for training.
        obs_dim: int,
            The dimension of the observation space.
        act_dim: int,
            The dimension of the action space.
    """
    # eval mode
    actor.eval()
    critic.eval()
    critic_target.eval()

    # set up env and suppress output
    with open(os.devnull, 'w') as devnull, \
     contextlib.redirect_stdout(devnull), \
     contextlib.redirect_stderr(devnull):
        env_eval = gym.vector.make(args.task.env_name, args.num_envs)
    normalizer = dataset.get_normalizer()
    episode_rewards = []

    prior = torch.zeros((args.num_envs * args.num_candidates, act_dim), device=args.device)
    for i in range(args.num_episodes):
        # reset env
        obs, ep_reward, cum_done, t = env_eval.reset(), 0., 0., 0

        while not np.all(cum_done) and t < 280 + 1:
            # normalize obs
            obs = torch.tensor(normalizer.normalize(obs), device=args.device, dtype=torch.float32)
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
                q = critic_target.q_min(obs, act)
                q = q.view(-1, args.num_candidates, 1)
                w = torch.softmax(q * args.task.weight_temperature, 1)
                act = act.view(-1, args.num_candidates, act_dim)
                indices = torch.multinomial(w.squeeze(-1), 1).squeeze(-1)
                sampled_act = act[torch.arange(act.shape[0]), indices].cpu().numpy()

            # step
            obs, rew, done, _ = env_eval.step(sampled_act)
            cum_done = done if cum_done is None else np.logical_or(cum_done, done)
            ep_reward += rew
            t += 1

            if np.all(cum_done):
                break

        episode_rewards.append(np.clip(ep_reward, 0., 4.))

    # normalize rewards
    episode_rewards = [list(map(lambda x: env.get_normalized_score(x), r)) for r in episode_rewards]
    episode_rewards = np.array(episode_rewards)

    return { # mean over all episodes' reward mean and std
        "mean_rew": np.mean(np.mean(episode_rewards, -1), -1),
        "std_rew": np.mean(np.std(episode_rewards, -1), -1)
    }
