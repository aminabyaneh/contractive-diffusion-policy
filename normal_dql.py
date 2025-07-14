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
from cleandiffuser.diffusion import DiscreteDiffusionSDE
from cleandiffuser.nn_condition import IdentityCondition
from cleandiffuser.nn_diffusion import DQLMlp
from cleandiffuser.utils import report_parameters, DQLCritic, FreezeModules, at_least_ndim

from src.contraction_loss import compute_contractive_loss
from src.utils import Logger, set_seed


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

    for _ in range(args.num_episodes):
        obs, ep_reward, cum_done, t = env_eval.reset(), 0., 0., 0

        while not np.all(cum_done) and t < args.max_episode_steps + 1:
            # normalize obs
            obs_tensor = torch.tensor(normalizer.normalize(obs), device=args.device, dtype=torch.float32)
            obs_tensor = obs_tensor.unsqueeze(1).repeat(1, args.num_candidates, 1).view(-1, obs_dim)

            # sample actions
            act = actor(obs_tensor)

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


class DeterministicActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
            nn.Tanh(),  # assuming actions in [-1,1]
        )

    def forward(self, obs):
        return self.net(obs)


class GaussianActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=512, log_std_min=-20, log_std_max=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_layer = nn.Linear(hidden_dim, act_dim)
        self.log_std_layer = nn.Linear(hidden_dim, act_dim)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, obs):
        x = self.net(obs)
        mean = torch.tanh(self.mean_layer(x))  # assuming actions in [-1,1]
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        return mean, std

    def sample(self, obs):
        mean, std = self.forward(obs)
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # reparameterization trick
        action = torch.tanh(x_t)
        return action, mean, std


@hydra.main(config_path="configs/dql", config_name="main", version_base=None)
def pipeline(args):
    """
    Main pipeline for training and evaluating contractive DQL on D4RL datasets.
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
    nn_diffusion = DQLMlp(obs_dim, act_dim, emb_dim=64, timestep_emb_type="positional").to(args.device)
    nn_condition = IdentityCondition(dropout=0.0).to(args.device) # TODO: Try other condition networks

    print(f"================================ Diffusion Model ==================================")
    report_parameters(nn_diffusion)
    print(f"===================================================================================\n")

    # --------------- Deterministic Actor --------------------
    actor = DeterministicActor(obs_dim, act_dim, hidden_dim=args.hidden_dim).to(args.device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args.actor_learning_rate)

    # ---------------------- Critic ------------------------
    critic = DQLCritic(obs_dim, act_dim, hidden_dim=args.hidden_dim).to(args.device)

    # target copy for stability
    critic_target = deepcopy(critic).requires_grad_(False).eval()
    critic_optim = torch.optim.Adam(critic.parameters(), lr=args.critic_learning_rate)

    # ---------------------- Training ----------------------
    if args.mode == "train":
        start_time = time.time()

        actor_lr_scheduler = CosineAnnealingLR(actor_optimizer, T_max=args.gradient_steps)
        critic_lr_scheduler = CosineAnnealingLR(critic_optim, T_max=args.gradient_steps)

        actor.train()
        critic.train()

        n_gradient_step = 0
        log = {"step": 0, "time": 0, "bc_loss": 0., "q_loss": 0., "critic_loss": 0., "target_q_mean": 0.,
               "target_q_std": 0.}

        for batch in loop_dataloader(dataloader):
            # load batch of data
            obs, next_obs = batch["obs"]["state"].to(args.device), batch["next_obs"]["state"].to(args.device)
            act = batch["act"].to(args.device)
            rew = batch["rew"].to(args.device)
            tml = batch["tml"].to(args.device)

            # ---------------------- Critic Training ----------------------
            current_q1, current_q2 = critic(obs, act)

            # sample actions
            next_act = actor(obs)

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
            bc_loss = F.mse_loss(next_act, act)

            with FreezeModules([critic, ]):
                q1_new_action, q2_new_action = critic(obs, next_act)
            if np.random.uniform() > 0.5:
                q_loss = - q1_new_action.mean() / q2_new_action.abs().mean().detach()
            else:
                q_loss = - q2_new_action.mean() / q1_new_action.abs().mean().detach()

            # compute the total actor loss
            actor_loss = bc_loss + args.task.eta * q_loss

            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            actor_lr_scheduler.step()
            critic_lr_scheduler.step()

            # EMA update
            with torch.no_grad():
                if n_gradient_step % args.ema_update_interval == 0:
                    # Polyak‐average update for the ensemble critic targets
                    for param, target_param in zip(critic.parameters(), critic_target.parameters()):
                        target_param.data.copy_(0.995 * param.data + (1 - 0.995) * target_param.data)

            # ---------------------- Logging ----------------------
            log["bc_loss"] += bc_loss.item()
            log["q_loss"] += q_loss.item()
            log["critic_loss"] += critic_loss.item()
            log["target_q_mean"] += target_q.mean().item()
            log["target_q_std"] += target_q.std().item()


            if (n_gradient_step + 1) % args.log_interval == 0:
                log["bc_loss"] /= args.log_interval
                log["q_loss"] /= args.log_interval
                log["critic_loss"] /= args.log_interval
                log["target_q_mean"] /= args.log_interval
                log["time"] = (time.time() - start_time) / 60
                log["step"] = n_gradient_step + 1
                log["target_q_std"] /= args.log_interval

                # log to console and wandb
                logger.log(log, category='train')
                log = {"step": 0, "time": 0, "bc_loss": 0., "q_loss": 0., "critic_loss": 0.,
                       "target_q_mean": 0., "target_q_std": 0.}

            # evaluation
            eval_time = time.time()
            if (n_gradient_step + 1) % args.eval_interval == 0:
                eval_log = eval(env, actor, critic, critic_target, dataset, args, obs_dim, act_dim)
                eval_time = (time.time() - eval_time)
                eval_log["step"] = n_gradient_step + 1
                logger.log(eval_log, category='eval')

            # # saving
            if (n_gradient_step + 1) % args.save_interval == 0:
                # save diffusion actor checkpoints
                save_path = f"{save_path}/models/{args.exp_name}/"
                os.makedirs(save_path, exist_ok=True)

                torch.save({
                    "actor": actor.state_dict(),
                }, save_path + f"diffusion_ckpt_{n_gradient_step + 1}.pt")

                torch.save({
                    "actor": actor.state_dict(),
                }, save_path + f"diffusion_ckpt_latest.pt")

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
