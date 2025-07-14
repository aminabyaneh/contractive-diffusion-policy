import hydra
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import gym
import pathlib
import time
import numpy as np
import torch
import torch.nn as nn

from cleandiffuser.env import pusht
from cleandiffuser.env.wrapper import VideoRecordingWrapper, MultiStepWrapper
from cleandiffuser.env.utils import VideoRecorder
from cleandiffuser.dataset.pusht_dataset import PushTStateDataset
from cleandiffuser.dataset.dataset_utils import loop_dataloader
from cleandiffuser.utils import report_parameters

from src.utils import set_seed, Logger
from src.contraction_diffusion import DiscreteDiffusionSDE as SDE


def make_env(args, idx):
    def thunk():
        env = gym.make(args.env_name)
        video_recorder = VideoRecorder.create_h264(
                            fps=10,
                            codec='h264',
                            input_pix_fmt='rgb24',
                            crf=22,
                            thread_type='FRAME',
                            thread_count=1
                        )
        env = VideoRecordingWrapper(env, video_recorder, file_path=None, steps_per_render=1)
        env = MultiStepWrapper(env, n_obs_steps=args.obs_steps, n_action_steps=args.action_steps, max_episode_steps=args.max_episode_steps)
        print("Env seed: ", args.seed + idx)
        return env

    return thunk


def inference(args, envs, dataset, agent, logger):
    """Evaluate a trained agent and optionally save a video."""
    # ---------------- Start Rollout ----------------
    episode_rewards = []
    episode_steps = []
    episode_success = []

    solver = args.solver

    for i in range(args.eval_episodes // args.num_envs):
        ep_reward = [0.0] * args.num_envs
        step_reward = []
        obs, t = envs.reset(), 0

        # initialize video stream
        if args.save_video:
            logger.video_init(envs.envs[0], enable=True, video_id=str(i))  # save videos

        while t < args.max_episode_steps:
            if args.env_name == 'pusht-v0':
                obs_seq = obs.astype(np.float32)  # (num_envs, obs_steps, obs_dim)
                # normalize obs
                nobs = dataset.normalizer['obs']['state'].normalize(obs_seq)
                nobs = torch.tensor(nobs, device=args.device, dtype=torch.float32)  # (num_envs, obs_steps, obs_dim)
            else:
                raise ValueError("fatal env")

            with torch.no_grad():
                if args.nn == "pearce_mlp" or args.nn == "pearce_transformer":
                    condition = nobs
                    # run sampling (num_envs, action_dim)
                    prior = torch.zeros((args.num_envs, args.action_dim), device=args.device)
                elif args.nn == 'dit':
                    # reshape observation to (num_envs, obs_horizon*obs_dim)
                    condition = nobs.flatten(start_dim=1)
                    # run sampling (num_envs, args.action_steps, action_dim)
                    prior = torch.zeros((args.num_envs, args.action_steps, args.action_dim), device=args.device)
                else:
                    ValueError("fatal nn")

                if not args.diffusion_x:
                    naction, _ = agent.sample(prior=prior, n_samples=args.num_envs, sample_steps=args.sample_steps,
                                              solver=solver, condition_cfg=condition, w_cfg=1.0, use_ema=True)
                else:
                    naction, _ = agent.sample_x(prior=prior, n_samples=args.num_envs, sample_steps=args.sample_steps,
                                                solver=solver, condition_cfg=condition, w_cfg=1.0, use_ema=True, extra_sample_steps=args.extra_sample_steps)

            # unnormalize prediction
            naction = naction.detach().to('cpu').clip(-1., 1.).numpy()  # (num_envs, action_dim)
            action_pred = dataset.normalizer['action'].unnormalize(naction)
            action = action_pred.reshape(args.num_envs, 1, args.action_dim)  # (num_envs, 1, action_dim)
            obs, reward, done, info = envs.step(action)

            ep_reward += reward
            step_reward.append(reward)
            t += args.action_steps

        ep_reward = np.around(np.array(ep_reward), 2)
        success = np.around(np.max(np.array(step_reward), axis=0), 2)

        episode_rewards.append(ep_reward)
        episode_steps.append(t)
        episode_success.append(success)

    return {'mean_step': np.nanmean(episode_steps), 'mean_reward': np.nanmean(episode_rewards), 'mean_success': np.nanmean(episode_success)}


@hydra.main(config_path="configs/dbc/pusht/pearce_mlp", config_name="pusht")
def pipeline(args):
    # --------------------- Create Path -----------------------
    set_seed(args.seed)

    save_path = f'{args.log_dir}/{args.pipeline_name}/{args.env_name}/'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    # ---------------------- Wandb Init -----------------------
    logger = Logger(pathlib.Path(save_path), args)

    # ------------------ Create Environment -------------------
    envs = gym.vector.SyncVectorEnv(
        [make_env(args, idx) for idx in range(args.num_envs)],
    )

    # ---------------- Create Dataset --------------------
    dataset_path = os.path.expanduser(args.dataset_path)
    if args.env_name == 'pusht-v0':
        dataset = PushTStateDataset(dataset_path, horizon=args.horizon, obs_keys=args.obs_keys,
                                pad_before=args.obs_steps-1, pad_after=args.action_steps-1, abs_action=args.abs_action)
    else:
        raise ValueError(f"Unsupported environment: {args.env_name}. Please check the configuration.")

    print(dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=True,
        # accelerate cpu-gpu transfer
        pin_memory=True,
        # don't kill worker process after each epoch
        persistent_workers=True
    )

    # --------------- Create Diffusion Model -----------------
    if args.nn == "pearce_mlp":
        from cleandiffuser.nn_condition import PearceObsCondition
        from cleandiffuser.nn_diffusion import PearceMlp

        nn_diffusion = PearceMlp(act_dim=args.action_dim, To=args.obs_steps, emb_dim=128,
                                 hidden_dim=512).to(args.device)
        nn_condition = PearceObsCondition(obs_dim=args.obs_dim, dropout=0.0).to(args.device)

    elif args.nn == "dit":
        from cleandiffuser.nn_condition import MLPCondition
        from cleandiffuser.nn_diffusion import DiT1d

        nn_diffusion = DiT1d(args.action_dim, emb_dim=256, d_model=384, n_heads=12,
                             depth=6, timestep_emb_type="fourier").to(args.device)
        nn_condition = MLPCondition(in_dim=args.obs_steps * args.obs_dim,
                                    out_dim=256, hidden_dims=[256, ], act=nn.ReLU(),
                                    dropout=0.25).to(args.device)
    else:
        raise ValueError(f"Invalid nn type {args.nn}")

    print(f"======================= Parameter Report of Diffusion Model =======================")
    report_parameters(nn_diffusion)
    print(f"==============================================================================")

    # ----------------- Diffusion Agent ----------------------
    args.diffusion_x = False  # SDE does not support diffusion_x
    agent = SDE(nn_diffusion, nn_condition, predict_noise=False,
                optim_params={"lr": args.lr},
                diffusion_steps=args.sample_steps,
                device=args.device,
                eigen_weight=args.loss_weights.eigen_max,
                lambda_contr=args.lambda_contr,
                jacobian_weight=args.loss_weights.jacobian,
                loss_type=args.loss_type)

    if args.mode == "train":
        # ----------------- Training ----------------------
        n_gradient_step = 0
        diffusion_loss_list = []
        jacobian_loss_list = []
        eigen_avg_loss_list = []
        eigen_max_loss_list = []
        jacobian_norm_list = []

        start_time = time.time()
        for batch in loop_dataloader(dataloader):
            nobs = batch['obs']['state'].to(args.device)
            naction = batch['action'].to(args.device)

            # diffusionBC
            condition = nobs[:, :args.obs_steps, :]  # (B, obs_horizon, obs_dim)
            if args.nn == "pearce_mlp":
                naction = naction[:, -1, :]  # (B, action_dim)
            elif args.nn == 'dit':
                condition = condition.flatten(start_dim=1)  # (B, obs_horizon*obs_dim)
                naction = naction[:, -args.action_steps:, :]  # (B, action_steps, action_dim)
            else:
                ValueError("fatal nn")

            # update diffusion
            diffusion_loss = agent.update(naction, condition)['loss']
            contraction_losses = agent.contraction_losses

            # log metrics
            diffusion_loss_list.append(diffusion_loss)
            jacobian_loss_list.append(contraction_losses['jacobian_loss'].item())
            eigen_max_loss_list.append(contraction_losses['eigen_max'].item())
            eigen_avg_loss_list.append(contraction_losses['eigen_avg'].item())
            jacobian_norm_list.append(contraction_losses['jacobian_norm'].item())

            if (n_gradient_step + 1) % args.log_freq == 0:
                metrics = {
                    'step': n_gradient_step + 1,
                    'total_time': time.time() - start_time,
                    'diffusion_loss': np.mean(diffusion_loss_list),
                    'jacobian_loss': np.mean(jacobian_loss_list),
                    'eigen_max_loss': np.mean(eigen_max_loss_list),
                    'eigen_avg_loss': np.mean(eigen_avg_loss_list),
                    'jacobian_norm': np.mean(jacobian_norm_list),
                }

                logger.log(metrics, category='train')

                diffusion_loss_list.clear()
                jacobian_loss_list.clear()
                eigen_max_loss_list.clear()
                eigen_avg_loss_list.clear()
                jacobian_norm_list.clear()

            if (n_gradient_step + 1) % args.eval_freq == 0:
                agent.model.eval()
                agent.model_ema.eval()
                metrics = {'step': n_gradient_step + 1}
                metrics.update(inference(args, envs, dataset, agent, logger))
                logger.log(metrics, category='eval')
                agent.model.train()
                agent.model_ema.train()

            n_gradient_step += 1
            if n_gradient_step > args.gradient_steps:
                break

    elif args.mode == "inference":
        # ----------------- Inference ----------------------
        if args.model_path:
            agent.load(args.model_path)
        else:
            raise ValueError("Empty model for inference")
        agent.model.eval()
        agent.model_ema.eval()

        metrics = {'step': 0}
        metrics.update(inference(args, envs, dataset, agent, logger))
        logger.log(metrics, category='inference')

    else:
        raise ValueError("Illegal mode")


if __name__ == "__main__":
    pipeline()











