"""
Diffusion Policy (DP) Imitation Learning on Robomimic datasets.

DP learns a diffusion policy model that approximates the behavior policy in the dataset.
It can be trained with different neural network architectures, including Chi-UNet, Chi-Transformer, and DiT.
"""
import hydra
import os
import warnings
warnings.filterwarnings('ignore')

import gym
import pathlib
import time
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import random_split

from cleandiffuser.env.robomimic.robomimic_lowdim_wrapper import RobomimicLowdimWrapper
from cleandiffuser.env.wrapper import VideoRecordingWrapper, MultiStepWrapper
from cleandiffuser.env.utils import VideoRecorder
from cleandiffuser.dataset.robomimic_dataset import RobomimicDataset
from cleandiffuser.dataset.dataset_utils import loop_dataloader
from cleandiffuser.utils import report_parameters

from src.utils import set_seed, Logger
from src.contractive_diffusion import DiscreteDiffusionSDE as SDE


def make_env(args, idx):
    def thunk():
        import robomimic.utils.file_utils as FileUtils
        import robomimic.utils.env_utils as EnvUtils
        import robomimic.utils.obs_utils as ObsUtils

        def create_robomimic_env(env_meta, obs_keys):
            ObsUtils.initialize_obs_modality_mapping_from_dict(
                {'low_dim': obs_keys})
            env = EnvUtils.create_env_from_metadata(
                env_meta=env_meta,
                render=False,
                # only way to not show collision geometry
                # is to enable render_offscreen
                # which uses a lot of RAM.
                render_offscreen=False,
                use_image_obs=False,
            )
            return env

        dataset_path = os.path.expanduser(args.dataset_path)
        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
        abs_action = args.abs_action
        if abs_action:
            env_meta['env_kwargs']['controller_configs']['control_delta'] = False
        env = create_robomimic_env(env_meta=env_meta, obs_keys=args.task.obs_keys)
        env = RobomimicLowdimWrapper(
                env=env,
                obs_keys=args.task.obs_keys,
                init_state=None,
                render_hw=(256, 256),
                render_camera_name='agentview'
        )

        if args.save_video:
            video_recorder = VideoRecorder.create_h264(
                                fps=10,
                                codec='h264',
                                input_pix_fmt='rgb24',
                                crf=22,
                                thread_type='FRAME',
                                thread_count=1
                            )
            env = VideoRecordingWrapper(env, video_recorder, file_path=None, steps_per_render=2)

        env = MultiStepWrapper(env, n_obs_steps=args.task.obs_steps, n_action_steps=args.task.action_steps, max_episode_steps=args.task.max_episode_steps)
        env.seed(args.seed + idx)
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
        obs, t = envs.reset(), 0

        # initialize video stream
        if args.save_video:
            logger.video_init(envs.envs[0], enable=True, video_id=str(i))  # save videos

        while t < args.task.max_episode_steps:
            obs_seq = obs.astype(np.float32)  # (num_envs, obs_steps, obs_dim)
            # normalize obs
            nobs = dataset.normalizer['obs']['state'].normalize(obs_seq)
            nobs = torch.tensor(nobs, device=args.device, dtype=torch.float32)  # (num_envs, obs_steps, obs_dim)
            with torch.no_grad():
                if args.nn == 'chi_unet' or args.nn == 'dit':
                    # reshape observation to (num_envs, obs_horizon*obs_dim)
                    condition = nobs.flatten(start_dim=1)
                else:
                    # reshape observation to (num_envs, obs_horizon, obs_dim)
                    condition = nobs

                # run sampling (num_envs, horizon, action_dim)
                prior = torch.zeros((args.num_envs, args.horizon, args.action_dim), device=args.device)
                naction, _ = agent.sample(prior=prior, n_samples=args.num_envs, sample_steps=args.sample_steps, solver=solver,
                                    condition_cfg=condition, w_cfg=1.0, temperature=args.temperature, use_ema=True)

            # unnormalize prediction
            naction = naction.detach().to('cpu').numpy()  # (num_envs, horizon, action_dim)
            action_pred = dataset.normalizer['action'].unnormalize(naction)

            # get action
            start = args.obs_steps - 1
            end = start + args.action_steps
            action = action_pred[:, start:end, :]

            if args.abs_action:
                action = dataset.undo_transform_action(action)
            obs, reward, _, _ = envs.step(action)
            ep_reward += reward
            t += args.task.action_steps

        success = [1.0 if s > 0 else 0.0 for s in ep_reward]
        print(f"[Episode {1+i*(args.num_envs)}-{(i+1)*(args.num_envs)}] reward: {np.around(ep_reward, 2)} success:{success}")
        episode_rewards.append(ep_reward)
        episode_steps.append(t)
        episode_success.append(success)
    print(f"Mean step: {np.nanmean(episode_steps)} Mean reward: {np.nanmean(episode_rewards)} Mean success: {np.nanmean(episode_success)}")
    return {'mean_step': np.nanmean(episode_steps), 'mean_reward': np.nanmean(episode_rewards), 'mean_success': np.nanmean(episode_success)}


@hydra.main(config_path="../configs/dp/robomimic/chi_unet", config_name="lift_abs")
def pipeline(args):
    # --------------------- Create Path -----------------------
    set_seed(args.seed)

    save_path = f'{args.log_dir}/{args.pipeline_name}/{args.task.env_name}/'
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
    dataset = RobomimicDataset(dataset_path, horizon=args.horizon, obs_keys=args.task.obs_keys,
                               pad_before=args.task.obs_steps-1, pad_after=args.task.action_steps-1, abs_action=args.abs_action)

    # compute sizes
    available_data_rate = args.available_data_rate
    total_len = len(dataset)
    subset_len = int(total_len * available_data_rate)
    other_len  = total_len - subset_len

    # randomly split into “half” and “the rest”
    limited_dataset, _ = random_split(
        dataset,
        [subset_len, other_len],
        generator=torch.Generator().manual_seed(42)  # for reproducibility
    )


    dataloader = torch.utils.data.DataLoader(
        limited_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=True,
        # accelerate cpu-gpu transfer
        pin_memory=True,
        # don't kill worker process after each epoch
        persistent_workers=True
    )

    # --------------- Create Diffusion Model -----------------
    if args.nn == "dit":
        from cleandiffuser.nn_condition import MLPCondition
        from cleandiffuser.nn_diffusion import DiT1d

        nn_diffusion = DiT1d(
            args.action_dim, emb_dim=256, d_model=384, n_heads=12, depth=6, timestep_emb_type="fourier").to(args.device)
        nn_condition = MLPCondition(
            in_dim=args.obs_steps*args.obs_dim, out_dim=256, hidden_dims=[256, ], act=nn.ReLU(), dropout=0.25).to(args.device)
    elif args.nn == "chi_unet":
        from cleandiffuser.nn_condition import IdentityCondition
        from cleandiffuser.nn_diffusion import ChiUNet1d

        nn_diffusion = ChiUNet1d(
            args.action_dim, args.obs_dim, args.obs_steps, model_dim=256, emb_dim=256, dim_mult=[1, 2, 2],
            obs_as_global_cond=True, timestep_emb_type="positional").to(args.device)
        # dropout=0.0 to use no CFG but serve as FiLM encoder
        nn_condition = IdentityCondition(dropout=0.0).to(args.device)
    elif args.nn == "chi_transformer":
        from cleandiffuser.nn_condition import IdentityCondition
        from cleandiffuser.nn_diffusion import ChiTransformer

        nn_diffusion = ChiTransformer(
            args.action_dim, args.obs_dim, args.horizon, args.obs_steps, d_model=256, nhead=4, num_layers=8,
            timestep_emb_type="positional", p_drop_emb=0.0, p_drop_attn=0.3).to(args.device)
        # dropout=0.0 to use no CFG but serve as FiLM encoder
        nn_condition = IdentityCondition(dropout=0.0).to(args.device)
    else:
        raise ValueError(f"Invalid nn type {args.nn}")

    print(f"======================= Parameter Report of Diffusion Model =======================")
    report_parameters(nn_diffusion)
    print(f"===================================================================================")

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

            # preprocess
            nobs = batch['obs']['state'].to(args.device)
            naction = batch['action'].to(args.device)

            # get condition
            condition = nobs[:, :args.obs_steps, :]  # (B, obs_horizon, obs_dim)
            if args.nn == 'dit' or args.nn == 'chi_unet':
                condition = condition.flatten(start_dim=1)  # (B, obs_horizon*obs_dim)

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

        metrics = {'step': args.gradient_steps}
        metrics.update(inference(args, envs, dataset, agent, logger))
        logger.log(metrics, category='inference')

    else:
        raise ValueError("Illegal mode")


if __name__ == "__main__":
    pipeline()
