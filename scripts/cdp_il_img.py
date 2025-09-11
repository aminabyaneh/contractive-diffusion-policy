import hydra
import sys
import os
import warnings
warnings.filterwarnings('ignore')

import pathlib
import time
import collections
import numpy as np
import torch
import torch.nn as nn

from cleandiffuser.env.robomimic.robomimic_image_wrapper import RobomimicImageWrapper
from cleandiffuser.env.wrapper import VideoRecordingWrapper, MultiStepWrapper
from cleandiffuser.env.utils import VideoRecorder
from cleandiffuser.env.async_vector_env import AsyncVectorEnv
from cleandiffuser.dataset.robomimic_dataset import RobomimicImageDataset
from cleandiffuser.dataset.dataset_utils import loop_dataloader
from cleandiffuser.utils import report_parameters

# add source to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from source.utils import set_seed, Logger
from source.contractive_diffusion import DiscreteDiffusionSDE as SDE

def make_async_envs(args):
    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.obs_utils as ObsUtils

    def create_robomimic_env(env_meta, shape_meta, enable_render=True):
        modality_mapping = collections.defaultdict(list)
        for key, attr in shape_meta['obs'].items():
            modality_mapping[attr.get('type', 'low_dim')].append(key)
        ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)

        env = EnvUtils.create_env_from_metadata(
            env_meta=env_meta,
            render=False,
            render_offscreen=False,
            use_image_obs=enable_render,
        )
        return env

    dataset_path = os.path.expanduser(args.dataset_path)
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
    # disable object state observation
    env_meta['env_kwargs']['use_object_obs'] = False
    abs_action = args.abs_action
    if abs_action:
        env_meta['env_kwargs']['controller_configs']['control_delta'] = False

    def env_fn():
        env = create_robomimic_env(
            env_meta=env_meta,
            shape_meta=args.task.shape_meta
        )
        # Robosuite's hard reset causes excessive memory consumption.
        # Disabled to run more envs.
        # https://github.com/ARISE-Initiative/robosuite/blob/92abf5595eddb3a845cd1093703e5a3ccd01e77e/robosuite/environments/base.py#L247-L248
        env.env.hard_reset = False
        return MultiStepWrapper(
            VideoRecordingWrapper(
                RobomimicImageWrapper(
                    env=env,
                    shape_meta=args.task.shape_meta,
                    init_state=None,
                    render_obs_key=args.task.render_obs_key
                ),
                video_recoder=VideoRecorder.create_h264(
                    fps=10,
                    codec='h264',
                    input_pix_fmt='rgb24',
                    crf=22,
                    thread_type='FRAME',
                    thread_count=1
                ),
                file_path=None,
                steps_per_render=2
            ),
            n_obs_steps=args.task.obs_steps,
            n_action_steps=args.task.action_steps,
            max_episode_steps=args.task.max_episode_steps
        )

    # See https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/env_runner/robomimic_image_runner.py
    # For each process the OpenGL context can only be initialized once
    # Since AsyncVectorEnv uses fork to create worker process,
    # a separate env_fn that does not create OpenGL context (enable_render=False)
    # is needed to initialize spaces.
    def dummy_env_fn():
        env = create_robomimic_env(
                env_meta=env_meta,
                shape_meta=args.task.shape_meta,
                enable_render=False
            )
        return MultiStepWrapper(
            VideoRecordingWrapper(
                RobomimicImageWrapper(
                    env=env,
                    shape_meta=args.task.shape_meta,
                    init_state=None,
                    render_obs_key=args.task.render_obs_key
                ),
                video_recoder=VideoRecorder.create_h264(
                    fps=10,
                    codec='h264',
                    input_pix_fmt='rgb24',
                    crf=22,
                    thread_type='FRAME',
                    thread_count=1
                ),
                file_path=None,
                steps_per_render=2
            ),
            n_obs_steps=args.task.obs_steps,
            n_action_steps=args.task.action_steps,
            max_episode_steps=args.task.max_episode_steps
        )

    env_fns = [env_fn] * args.num_envs
    # env_fn() and dummy_env_fn() should be function!
    envs = AsyncVectorEnv(env_fns, dummy_env_fn=dummy_env_fn)
    envs.seed(args.seed)
    return envs


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
            obs_dict = {}
            for k in obs.keys():
                obs_seq = obs[k].astype(np.float32)  # (num_envs, obs_steps, obs_dim)
                nobs = dataset.normalizer['obs'][k].normalize(obs_seq)
                obs_dict[k] = nobs = torch.tensor(nobs, device=args.device, dtype=torch.float32)  # (num_envs, obs_steps, obs_dim)
            with torch.no_grad():
                condition = obs_dict
                if args.nn == "pearce_mlp":
                    # run sampling (num_envs, action_dim)
                    prior = torch.zeros((args.num_envs, args.task.action_dim), device=args.device)
                elif args.nn == 'dit':
                    # run sampling (num_envs, args.task.action_steps, action_dim)
                    prior = torch.zeros((args.num_envs, args.task.action_steps, args.task.action_dim), device=args.device)
                else:
                    raise ValueError("NN type not supported")

                if not args.diffusion_x:
                    naction, _ = agent.sample(prior=prior, n_samples=args.num_envs, sample_steps=args.sample_steps, solver=solver,
                                        condition_cfg=condition, w_cfg=1.0, use_ema=True)
                else:
                    naction, _ = agent.sample_x(prior=prior, n_samples=args.num_envs, sample_steps=args.sample_steps, solver=solver,
                                        condition_cfg=condition, w_cfg=1.0, use_ema=True, extra_sample_steps=args.extra_sample_steps)

            # unnormalize prediction
            naction = naction.detach().to('cpu').clip(-1., 1.).numpy()  # (num_envs, action_dim)
            action_pred = dataset.normalizer['action'].unnormalize(naction)
            action = action_pred.reshape(args.num_envs, 1, args.task.action_dim)  # (num_envs, 1, action_dim)

            if args.abs_action:
                action = dataset.undo_transform_action(action)
            obs, reward, _, _ = envs.step(action)
            ep_reward += reward
            t += args.task.action_steps

        success = [1.0 if s > 0 else 0.0 for s in ep_reward]
        episode_rewards.append(ep_reward)
        episode_steps.append(t)
        episode_success.append(success)

    return {'mean_step': np.nanmean(episode_steps), 'mean_reward': np.nanmean(episode_rewards), 'mean_success': np.nanmean(episode_success)}


@hydra.main(config_path="../configs/dbc/robomimic_image", config_name="main")
def pipeline(args):
    # --------------------- Create Path -----------------------
    set_seed(args.seed)

    save_path = f'{args.log_dir}/{args.pipeline_name}/{args.env_name}/'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    # ---------------------- Wandb Init -----------------------
    logger = Logger(pathlib.Path(save_path), args)

    # ---------------- Create Environment ----------------
    envs = make_async_envs(args)

    # ---------------- Create Dataset --------------------
    dataset_path = os.path.expanduser(args.dataset_path)
    dataset = RobomimicImageDataset(dataset_path, horizon=args.horizon, shape_meta=args.task.shape_meta,
                                    n_obs_steps=args.task.obs_steps, pad_before=args.task.obs_steps-1,
                                    pad_after=args.task.action_steps-1, abs_action=args.abs_action)
    print(dataset)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=8,
        shuffle=True,
        # accelerate cpu-gpu transfer
        pin_memory=True,
        # don't kill worker process after each epoch
        persistent_workers=True
    )

    # --------------- Create Diffusion Model -----------------
    if args.nn == "pearce_mlp":
        from cleandiffuser.nn_condition import MultiImageObsCondition
        from cleandiffuser.nn_diffusion import PearceMlp

        nn_diffusion = PearceMlp(act_dim=args.task.action_dim, To=args.task.obs_steps, emb_dim=256, hidden_dim=512).to(args.device)
        nn_condition = MultiImageObsCondition(
            shape_meta=args.task.shape_meta, emb_dim=256, rgb_model_name=args.rgb_model, resize_shape=args.resize_shape,
            crop_shape=args.crop_shape, random_crop=args.random_crop, use_group_norm=args.use_group_norm,
            use_seq=args.use_seq).to(args.device)
    elif args.nn == "dit":
        from cleandiffuser.nn_condition import MultiImageObsCondition
        from cleandiffuser.nn_diffusion import DiT1d

        nn_condition = MultiImageObsCondition(
            shape_meta=args.task.shape_meta, emb_dim=256, rgb_model_name=args.rgb_model, resize_shape=args.resize_shape,
            crop_shape=args.crop_shape, random_crop=args.random_crop,
            use_group_norm=args.use_group_norm, use_seq=args.use_seq).to(args.device)
        nn_diffusion = DiT1d(
            args.task.action_dim, emb_dim=256*args.task.obs_steps, d_model=320, n_heads=10, depth=2, timestep_emb_type="fourier").to(args.device)
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
            # preprocessing
            nobs = batch['obs']
            condition = {}
            for k in nobs.keys():
                condition[k] = nobs[k][:, :args.task.obs_steps, :].to(args.device)

            naction = batch['action'].to(args.device)
            if args.nn == "pearce_mlp":
                naction = naction[:, -1, :]  # (B, action_dim)
            elif args.nn == 'dit':
                naction = naction[:, -args.task.action_steps:, :]  # (B, action_steps, action_dim)
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

        metrics = {'step': args.gradient_steps}
        metrics.update(inference(args, envs, dataset, agent, logger))
        logger.log(metrics, category='inference')

    else:
        raise ValueError("Illegal mode")


if __name__ == "__main__":
    pipeline()
