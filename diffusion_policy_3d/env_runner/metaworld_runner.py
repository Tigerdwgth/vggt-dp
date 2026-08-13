import wandb
import numpy as np
import torch
import collections
import tqdm
from diffusion_policy_3d.env import MetaWorldEnv
from diffusion_policy_3d.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy_3d.gym_util.video_recording_wrapper import SimpleVideoRecordingWrapper

from diffusion_policy_3d.policy.base_policy import BasePolicy
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.env_runner.base_runner import BaseRunner
import diffusion_policy_3d.common.logger_util as logger_util
from termcolor import cprint
import open3d as o3d
import os

class MetaworldRunner(BaseRunner):
    def __init__(self,
                 output_dir,
                 eval_episodes=20,
                 max_steps=1000,
                 n_obs_steps=8,
                 n_action_steps=8,
                 fps=10,
                 crf=22,
                 render_size=84,
                 tqdm_interval_sec=5.0,
                 n_envs=None,
                 task_name=None,
                 n_train=None,
                 n_test=None,
                 device="cuda:0",
                 use_point_crop=True,
                 num_points=512
                 ):
        super().__init__(output_dir)
        self.task_name = task_name


        def env_fn(task_name):
            return MultiStepWrapper(
                SimpleVideoRecordingWrapper(
                    MetaWorldEnv(task_name=task_name,device=device, 
                                 use_point_crop=use_point_crop, num_points=num_points)),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps,
                reward_agg_method='sum',
            )
        self.eval_episodes = eval_episodes
        self.env = env_fn(self.task_name)

        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec

        self.logger_util_test = logger_util.LargestKRecorder(K=3)
        self.logger_util_test10 = logger_util.LargestKRecorder(K=5)

    def run(self, policy: BasePolicy, save_video=True):
        device = policy.device
        dtype = policy.dtype

        all_traj_rewards = []
        all_success_rates = []
        env = self.env
        
        # 保存视频的相关变量
        success_videos = []
        failed_videos = []
        has_success = False
        has_failure = False

        
        for episode_idx in tqdm.tqdm(range(self.eval_episodes), desc=f"Eval in Metaworld {self.task_name} Pointcloud Env", leave=False, mininterval=self.tqdm_interval_sec):
            
            # start rollout
            obs = env.reset()
            policy.reset()

            done = False
            traj_reward = 0
            is_success = False
            while not done:
                np_obs_dict = dict(obs)
                obs_dict = dict_apply(np_obs_dict,
                                      lambda x: torch.from_numpy(x).to(
                                          device=device))

                with torch.no_grad():
                    obs_dict_input = {}
                    obs_dict_input['point_cloud'] = obs_dict['point_cloud'].unsqueeze(0)
                    obs_dict_input['agent_pos'] = obs_dict['agent_pos'].unsqueeze(0)
                    obs_dict_input['image'] = obs_dict['image'].unsqueeze(0)
                    action_dict = policy.predict_action(obs_dict_input)

                np_action_dict = dict_apply(action_dict,
                                            lambda x: x.detach().to('cpu').numpy())
                action = np_action_dict['action'].squeeze(0)

                obs, reward, done, info = env.step(action)


                traj_reward += reward
                done = np.all(done)
                is_success = is_success or max(info['success'])

            all_success_rates.append(is_success)
            all_traj_rewards.append(traj_reward)
            
            # 获取当前episode的视频
            episode_videos = env.env.get_video()
            if len(episode_videos.shape) == 5:
                episode_videos = episode_videos[:, 0]  # select first frame
            
            # 根据成功与否保存视频
            if is_success:
                if not has_success:  # 只保存第一个成功的视频
                    success_videos = episode_videos
                    has_success = True
            else:
                if not has_failure:  # 只保存第一个失败的视频
                    failed_videos = episode_videos
                    has_failure = True
            
            # 如果已经有成功和失败的视频，可以提前结束（可选）
            # if has_success and has_failure:
            #     break

        max_rewards = collections.defaultdict(list)
        log_data = dict()

        log_data['mean_traj_rewards'] = np.mean(all_traj_rewards)
        log_data['mean_success_rates'] = np.mean(all_success_rates)

        log_data['test_mean_score'] = np.mean(all_success_rates)
        
        cprint(f"test_mean_score: {np.mean(all_success_rates)}", 'green')

        self.logger_util_test.record(np.mean(all_success_rates))
        self.logger_util_test10.record(np.mean(all_success_rates))
        log_data['SR_test_L3'] = self.logger_util_test.average_of_largest_K()
        log_data['SR_test_L5'] = self.logger_util_test10.average_of_largest_K()
        
        if save_video:
            # 优先保存成功的视频，如果没有成功的则保存失败的视频
            if has_success:
                cprint(f"Saving successful episode video", 'green')
                videos_wandb = wandb.Video(success_videos, fps=self.fps, format="mp4")
                log_data[f'sim_video_eval_success'] = videos_wandb
            elif has_failure:
                cprint(f"Saving failed episode video", 'red')
                videos_wandb = wandb.Video(failed_videos, fps=self.fps, format="mp4")
                log_data[f'sim_video_eval_failure'] = videos_wandb
            else:
                cprint(f"No videos to save", 'yellow')
            
            # 如果需要同时保存成功和失败的视频，可以取消注释下面的代码
            # if has_success and has_failure:
            #     videos_wandb_success = wandb.Video(success_videos, fps=self.fps, format="mp4")
            #     videos_wandb_failure = wandb.Video(failed_videos, fps=self.fps, format="mp4")
            #     log_data[f'sim_video_eval_success'] = videos_wandb_success
            #     log_data[f'sim_video_eval_failure'] = videos_wandb_failure

            # 以下是原有的保存逻辑（已注释）
            # save agent poseo
            # disturbance = env.env.disturbance
            # agent_poses = env.env.agent_poses
            # os.makedirs(f"{self.output_dir}/videos", exist_ok=True)
            # import dill
            # with open(f"{self.output_dir}/videos/{self.task_name}_{disturbance}_agent_poses.pkl", 'wb') as f:
            #     dill.dump(agent_poses, f) 
          
            # save_video_path = f"{self.output_dir}/videos/{self.task_name}_{disturbance}_invis_eval.mp4"
            # print(len(videos)," ", videos.shape)
            # print(f"Saving video to {save_video_path}")
            # # 201   (201, 3, 128, 128) ->201 128 128 3
            # videos = np.transpose(videos, (0, 2, 3, 1))
            # import cv2
            # # save as mp4
            # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # out = cv2.VideoWriter(save_video_path, fourcc, 10, (videos.shape[2], videos.shape[1]))
            # for i in range(videos.shape[0]):
            #     out.write(videos[i])
            # out.release()
            # # #save ply
            # save_pc_path = f"{self.output_dir}/videos/{self.task_name}_{disturbance}_pc"
            # os.makedirs(save_pc_path, exist_ok=True)  # 确保目录存在
            
            # for i, pc in enumerate(env.env.pcs):
            #     # 检查点云数据
            #     # print(f"pc shape: {pc.shape}, dtype: {pc.dtype}")
            #     if pc.ndim != 2 or pc.shape[1] not in [3, 6]:
            #         raise ValueError(f"Invalid point cloud shape: {pc.shape}. Expected shape (N, 3) or (N, 6).")
                
            #     # 确保数据类型正确
            #     pc = pc.astype(np.float32)

            #     # 将 NumPy 数组转换为 Open3D 点云对象
            #     point_cloud = o3d.geometry.PointCloud()
            #     point_cloud.points = o3d.utility.Vector3dVector(pc[:, :3])  # 前 3 列是点的坐标

            #     # 如果点云包含 RGB 信息，处理 RGB 数据
            #     if pc.shape[1] == 6:
            #         colors = pc[:, 3:6] / 255.0  # 假设 RGB 值在 0-255 范围内，归一化到 0-1
            #         point_cloud.colors = o3d.utility.Vector3dVector(colors)

            #     # 保存为 .ply 文件
            #     o3d.io.write_point_cloud(save_pc_path + f"/{i}.ply", point_cloud)

            # print(f"Saving video to {save_video_path}")
            # env.env.save_video(save_video_path, videos, fps=self.fps, crf=self.crf)
            
            # videos_wandb = wandb.Video(videos, fps=self.fps, format="mp4")
            # log_data[f'sim_video_eval'] = videos_wandb

        _ = env.reset()

        return log_data
