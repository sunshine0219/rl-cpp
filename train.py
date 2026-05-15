import argparse #接收命令行参数
import importlib
import json
import os
from argparse import BooleanOptionalAction
from datetime import datetime
from stable_baselines3.common.monitor import Monitor #Stable-Baselines3 的环境包装器

from rlm.architectures import StackedMapFeaturesExtractor # 自定义特征提取器
from rlm.mower_env import MowerEnv # 自定义环境类


def main():
    parser = argparse.ArgumentParser()
    ## agent 参数组
    '''
    algo：用 SAC 还是 PPO
    learning_rate：学习率
    cnn：是否用 CNN 特征提取器
    cnn_dims：隐藏层维度
    grouped_convs：是否使用 grouped convolution,这对应论文里的 SGCNN
    buffer_size：经验回放池大小（SAC 用）
    '''
    agent_args = parser.add_argument_group('agent')
    agent_args.add_argument('--algo', default='SAC', type=str)
    agent_args.add_argument('--learning_rate', default=2e-5, type=float)
    agent_args.add_argument('--cnn', default=True, action=BooleanOptionalAction)
    agent_args.add_argument('--cnn_dims', default=256, type=int)
    agent_args.add_argument('--grouped_convs', default=True, action=BooleanOptionalAction)
    agent_args.add_argument('--buffer_size', default=500_000, type=int)
    agent_args.add_argument('--train_freq', default=1, type=int)
    agent_args.add_argument('--gradient_steps', default=1, type=int)
    ## train 参数组
    '''
    checkpoint：从已有模型继续训练
    steps：总训练步数
    logdir：日志和模型保存目录
    '''
    train_args = parser.add_argument_group('train')
    train_args.add_argument('--checkpoint', default=None, type=str)
    train_args.add_argument('--steps', default=1_000_000, type=int)
    train_args.add_argument('--logdir', default=None, type=str)
    ## env 参数组
    # 多尺度地图设置
    env_args = parser.add_argument_group('env')
    env_args.add_argument('--input_size', default=32, type=int)
    env_args.add_argument('--num_maps', default=4, type=int)
    env_args.add_argument('--scale_factor', default=4, type=float)
    env_args.add_argument('--meters_per_pixel', default=0.0375, type=float)
    env_args.add_argument('--min_size', default=None, type=int)
    env_args.add_argument('--max_size', default=None, type=int)
    env_args.add_argument('--stacks', default=1, type=int)
    env_args.add_argument('--step_size', default=0.5, type=float)
    env_args.add_argument('--constant_lin_vel', default=True, action=BooleanOptionalAction)
    env_args.add_argument('--max_lin_vel', default=0.26, type=float)
    env_args.add_argument('--max_ang_vel', default=1.0, type=float)
    env_args.add_argument('--max_lin_acc', default=None, type=float)
    env_args.add_argument('--max_ang_acc', default=None, type=float)
    env_args.add_argument('--action_delay', default=0, type=float)
    env_args.add_argument('--steering_limits_lin_vel', default=True, action=BooleanOptionalAction)
    env_args.add_argument('--mower_radius', default=0.15, type=float)
    env_args.add_argument('--lidar_rays', default=24, type=int)
    env_args.add_argument('--lidar_range', default=3.5, type=float)
    env_args.add_argument('--lidar_fov', default=345, type=float)
    env_args.add_argument('--position_noise', default=0.01, type=float)
    env_args.add_argument('--heading_noise', default=0.05, type=float)
    env_args.add_argument('--lidar_noise', default=0.05, type=float)
    # 任务设置
    env_args.add_argument('--exploration', default=False, action=BooleanOptionalAction)
    env_args.add_argument('--overlap_observation', default=True, action=BooleanOptionalAction)
    env_args.add_argument('--frontier_observation', default=True, action=BooleanOptionalAction)
    env_args.add_argument('--action_observations', default=0, type=int)
    env_args.add_argument('--eval', default=False, action=BooleanOptionalAction)
    env_args.add_argument('--p_use_known_obstacles', default=0.7, type=float)
    env_args.add_argument('--p_use_unknown_obstacles', default=0.7, type=float)
    env_args.add_argument('--p_use_floor_plans', default=0.7, type=float)
    env_args.add_argument('--max_known_obstacles', default=100, type=int)
    env_args.add_argument('--max_unknown_obstacles', default=100, type=int)
    env_args.add_argument('--obstacle_radius', default=0.25, type=float)
    env_args.add_argument('--all_unknown', default=True, action=BooleanOptionalAction)
    env_args.add_argument('--max_episode_steps', default=None, type=int)
    env_args.add_argument('--max_non_new_steps', default=1000, type=int)
    env_args.add_argument('--collision_ends_episode', default=False, action=BooleanOptionalAction)
    env_args.add_argument('--flip_when_stuck', default=False, action=BooleanOptionalAction)
    env_args.add_argument('--max_stuck_steps', default=5, type=int)
    env_args.add_argument('--start_level', default=1, type=int)
    env_args.add_argument('--use_goal_time_in_levels', default=False, action=BooleanOptionalAction)
    env_args.add_argument('--goal_coverage', default=0.9, type=float)
    env_args.add_argument('--goal_coverage_reward', default=0, type=float)
    # reward 相关
    '''
    newly_visited_reward_scale：新覆盖区域奖励
    local_tv_reward_scale：局部 TV 奖励
    global_tv_reward_scale：全局 TV 奖励
    constant_reward：每步固定惩罚
    collision_reward：碰撞惩罚
    '''
    env_args.add_argument('--wall_collision_reward', default=-10, type=float)
    env_args.add_argument('--obstacle_collision_reward', default=-10, type=float)
    env_args.add_argument('--newly_visited_reward_scale', default=1, type=float)
    env_args.add_argument('--newly_visited_reward_max', default=2, type=float)
    env_args.add_argument('--overlap_reward_scale', default=0, type=float)
    env_args.add_argument('--overlap_reward_max', default=5, type=float)
    env_args.add_argument('--overlap_reward_always', default=False, action=BooleanOptionalAction)
    env_args.add_argument('--local_tv_reward_scale', default=1, type=float)
    env_args.add_argument('--local_tv_reward_max', default=5, type=float)
    env_args.add_argument('--global_tv_reward_scale', default=0, type=float)
    env_args.add_argument('--global_tv_reward_max', default=5, type=float)
    env_args.add_argument('--use_known_obstacles_in_tv', default=True, action=BooleanOptionalAction)
    env_args.add_argument('--use_unknown_obstacles_in_tv', default=True, action=BooleanOptionalAction)
    env_args.add_argument('--frontier_reward_scale', default=0, type=float)
    env_args.add_argument('--frontier_reward_max', default=5, type=float)
    env_args.add_argument('--turn_reward_scale', default=0, type=float)
    env_args.add_argument('--obstacle_dilation', default=9, type=int)
    env_args.add_argument('--constant_reward', default=-0.1, type=float)
    env_args.add_argument('--constant_reward_always', default=True, action=BooleanOptionalAction)
    env_args.add_argument('--truncation_reward_scale', default=0, type=float)
    env_args.add_argument('--coverage_pad_value', default=0, type=int)
    env_args.add_argument('--obstacle_pad_value', default=1, type=int)
    env_args.add_argument('--verbose', default=False, action=BooleanOptionalAction)
    env_args.add_argument('--metrics_dir', default=None, type=str)
    args = parser.parse_args()
    assert args.algo in ['SAC', 'PPO'], 'Only SAC/PPO algorithms currently supported'
    print(args, flush=True)

    # 把所有参数按 agent / train / env 分类存起来
    arg_groups = {}
    for group in parser._action_groups:
        group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
        arg_groups[group.title] = argparse.Namespace(**group_dict)

    # 保存参数在实验目录，如果你手动指定了 --logdir exp1，就存到 exp1/,否则自动建一个按时间命名的目录
    if args.logdir is not None:
        logdir = args.logdir
    else:
        logdir = os.path.join('experiments', datetime.now().strftime("%Y-%m-%d_%H%M%S"))
    if arg_groups['env'].metrics_dir is not None:
        arg_groups['env'].metrics_dir = os.path.join(logdir, arg_groups['env'].metrics_dir)
    os.makedirs(logdir)
    with open(os.path.join(logdir, 'agent_parameters.json'), 'w') as f:
        json.dump(arg_groups['agent'].__dict__, f, indent=2)
    with open(os.path.join(logdir, 'train_parameters.json'), 'w') as f:
        json.dump(arg_groups['train'].__dict__, f, indent=2)
    with open(os.path.join(logdir, 'env_parameters.json'), 'w') as f:
        json.dump(arg_groups['env'].__dict__, f, indent=2)

    # 如果用户开启了 --cnn
    if args.cnn:

        # 如果当前强化学习算法是 SAC
        if 'SAC' in args.algo:
            # net_arch 是 stable-baselines3 里策略网络的结构配置
            # pi = policy network（策略网络，负责输出动作）
            # qf = Q-function network（Q价值网络，负责评估动作价值）
            # [args.cnn_dims, args.cnn_dims] 表示各有两层全连接隐藏层，
            # 每层神经元个数都是 args.cnn_dims，比如默认 256
            net_arch = dict(
                pi=[args.cnn_dims, args.cnn_dims],
                qf=[args.cnn_dims, args.cnn_dims])

        # 如果当前算法是 PPO
        elif args.algo == 'PPO':
            # PPO 的网络结构写法和 SAC 不一样
            # pi = policy network（策略网络）
            # vf = value function network（状态价值网络）
            # 注意这里外面要包一层 list，这是 stable-baselines3 对 PPO 的接口要求
            net_arch = [dict(
                pi=[args.cnn_dims, args.cnn_dims],
                vf=[args.cnn_dims, args.cnn_dims])]

        # policy_kwargs 是传给 stable-baselines3 的“策略配置字典”
        # 它告诉算法：
        # 1. 后面的策略/价值网络怎么搭
        # 2. 输入先用哪个特征提取器处理
        policy_kwargs = dict(

            # 指定策略网络/价值网络的隐藏层结构
            net_arch=net_arch,

            # 指定“特征提取器”的类
            # 也就是说：环境输出的原始 observation
            # 不会直接喂给策略网络，而是先经过这个类做特征提取
            features_extractor_class=StackedMapFeaturesExtractor,

            # 这是传给上面这个特征提取器类的初始化参数
            features_extractor_kwargs=dict(

                # 提取后的特征维度，最终会压成一个 features_dim 长度的向量
                features_dim=args.cnn_dims,

                # 每张地图的边长（输入分辨率）
                map_size=args.input_size,

                # 多尺度地图的数量
                num_maps=args.num_maps,

                # 激光雷达射线数量
                lidar_rays=args.lidar_rays,

                # 堆叠多少帧观测
                stacks=args.stacks,

                # 是否使用 grouped convolution（分组卷积）
                # 这对应论文中的 scale-grouped CNN / SGCNN 思路
                grouped_convs=args.grouped_convs,

                # 是否把 frontier map（前沿地图）也作为输入
                frontier_observation=args.frontier_observation))

    # 如果没有开启 --cnn
    else:
        # 那就不给算法额外的 CNN 配置
        # 算法会退回到默认策略，一般就是 MLP 直接处理输入
        policy_kwargs = None

    # Train agent
    # 创建环境
    env = MowerEnv(**vars(arg_groups['env']))
    env = Monitor(env, os.path.join(logdir, 'logs'), info_keywords=('level',))
    algo = getattr(importlib.import_module('stable_baselines3'), args.algo)
    # 是否训练未训练完的模型
    if args.checkpoint is not None:
        # TODO: also load parameters.json from previous run
        model = algo.load(args.checkpoint, env=env)
    else:
        kwargs = dict(verbose=1, policy_kwargs=policy_kwargs)
        if args.buffer_size is not None and 'SAC' in args.algo:
            kwargs['buffer_size'] = args.buffer_size
        if args.learning_rate is not None:
            kwargs['learning_rate'] = args.learning_rate
        if 'SAC' in args.algo:
            kwargs['train_freq'] = args.train_freq
            kwargs['gradient_steps'] = args.gradient_steps
        # 调用库函数创建模型对象
        model = algo("MultiInputPolicy", env, **kwargs)
    model.learn(total_timesteps=args.steps)
    model.save(os.path.join(logdir, 'agent'))
    env.close()

if __name__ == '__main__':
    main()
