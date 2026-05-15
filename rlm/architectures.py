import gym
import torch as th
from torch import nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# 自己写的一个CNN作为类型提取器
# 把 MowerEnv 输出的复杂观测（多尺度地图、lidar、可选历史动作）变成一个固定长度的特征向量
class StackedMapFeaturesExtractor(BaseFeaturesExtractor):
    # 初始化CNN
    def __init__(
            self,
            observation_space: gym.spaces.Dict,
            features_dim, # 输出的特征向量长度
            map_size,
            num_maps,
            lidar_rays,
            stacks,
            grouped_convs,
            frontier_observation):
        super(StackedMapFeaturesExtractor, self).__init__(observation_space, features_dim=features_dim)

        num_map_observations = 2
        if frontier_observation:
            num_map_observations = 3

        in_channels = num_map_observations * stacks * num_maps # 输入通道数
        out_channels = 2 * in_channels
        out_size = (map_size // 2 - 2 - 2 - 2)**2 * out_channels #卷积做完以后，展平前有多少个数（可深入）

        # 是否进行分组卷积
        if grouped_convs:
            self.map_extractor = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0, groups=num_maps),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=0, groups=num_maps),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=0, groups=num_maps),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=0, groups=num_maps),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(out_size, features_dim),
                nn.ReLU()
            )
        else:
            self.map_extractor = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(out_size, features_dim),
                nn.ReLU()
            )

        # 用MLP将lidar从二维展平成一维
        self.lidar_extractor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(stacks * lidar_rays, lidar_rays),
            nn.ReLU()
        )
        # 如果有历史动作，直接展平作为额外特征
        if 'action' in observation_space.keys():
            self.action_extractor = nn.Sequential(
                nn.Flatten()
            )
            action_obs_shape = observation_space['action'].shape
            action_dim = action_obs_shape[0] * action_obs_shape[1]
        else:
            action_dim = 0
        # 拼接向量并压缩
        self.fused_extractor = nn.Sequential(
            nn.Linear(features_dim + lidar_rays + action_dim, features_dim),
            nn.ReLU()
        )
    # 向前传播
    def forward(self, observations) -> th.Tensor:
        use_frontier = 'frontier' in observations.keys()
        use_action = 'action' in observations.keys()

        # Observations提取数据
        lidar = observations['lidar']           # batch x stacks x lidar_rays
        coverage = observations['coverage']     # batch x stacks x nmaps x W x H
        obstacles = observations['obstacles']   # batch x stacks x nmaps x W x H
        if use_frontier:
            frontier = observations['frontier'] # batch x stacks x nmaps x W x H
        if use_action:
            action = observations['action']     # batch x nactions x (1 or 2)

        # Map features
        if use_frontier:
            maps = th.cat([coverage, obstacles, frontier], dim=1) # batch x 3*stacks x nmaps x W x H
        else:
            maps = th.cat([coverage, obstacles], dim=1) # batch x 2*stacks x nmaps x W x H
        maps = maps.permute(0, 2, 1, 3, 4).contiguous() # batch x nmaps x (2 or 3)*stacks x W x H (for grouping correctly)
        b, _, _, w, h = maps.shape
        maps = maps.reshape(b, -1, w, h)                # batch x nmaps*(2 or 3)*stacks x W x H
        map_features = self.map_extractor(maps)         # batch x features_dim

        # Sensor features
        lidar_features = self.lidar_extractor(lidar)    # batch x lidar_rays

        # Action features
        if use_action:
            action_features = self.action_extractor(action) # batch x nactions*(1 or 2)

        # Fused features：所有特征融合
        if use_action:
            features = th.cat([map_features, lidar_features, action_features], dim=1)
        else:
            features = th.cat([map_features, lidar_features], dim=1)
        features = self.fused_extractor(features)
        return features
