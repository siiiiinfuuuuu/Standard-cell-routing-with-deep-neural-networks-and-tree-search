import gym
import torch as th
from torch import distributions
import torch.nn as nn
from torch.distributions import Categorical
import os

from typing import Callable, Dict, List, Optional, Tuple, Type, Union
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    make_proba_distribution,
)
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.preprocessing import get_action_dim, is_image_space, maybe_transpose, preprocess_obs
from stable_baselines3.common.utils import get_device, is_vectorized_observation, obs_as_tensor
# from stable_baselines3.common.vec_env import DummyVecEnv
from functools import partial


from env import *

class ResNet(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, inputs):
        return self.module(inputs) + inputs

class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        *args,
        **kwargs,
    ):

        super(CustomActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
        # Disable orthogonal initialization
        self.ortho_init = False

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(self.features_dim)

class CustomNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the feature extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
    ):
        super(CustomNetwork, self).__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_pi), nn.ReLU()
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_vf), nn.ReLU()
        )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.policy_net(features), self.value_net(features)

class ResNetFeaturesExtractor(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256, num_filter: int = 128):
        print("ResNetFeaturesExtractor=======================")
        super(ResNetFeaturesExtractor, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, num_filter, kernel_size=3, stride=1, padding= 'same'),
            nn.ReLU(),
            nn.BatchNorm2d(num_filter),
            # 32 filters in and out, no max pooling so the shapes can be added
            ResNet(
                nn.Sequential(
                    nn.Conv2d(num_filter, num_filter, kernel_size=3, stride=1, padding= 'same'),
                    nn.ReLU(),
                    nn.BatchNorm2d(num_filter),
                    nn.Conv2d(num_filter, num_filter, kernel_size=3, stride=1, padding= 'same'),
                    nn.ReLU(),
                    nn.BatchNorm2d(num_filter),
                )
            ),
            ResNet(
                nn.Sequential(
                    nn.Conv2d(num_filter, num_filter, kernel_size=3, stride=1, padding= 'same'),
                    nn.ReLU(),
                    nn.BatchNorm2d(num_filter),
                    nn.Conv2d(num_filter, num_filter, kernel_size=3, stride=1, padding= 'same'),
                    nn.ReLU(),
                    nn.BatchNorm2d(num_filter),
                )
            ),
            ResNet(
                nn.Sequential(
                    nn.Conv2d(num_filter, num_filter, kernel_size=3, stride=1, padding= 'same'),
                    nn.ReLU(),
                    nn.BatchNorm2d(num_filter),
                    nn.Conv2d(num_filter, num_filter, kernel_size=3, stride=1, padding= 'same'),
                    nn.ReLU(),
                    nn.BatchNorm2d(num_filter),
                )
            ),
            nn.Flatten(),
        )
        # # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))

class VariableInput_ResNetFeaturesExtractor(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256, num_filter: int = 64):
        print("ResNetFeaturesExtractor=======================")
        super(VariableInput_ResNetFeaturesExtractor, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        self.observation_space = observation_space
        n_input_channels = self.observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, num_filter, kernel_size=3, stride=1, padding= 'same'),
            nn.ReLU(),
            nn.BatchNorm2d(num_filter),
            # 32 filters in and out, no max pooling so the shapes can be added
            ResNet(
                nn.Sequential(
                    nn.Conv2d(num_filter, num_filter, kernel_size=3, stride=1, padding= 'same'),
                    nn.ReLU(),
                    nn.BatchNorm2d(num_filter),
                    nn.Conv2d(num_filter, num_filter, kernel_size=3, stride=1, padding= 'same'),
                    nn.ReLU(),
                    nn.BatchNorm2d(num_filter),
                )
            ),
            ResNet(
                nn.Sequential(
                    nn.Conv2d(num_filter, num_filter, kernel_size=3, stride=1, padding= 'same'),
                    nn.ReLU(),
                    nn.BatchNorm2d(num_filter),
                    nn.Conv2d(num_filter, num_filter, kernel_size=3, stride=1, padding= 'same'),
                    nn.ReLU(),
                    nn.BatchNorm2d(num_filter),
                )
            ),
            # nn.Flatten(),
        )
        # # Compute shape by doing one forward pass
        # with th.no_grad():
        #     n_flatten = self.cnn(
        #         th.as_tensor(observation_space.sample()[None]).float()
        #     ).shape[1]
        #     # n_flatten = observation_space.shape[1] * observation_space.shape[2]
        # self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # print("FeaturesExtractor forward")
        # print("obs size",observations.size())
        # print(observations[0][1])
        # print(self.cnn(observations).size())
        # return self.linear(self.cnn(observations) * observations[0][1])
        legal_action = th.reshape(observations[:,1,:,:], (-1, 1, self.observation_space.shape[1], self.observation_space.shape[2]))
        feature = th.cat((self.cnn(observations), legal_action), 1)
        # print(feature.size())
        return feature
        # return self.cnn(observations)

class VariableInput_Network(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the feature extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """
    
    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = -1,
        last_layer_dim_vf: int = 32,
    ):
        super(VariableInput_Network, self).__init__()
        print("CustomNetwork======================")
        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        num_in_channel = 64
        # Policy network
        self.policy_net = nn.Sequential(
            nn.Conv2d(num_in_channel, 2, kernel_size=1, stride=1, padding= 'same'),
            nn.ReLU(),
            nn.BatchNorm2d(2),
            nn.Conv2d(2, 1, kernel_size=1, stride=1, padding= 'same'),
            nn.Flatten(),
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Conv2d(num_in_channel, last_layer_dim_vf, kernel_size=1, stride=1, padding= 'same'),
            nn.ReLU(),
            nn.BatchNorm2d(last_layer_dim_vf),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
    def masked(self, x, mask):
        x_masked = x.clone()
        x_masked[mask == 0] = -float("inf")
        return x_masked

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        # print("pn forward")
        # th.reshape(observations[0][1], (1, 1, self.observation_space.shape[1], self.observation_space.shape[2]))
        legal_action = th.reshape(features[:,-1,:,:], (-1, features.size()[2] * features.size()[3]))
        # print("feature size = ", features.size())
        # print("legal_action", features[:,-1,:,:])
        # print("policy_net output size",self.policy_net(features[:,:-1]).size())
        # print("legal_action size =", legal_action.size())
        # print("policy before =", self.policy_net(features[:,:-1]))
        # print("policy  = ", self.masked(self.policy_net(features[:,:-1]), legal_action).size())
        return self.masked(self.policy_net(features[:,:-1]), legal_action), self.value_net(features[:,:-1])

class VariableInput_ActorCriticPolicy(CustomActorCriticPolicy):   
    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = VariableInput_Network(self.features_dim)
    
    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        latent_pi, latent_vf, latent_sde = self._get_latent(obs)
        # print("latent_pi", latent_pi)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        # distribution = Categorical(nn.functional.softmax(latent_pi, dim=1))
        # print("distribution", distribution.probs)
        # actions = distribution.sample()
        # print("action", actions)
        # log_prob = distribution.log_prob(actions)
        distribution = self._get_action_dist_from_latent(latent_pi, latent_sde=latent_sde)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Get the policy action and state from an observation (and optional state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param state: The last states (can be None, used in recurrent policies)
        :param mask: The last masks (can be None, used in recurrent policies)
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next state
            (used in recurrent policies)
        """
        # TODO (GH/1): add support for RNN policies
        # if state is None:
        #     state = self.initial_state
        # if mask is None:
        #     mask = [False for _ in range(self.n_envs)]
        # print("test_predict")
        vectorized_env = False
        if isinstance(observation, dict):
            # need to copy the dict as the dict in VecFrameStack will become a torch tensor
            observation = copy.deepcopy(observation)
            for key, obs in observation.items():
                obs_space = self.observation_space.spaces[key]
                if is_image_space(obs_space):
                    obs_ = maybe_transpose(obs, obs_space)
                else:
                    obs_ = np.array(obs)
                vectorized_env = vectorized_env or is_vectorized_observation(obs_, obs_space)
                # Add batch dimension if needed
                observation[key] = obs_.reshape((-1,) + self.observation_space[key].shape)

        elif is_image_space(self.observation_space):
            # Handle the different cases for images
            # as PyTorch use channel first format
            observation = maybe_transpose(observation, self.observation_space)

        else:
            observation = np.array(observation)

        if not isinstance(observation, dict):
            # Dict obs need to be handled separately
            vectorized_env = is_vectorized_observation(observation, self.observation_space)
            # Add batch dimension if needed
            observation = observation.reshape((-1,) + self.observation_space.shape)

        observation = obs_as_tensor(observation, self.device)

        with th.no_grad():
            actions = self._predict(observation, deterministic=deterministic)
        # Convert to numpy
        actions = actions.cpu().numpy()

        if isinstance(self.action_space, gym.spaces.Box):
            if self.squash_output:
                # Rescale to proper domain when using squashing
                actions = self.unscale_action(actions)
            else:
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                actions = np.clip(actions, self.action_space.low, self.action_space.high)

        if not vectorized_env:
            if state is not None:
                raise ValueError("Error: The environment must be vectorized when using recurrent policies.")
            actions = actions[0]
        return actions, state

    def get_latent(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        return self._get_latent(obs)

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self._build_mlp_extractor()

        latent_dim_pi = self.mlp_extractor.latent_dim_pi

        # Separate features extractor for gSDE
        if self.sde_net_arch is not None:
            self.sde_features_extractor, latent_sde_dim = create_sde_features_extractor(
                self.features_dim, self.sde_net_arch, self.activation_fn
            )

        # Not to use nn.Linear on action in order to handle variable input size 
        self.action_net = nn.Linear(10, 10)
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # TODO: check for features_extractor
            # Values from stable-baselines.
            # features_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
    
    def _get_action_dist_from_latent(self, latent_pi: th.Tensor, latent_sde: Optional[th.Tensor] = None) -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :param latent_sde: Latent code for the gSDE exploration function
        :return: Action distribution
        """
        # Not to use nn.Linear on action in order to handle variable input size 
        mean_actions = latent_pi
        # print("latent_pi = ", latent_pi)
        # print(self.action_dist.proba_distribution(action_logits=mean_actions))
        return self.action_dist.proba_distribution(action_logits=mean_actions)


def create_sde_features_extractor(
    features_dim: int, sde_net_arch: List[int], activation_fn: Type[nn.Module]
) -> Tuple[nn.Sequential, int]:
    """
    Create the neural network that will be used to extract features
    for the gSDE exploration function.

    :param features_dim:
    :param sde_net_arch:
    :param activation_fn:
    :return:
    """
    # Special case: when using states as features (i.e. sde_net_arch is an empty list)
    # don't use any activation function
    sde_activation = activation_fn if len(sde_net_arch) > 0 else None
    latent_sde_net = create_mlp(features_dim, -1, sde_net_arch, activation_fn=sde_activation, squash_output=False)
    latent_sde_dim = sde_net_arch[-1] if len(sde_net_arch) > 0 else features_dim
    sde_features_extractor = nn.Sequential(*latent_sde_net)
    return sde_features_extractor, latent_sde_dim

class CustomPPO(PPO):
    def network(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        obs = obs.reshape((-1,) + obs.shape)
        obs = obs_as_tensor(obs, self.device)
        print("net obs size",obs.shape)
        # features = self.policy.features_extractor(obs)
        # latent_pi, latent_vf = self.policy.mlp_extractor(features)
        latent_pi, latent_vf, latent_sde = self.policy.get_latent(obs)
        values = self.policy.value_net(latent_vf)
        # print(latent_pi)
        distribution = nn.functional.softmax(latent_pi, dim=1)
        return distribution, values

if __name__ == '__main__':

    print("Generate layouts ......", end=" ")
    layouts_train = [Layout(LAYOUT_SIZE) for _ in range(NUM_LAYOUT_TRAIN)]
    # layouts_test = [Layout(LAYOUT_SIZE) for _ in range(NUM_LAYOUT_TEST)]
    env = RoutingEnv(layouts_train)
    # eval_env = EvalEnv(layouts_train)
    print("Finish!")


    policy_kwargs = dict(features_extractor_class = ResNetFeaturesExtractor)
    # train
    # print("training ...")

    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    # # # # model = A2C(CustomActorCriticPolicy, env, verbose=1, policy_kwargs = policy_kwargs, tensorboard_log="tensorboard")
    # model = PPO(CustomActorCriticPolicy, env, batch_size = 512, verbose=1, policy_kwargs = dict(features_extractor_class = ResNetFeaturesExtractor), tensorboard_log="tensorboard")
    # model = A2C(VariableInput_ActorCriticPolicy, env, verbose=1, policy_kwargs = dict(features_extractor_class = VariableInput_ResNetFeaturesExtractor), tensorboard_log="tensorboard")
    # model = CustomPPO(VariableInput_ActorCriticPolicy, env, batch_size=1024, verbose=1, ent_coef= 0.01, policy_kwargs = dict(features_extractor_class = VariableInput_ResNetFeaturesExtractor), tensorboard_log="tensorboard")
    # model.learn(1000000*3)
    # file_name = "PPO_15x15x_1"
    # model.save(file_name)
    # print(file_name)

    # model.save("A2C_10x10_1")
    
    # del model # remove to demonstrate saving and loading

    
    # os.environ["CUDA_VISIBLE_DEVICES"]="1"
    # model = DQN("CnnPolicy", env, verbose=1, policy_kwargs = policy_kwargs, tensorboard_log="tensorboard", exploration_fraction= 0.1, gamma=1)
    # model.learn(total_timesteps=1000000*0.3, log_interval=100)
    # model.save("dqn_10x10")

    
    print("testing")
    # model = DQN.load("dqn_8x8")
    # env.reset()
    model = CustomPPO.load("PPO_15x15_0")
    total_return = 0
    for _ in range(10):
        obs = env.state.envState()
        step = 0
        env_return = 0
        while True:
            # env.render()
            action, _states = model.predict(obs, deterministic=True)
            print("obs shape", obs.shape)
            obs = th.rand((3,4,15,15))
            obs = obs.reshape((-1,4,15,15))
            distribution, value = model.network(obs)
            # print("action = ",action)
            # print("distribution", distribution.data)
            # print("value", value)
            obs, reward, done, info = env.step(action)
            env_return += reward
            step += 1
            if done:
                env.render()
                total_return += env_return
                # print("step = ", step, "env_return = ", env_return)
                env.reset()
                break
    print("avg_return = ", total_return/10)