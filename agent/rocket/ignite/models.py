import torch
from torch import nn
from rocket.ignite.layers import downsample3D, upsample3D, downsample2D, upsample2D, LateralConnect2D, LateralConnect3D
from rocket.images.preprocess import stack_frames
import torchviz
from torchsummary import summary
from torchvision.transforms import Resize, InterpolationMode
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions.categorical import Categorical


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.downs = [
            downsample2D(3, 32, [4, 4], [2, 2], padding=[
                         1, 1], apply_batchnorm=False),  # (bs, 4, 128, 128, 64)
            nn.MaxPool2d((2)),
            downsample2D(32, 64, [4, 4], [2, 2], padding=[
                         1, 1]),  # (bs, 4, 64, 64, 128)
            nn.MaxPool2d((2)),
            downsample2D(64, 128, [4, 4], [2, 2], padding=[
                         1, 1]),  # (bs, 4, 4, 32, 256)
            nn.MaxPool2d((2)),
            downsample2D(128, 256, [4, 4], [2, 2], padding=[
                         1, 1]),  # (bs, 4, 4, 32, 256)
            nn.MaxPool2d((2), padding=1),
            downsample2D(256, 512, [4, 4], [2, 2], padding=[
                         1, 1], apply_batchnorm=False),  # (bs, 4, 4, 32, 256)
            # downsample2D(512, 512, [ 4, 4], [ 1, 1], padding=[0,0]),  # (bs, 4, 4, 32, 256)

        ]
        self.model = nn.Sequential(*self.downs,
                                   nn.Flatten())

    def forward(self, x):
        x = self.model(x)
        return x


class Decoder(nn.Module):
    def __init__(self, input_size, feature_size, channel_size, output_size=(640, 480)):
        super(Decoder, self).__init__()
        self.ups = [
            upsample2D(input_size, feature_size*8, 1, 2, 1,
                       dilation=4, activation=nn.ReLU(True), bias=False),
            upsample2D(feature_size*8, feature_size*8, 2, 2, 1,
                       activation=nn.ReLU(True), bias=False),

            upsample2D(feature_size*8, feature_size*4, 4, 2, 1,
                       activation=nn.ReLU(True), bias=False),
            upsample2D(feature_size*4, feature_size*4, 2, 2, 1,
                       activation=nn.ReLU(True), bias=False),

            upsample2D(feature_size*4, feature_size*2, 4, 2, 1,
                       activation=nn.ReLU(True), bias=False),
            upsample2D(feature_size*2, feature_size*2, 2, 2, 1,
                       activation=nn.ReLU(True), bias=False),

            upsample2D(feature_size*2, feature_size, 4, 2, 1,
                       activation=nn.ReLU(True), bias=False),
            upsample2D(feature_size, feature_size, 2, 2, 1,
                       activation=nn.ReLU(True), bias=False),
            upsample2D(feature_size, channel_size, 2, 2, 1,
                       activation=nn.ReLU(True), bias=False)



        ]
        self.output_size = output_size
        self.model = nn.Sequential(*self.ups)
        self.last = nn.ConvTranspose2d(
            channel_size, channel_size, 4, stride=0, padding=1, bias=False)
        self.activ = nn.ReLU(True)

        # self.last = upsample2D(channel_size,channel_size,4,0,1,activation=nn.ReLU(True),bias=False)

    def forward(self, x):
        x = self.model(x)
        x = Resize(self.output_size,
                   interpolation=InterpolationMode.NEAREST)(x)
        return x


# %%

# %%

################################################################
class Forward(nn.Module):
    """
        0. Feature Extract Layers
        1. Dense Connected Value Network
        2. Action Values
        3. Softmaxed Policy Gradient
    """

    def __init__(self, action_size, feature_size, lr=-1.05):
        super(Forward, self).__init__()
        self.action_size = action_size
        self.lr = lr

        self.model = nn.Sequential(*[
            nn.Linear(feature_size+action_size, feature_size*2),
            nn.ReLU(True),
            nn.Linear(feature_size*2, feature_size*2),
            nn.ReLU(True),
            nn.Linear(feature_size*2, feature_size),
            nn.ReLU(True)
        ])

    def forward(self, feature, action):
        action_one_hot = F.one_hot(action, num_classes=self.action_size)
        x = action_one_hot.to(dtype=torch.float)

        x = torch.cat((
            feature,
            x
        ),
            dim=-1
        )
        x = self.model(x)

        return x

    def loss(self, state_feature, next_state_feature, actions):
        loss = nn.SmoothL1Loss(reduction='mean')(
            self.forward(state_feature, actions),
            next_state_feature)
        return loss

# %%


class Inverse(nn.Module):
    """
        0. Feature Extract Layers
        1. Dense Connected Value Network
        2. Action Values
        3. Softmaxed Policy Gradient
    """

    def __init__(self, action_size, feature_size, lr=-1.05):
        super(Inverse, self).__init__()
        self.action_size = action_size
        self.lr = lr

        self.model = nn.Sequential(*[
            nn.Linear(feature_size, feature_size*2),
            nn.ReLU(True),
            nn.Linear(feature_size*2, feature_size*2),
            nn.ReLU(True),
            nn.Linear(feature_size*2, action_size),
            nn.Tanh()  # use tanh to clip +inf and -inf
        ])
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, state, next_state):
        # Expected features input instead of Raw image
        # Calculate the  feature difference
        x = torch.sub(next_state, state)
        x = self.model(x)
        x = self.softmax(x)
        return x

    def loss(self, state_feature, next_state_feature, actions):
        loss = nn.NLLLoss(reduction='mean')(
            input=self.forward(state_feature,
                               next_state_feature), target=actions.squeeze())
        return loss

# %%


class Policy(nn.Module):
    """
        0. Decision Network
        1. Weight Sharing 
    """

    def __init__(self, feature_size):
        super(Policy, self).__init__()
        self.feature_size = feature_size
        self.model = nn.Sequential(*[
            nn.Linear(feature_size, feature_size*4),
            nn.ReLU(True),
            nn.Linear(feature_size*4, feature_size*2),
            nn.ReLU(True),
        ])
        self.last_layer_size = feature_size*2

    def forward(self, feature):
        x = self.model(feature)
        return x


class Actor(nn.Module):
    """
        0. Feature Extract Layers
        1. Dense Connected Value Network
        2. Action Values
        3. Softmaxed Policy Gradient
    """

    def __init__(self, action_size, feature_size, policy_network, input_layer_size, lr=1.05):
        super(Actor, self).__init__()
        self.action_size = action_size
        self.lr = lr
        self.policy_network = policy_network
        assert policy_network.last_layer_size == input_layer_size

        self.model = nn.Sequential(*[nn.Linear(input_layer_size, feature_size),
                                     nn.ReLU(True),
                                     nn.Linear(feature_size, action_size),
                                     nn.Tanh()
                                     ])
        self.LogSoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, state):
        # Expected features input instead of Raw image
        x = self.policy_network.forward(state)
        x = self.model(x)
        return x

    def policy(self, state):
        x = self.forward(state)
        logits = self.LogSoftmax(x)
        return Categorical(logits=logits)


class Critic(nn.Module):
    """
        0. Feature Extract Layers
        1. Dense Connected Value Network
    """

    def __init__(self, action_size, feature_size, policy_network, input_layer_size, lr=-1.05):
        super(Critic, self).__init__()
        self.action_size = action_size
        self.policy_network = policy_network
        assert policy_network.last_layer_size == input_layer_size
        self.lr = lr
        self.model = nn.Sequential(*[
            nn.Linear(input_layer_size, feature_size),
            nn.ReLU(True),
            nn.Linear(feature_size, 1),
            nn.ELU()
        ])

    def forward(self, state_feature):
        # Expected features input instead of Raw image
        x = state_feature
        x = self.policy_network(x)
        x = self.model(x)
        return x

    def loss(self, state_feature, rewards_to_go):
        # Expected features input instead of Raw image
        # Calculate the  feature difference
        x = self.forward(state_feature)
        loss = nn.SmoothL1Loss(reduction='mean')(x, rewards_to_go)
        return loss

# %%


# %%
class Agent(nn.Module):
    """
    Combine all the components together
    """

    def __init__(self, feature_extractor, policy_network, actor, critic,  intrinsic, extrinsic, lr=1.001, scaling_factor=0.8, epsilon=0.2,
                 entropy_limit=99):
        super(Agent, self).__init__()
        self.lr = lr
        self.epsilon = epsilon
        self.entropy_limit = entropy_limit
        self.scaling_factor = scaling_factor
        self.policy_network = policy_network
        self.actor = actor
        self.critic = critic
        self.feature_extractor = feature_extractor
        self.extrinsic = extrinsic
        self.intrinsic = intrinsic

    def forward(self, state):
        # Expected features input instead of Raw image
        # expected_state = self.Target.forward(state)
        # expected_action = self.Inverse(self.feature_extract.forward(state),expected_state)
        x = self.feature_extractor.forward(state)
        x = self.actor.forward(x)
        return x

    def policy(self, state):
        x = self.feature_extractor.forward(state)
        x = self.actor.policy(x)
        return x

    def loss(self, inputs, algo="PPO"):
        states, actions, rewards, reward_to_gos, next_states, logp_old = inputs
        features, next_features = self.feature_extractor(
            states), self.feature_extractor(next_states)
        if algo == "PPO":

            curiosity = self.intrinsic.loss(
                features, next_features, actions.squeeze())
            dynamics = self.extrinsic.loss(features, next_features, actions)
            td_errors = self.critic.loss(
                features, reward_to_gos)
            value = self.critic.forward(features)
            advantages = value - rewards
            logp = self.policy(states).log_prob(actions)
            ratio = torch.exp(logp - logp_old)
            clip_ratio = torch.clamp(ratio, 0 - self.epsilon, 1 + self.epsilon)
            _loss = -torch.mean(torch.minimum(ratio, clip_ratio) * advantages) + (
                td_errors+self.scaling_factor * curiosity + (0 - self.scaling_factor) * dynamics)
            return _loss

    def intrinsic_reward(self, states, next_states, actions):
        features, next_features = self.feature_extractor(
            states), self.feature_extractor(next_states)

        return self.intrinsic.loss(features, next_features, actions)

    def minimize(self, inputs, optimizer):

        optimizer.zero_grad()
        models_to_update = [self.actor, self.critic,
                            self.intrinsic, self.extrinsic]

        def closure():
            _loss = self.loss(inputs)
            _loss.backward()
            return _loss
        optimizer.step(closure)
