# Import simulation Env
# %%
from collections import deque  # Ordered collection with ends
import datetime  # Help us logging time
import environments
import gym
from gym.envs.registration import register
import os
import matplotlib.pyplot as plt
import numpy as np  # Handle matrices
# Import training required packags
import torch  # Deep Learning library
import torch.nn as nn
import torch.nn.functional as F
from rocket import ignite, images
import torch.optim as optim
from torchvision.transforms import Resize
from rocket.ignite.types import Transition
from pathlib import Path
from rocket.utils.logging import MetricLogger

from rocket.images.preprocess import stack_frames
from rocket.utils.exprience import SequentialBuffer
from rocket.ignite.models import Encoder, Mlp, Actor, Critic, Forward, Inverse, Policy, Agent
from rocket.ignite.initializers import init_conv, init_linear
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def discount_and_normalize_rewards(episode_rewards, gamma):
    discounted_episode_rewards = np.zeros_like(episode_rewards)
    cumulative = 0.0
    for i in reversed(range(len(episode_rewards))):
        cumulative = cumulative * gamma + episode_rewards[i]
        discounted_episode_rewards[i] = cumulative

    mean = np.mean(discounted_episode_rewards)
    std = np.std(discounted_episode_rewards)
    discounted_episode_rewards = (discounted_episode_rewards - mean) / (std)

    return discounted_episode_rewards


def make_batch(env, model, episodes, memory, config):
    # Initialize lists: states, actions, rewards_of_episode, rewards_of_batch, discounted_rewards
    # states, actions, rewards_of_episode, rewards_of_batch, discounted_rewards = [], [], [], [], []

    # Reward of batch is also a trick to keep track of how many timestep we made.
    # We use to to verify at the end of each episode if > batch_size or not.

    # Keep track of how many episodes in our batch (useful when we'll need to calculate the average reward per episode)

    state_size = config['state_size']
    action_size = config['action_size']
    learning_rate = config['learning_rate']
    gamma = config['gamma']
    explore_rate = config['explore_rate']
    batch_size = config['batch_size']
    logger = config['logger']

    episode_num = 0
    episodes_rewards = deque([])
    run_steps = 1

    # Launch a new episode

    state = env.reset()  # Get a new state

    # states = deque([])
    # rewards = deque([])
    # rewards_to_gos = deque([])
    # actions = deque([])
    # next_states = deque([])
    # logprobs = deque([])

    rewards_of_episode = 0
    state = torch.tensor(state)
    # state = torch.as_tensor(state).moveaxis(-1, 0)
    # state = Resize((state_size[-2], state_size[-1])
    #                )(state).float()
    while True:
        # Run State Through Policy & Calculate Action

        # REMEMBER THAT WE ARE IN A STOCHASTIC POLICY SO WE DON'T ALWAYS TAKE THE ACTION WITH THE HIGHEST PROBABILITY
        # (For instance if the action with the best probability for state S is a1 with 70% chances, there is
        # 30% chance that we take action a2)

        with torch.no_grad():
            prob = model.policy(
                state.to(device=device, dtype=torch.float)
                # .reshape(
                #     1, *state_size)
            )

        # if np.random.randn() < explore_rate:

        action = prob.sample()  # select action w.r.t the actionss prob
        # else:
        #     action = torch.as_tensor(
        #         np.random.choice(action_size)).unsqueeze(0)

        # Perform action
        next_state, reward, done, _ = env.step(action.item())
        # env.render()
        next_state = torch.tensor(next_state)
        # next_state = torch.as_tensor(next_state).moveaxis(-1, 0)
        # next_state = Resize(
        #     (state_size[-2], state_size[-1]))(next_state).float()
        rewards_of_episode = gamma * rewards_of_episode + reward
        # # Store results
        # states.append(state)
        # actions.append(action)
        # rewards_to_gos.append(0)
        # rewards.append(reward)
        # next_states.append(next_state)
        # logprobs.append(prob.log_prob(action))
        memory.add(Transition(state, action, torch.as_tensor(reward), 0,
                   next_state, prob.log_prob(action)))

        if done:
            # The episode ends so no next state
            # next_state = np.zeros((256, 256, 3), dtype=np.int)
            # next_state, stacked_frames = stack_frames(stacked_frames, next_state, False, stack_size=stack_size)

            # Append the rewards_of_batch to reward_of_episode
            # rewards_of_batch.append(rewards_of_episode)

            # Calculate gamma Gt
            # discounted_rewards.append(discount_and_normalize_rewards(rewards_of_episode,gamma))

            # If the number of rewards_of_batch > batch_size stop the minibatch creation
            # (Because we have sufficient number of episode mb)
            # Remember that we put this condition here, because we want entire episode (Monte Carlo)
            # so we can't check that condition for each step but only if an episode is finished
            if episode_num >= episodes:
                break

            # Store Episodes into memory
            # n = len(rewards_to_gos)
            # for i in reversed(range(n)):
            #     rewards_to_gos[i] = rewards[i] + \
            #         gamma * (rewards_to_gos[i + 1] if i + 1 < n else 0)

            # store_episode(model, memory, list(states), list(actions), list(rewards), list(rewards_to_gos), list(next_states),
            #               list(logprobs))

            memory.pack_episodes()

            # Reset the transition stores
            # states = deque([])
            # rewards = deque([])
            # rewards_to_gos = deque([])
            # actions = deque([])
            # next_states = deque([])
            # logprobs = deque([])
            #########
            # Episode base
            #########
            episodes_rewards.append(rewards_of_episode)
            rewards_of_episode = 0

            # Add episode
            episode_num += 1

            # Start a new episode
            state = env.reset()
            state = torch.tensor(state)
            # state = torch.as_tensor(state).moveaxis(-1, 0)

            # state = Resize((state_size[-2], state_size[-1])
            #                )(state).float()

        else:
            # If not done, the next_state become the current state
            # run_steps += 1

            state = next_state

    # plt.close()
    return episodes_rewards


def unpack(data):
    return [*zip(*data)]


def pack(data, type):
    return type(*zip(*data))


def make_mini_batch(data, batch_size, mini_batch_size):
    mini_batch = []
    sample_number = int(batch_size / mini_batch_size)
    for (start_idx, end_idx) in zip(np.linspace(0, batch_size - mini_batch_size, sample_number, endpoint=True),
                                    np.linspace(mini_batch_size, batch_size, sample_number, endpoint=True)):
        mini_batch.append(data[int(start_idx):int(end_idx)])

    return mini_batch


def store_episode(model, memory, states, actions, rewards, rewards_to_gos, next_states, logps):
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    for s, a, r, rg, st, logp in zip(states, actions, rewards, rewards_to_gos, next_states, logps):

        s, a, r, rg, st, logp = (torch.as_tensor(s).to(dtype=torch.float),
                                 a.clone().detach().to(
                                     dtype=torch.long),
                                 torch.as_tensor(r).to(dtype=torch.float),
                                 torch.as_tensor(rg).to(dtype=torch.float),
                                 torch.as_tensor(st).to(
                                     dtype=torch.float),
                                 logp.clone().detach().requires_grad_(True).to(dtype=torch.float))
        memory.store(Transition(s, a, r, rg, st, logp
                                ), model.intrinsic_reward(s.unsqueeze(0), st.unsqueeze(0), a))


def evaluate(model, env, config, num_episodes=3):
    # This function will only work for a single Environment
    all_episode_rewards = []
    model.eval()
    state_size = config['state_size']
    for _ in range(num_episodes):
        episode_rewards = []
        done = False

        state = env.reset()

        while not done:
            state = torch.as_tensor(state).moveaxis(-1, 0)
            state = Resize((state_size[-2], state_size[-1])
                           )(state).float()
            with torch.no_grad():

                prob_distribution = model.policy(
                    state.to(device=device, dtype=torch.float)
                    .reshape(
                        1, *state_size))
                action = prob_distribution.sample()
            state, reward, done, info = env.step(action.item())

            episode_rewards.append(reward)

        all_episode_rewards.append(sum(episode_rewards))
    # plt.close()
    mean_episode_reward = np.mean(all_episode_rewards)
    return mean_episode_reward


# %%
if __name__ == "__main__":
    env_list = {'Breakout': 'Breakout-v0',
                'DeepRacer': 'DeepRacer-v1', 'CartPole': 'CartPole-v1'}
    game_env = 'CartPole'
    env = gym.make(env_list[game_env])
    env = env.unwrapped

    save_dir = Path(
        "ckpt")/datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")    # %%
    save_dir.mkdir(parents=True, exist_ok=True)
    save_point = save_dir / "{}+ppo.pt".format(game_env)
    writer = SummaryWriter(save_dir)
    ################################
    #
    #   Initialize training hyperparameters
    #
    #################################
    state_size = [3, 640, 360]
    # Our input is a stack of 4 frames hence 4x160x120x3 (stack size,Width, height, channels)
    # 10 possible actions: turn left, turn right, move forward
    action_size = env.action_space.n
    print(str(env.action_space.n))
    print(str(env.observation_space))
    possible_actions = [x for x in range(action_size)]

    # TRAINING HYPERPARAMETERS
    learning_rate = 1e-3
    max_episodes = 5
    explore_rate = 0.5
    min_explore_rate = 0.05
    explore_decay_rate = 0.95
    explore_decay_step = 50
    gamma = 0.995  # Discounting rate
    min_clip_value = -1
    max_clip_value = 1
    # Starting Epoch
    epoch = 0  # tune this > 51 to reduce exploration rate
    batch_size = 128  # Each 1 is AN EPISODE
    mini_batch_size = 64

    logger = MetricLogger(save_dir)

    config = {'state_size': state_size,
              'action_size': action_size,
              'learning_rate': learning_rate,
              'max_episodes': max_episodes,
              'gamma': gamma,
              'lambda': 0.95,
              'explore_rate': explore_rate,
              'explore_decay_rate': explore_decay_rate,
              'batch_size': batch_size,
              'logger': logger
              }

    # MODIFY THIS TO FALSE IF YOU JUST WANT TO SEE THE TRAINED AGENT
    training = True

    ##################################
    ##################################
    #
    #   End of Hyperparameters
    #
    ##################################

    ################################
    #
    #   Initialize  variables
    #
    #################################

    # Create Save checkpoint

    if not os.path.exists(save_point):
        with open(save_point, "w+") as f:
            f.close()
    """
        # Set up neural models
        # Remember moving models to selected device (CPU/GPU) before exporting parameters to optimizer
        # initialize model weights with .apply()
        # change the model to float32
    """

    feature_size = 64
    mlp = Mlp(env.observation_space.shape[0], feature_size)
    encoder = Encoder(feature_size) \
        .to(device=device) \
        .apply(init_conv)
    policy_network = Policy(feature_size) \
        .to(device=device) \
        .apply(init_linear) \
        .float()
    actor = Actor(action_size, feature_size, policy_network, policy_network.last_layer_size) \
        .to(device=device) \
        .apply(init_linear) \
        .float()
    critic = Critic(action_size, feature_size, policy_network, policy_network.last_layer_size) \
        .to(device=device) \
        .apply(init_linear) \
        .float()
    intrinsic = Forward(action_size, feature_size) \
        .to(device=device) \
        .apply(init_linear) \
        .float()
    extrinsic = Inverse(action_size, feature_size) \
        .to(device=device) \
        .apply(init_linear) \
        .float() \

    model = Agent(mlp, policy_network,
                  actor, critic, intrinsic, extrinsic) \
        .to(device=device) \
        .apply(init_linear) \
        .float() \

    if not os.stat(save_point).st_size == 0:
        checkpoint = torch.load(save_point, map_location=device)

        agent_state_dict = checkpoint['agent_state_dict']
        model.load_state_dict(
            agent_state_dict, strict=False)
        epoch = checkpoint['epoch']

    optimizer = optim.Adam([
        # {'params': policy_network.parameters()},
        # {'params': actor.parameters()},
        # {'params': critic.parameters()},
        # {'params': extrinsic.parameters()},
        # {'params': intrinsic.parameters()},
        # {'params': encoder.parameters()},
        {'params': model.parameters()},
    ],
        lr=learning_rate)

    networks = [policy_network, encoder, actor, critic, extrinsic, intrinsic]

    """
    regiter hook for gradient clamping
    """

    # restore saved model if available

    memory = SequentialBuffer(max_episodes, config['gamma'], config['lambda'])
    # episode base
    # memory = ExperienceBuffer(10)

    ################################
    #
    #   End of  variables
    #
    #################################

    #############################
    #
    # Start Training
    #
    #############################

    # Define the Keras TensorBoard callback.

    # Define Summary Metrics

    print("[INFO] START TRAINING")
epoch = 1

training = True
while training:
    print("==========================================")
    print("Epoch: ", epoch)
    # Gather training data
    model.train()
    print("==========================================")
    print("Agent start Playing")
    print("==========================================")
    episodes_rewards = make_batch(env, model, max_episodes, memory,
                                  config)

    # These part is used for analytics
    # Calculate the total reward ot the batch

    # Calculate the mean reward of the batch

    # Calculate the average reward of all training

    # Calculate maximum reward recorded
    # maximumRewardRecorded = np.amax(allRewards)

    print("-----------")
    print("Number of training episodes: {}".format(len(episodes_rewards)))
    print("Rewards over the epoch: {}".format(np.mean(episodes_rewards)))
    mean_reward = []
    mean_loss = []
    # trainning step
    # model.zero_grad()
    optimizer.zero_grad()
    # Feedforward, gradient and backpropagation
    step = 0

    eps = unpack(memory.samples())
    for mini_batch in make_mini_batch(
            eps,
            len(eps),
            min(len(eps), mini_batch_size)):

        mb = pack(mini_batch, Transition)
        print(mb)
        states_mb = torch.stack(mb.state).to(
            device=device, dtype=torch.float)
        actions_mb = torch.stack(mb.action).to(
            device=device, dtype=torch.long).unsqueeze(-1)
        rewards_mb = torch.stack(mb.reward).to(
            device=device, dtype=torch.float).unsqueeze(-1)
        reward_to_gos_mb = torch.stack(mb.reward_to_go).to(
            device=device, dtype=torch.float).unsqueeze(-1)
        next_states_mb = torch.stack(mb.next_state).to(
            device=device, dtype=torch.float)
        logps_mb = torch.stack(mb.logp).to(
            device=device, dtype=torch.float).unsqueeze(-1)

        # states_mb = torch.stack(mb.state).pin_memory().to(
        #     device=device, non_blocking=True)
        # actions_mb = torch.stack(mb.action).pin_memory().to(
        #     device=device, non_blocking=True).unsqueeze(-1)
        # rewards_mb = torch.stack(mb.reward).pin_memory().to(
        #     device=device, non_blocking=True).unsqueeze(-1)
        # reward_to_gos_mb = torch.stack(mb.reward_to_go).pin_memory().to(
        #     device=device, non_blocking=True).unsqueeze(-1)
        # next_states_mb = torch.stack(mb.next_state).pin_memory().to(
        #     device=device, non_blocking=True)
        # logps_mb = torch.stack(mb.logp).pin_memory().to(
        #     device=device, non_blocking=True).unsqueeze(-1)

        small_batch = (states_mb, actions_mb, rewards_mb,
                       reward_to_gos_mb, next_states_mb, logps_mb)

        loss = model.loss(small_batch)
        loss.backward()
        for p in model.parameters():
            torch.nn.utils.clip_grad_norm_(
                p, 2.0)
        mean_reward.append(np.sum(rewards_mb.detach().cpu().numpy()))
        mean_loss.append(loss.detach().cpu().numpy())

    optimizer.step()
    optimizer.zero_grad()

    print("Total reward: {}".format(np.mean(mean_reward)))
    print("Training Loss: {}".format(np.mean(mean_loss)))

    writer.add_scalar('loss', np.mean(mean_loss), epoch)
    writer.add_scalar('reward', np.mean(mean_reward), epoch)

    # write summary to files

    # save checkpoint
    if epoch % 10 == 0:
        torch.save({
            'epoch':                                epoch,
            'agent_state_dict':           model.state_dict(),
            'optimizer_state_dict':                 optimizer.state_dict(),
        }, save_point)

    epoch += 1
    config['explore_rate'] = explore_rate * explore_decay_rate ** epoch

# %%
