# Import simulation Env
# %%
from collections import deque  # Ordered collection with ends
from datetime import datetime  # Help us logging time
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

from rocket.images.preprocess import stack_frames
from rocket.utils.exprience import ReplayBuffer
from rocket.ignite.models import Encoder, Actor, Critic, Forward, Inverse, Policy, Agent
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

    episode_num = 1
    episodes_rewards = deque([])
    run_steps = 1

    # Launch a new episode

    state = env.reset()  # Get a new state

    states = deque([])
    rewards = deque([])
    rewards_to_gos = deque([])
    actions = deque([])
    next_states = deque([])
    logprobs = deque([])

    rewards_of_episode = 0
    state = torch.as_tensor(state).moveaxis(-1, 0)
    state = Resize((state_size[-2], state_size[-1])
                   )(state).float()
    while True:
        # Run State Through Policy & Calculate Action

        # REMEMBER THAT WE ARE IN A STOCHASTIC POLICY SO WE DON'T ALWAYS TAKE THE ACTION WITH THE HIGHEST PROBABILITY
        # (For instance if the action with the best probability for state S is a1 with 70% chances, there is
        # 30% chance that we take action a2)

        with torch.no_grad():
            prob = model.policy(
                state.to(device=device, dtype=torch.float)
                .reshape(
                    1, *state_size))

        if np.random.randn() < explore_rate:

            action = prob.sample()  # select action w.r.t the actionss prob
        else:
            action = torch.as_tensor(
                np.random.choice(action_size)).unsqueeze(0)

        # Perform action
        next_state, reward, done, _ = env.step(action)
        # env.render()
        next_state = torch.as_tensor(next_state).moveaxis(-1, 0)
        next_state = Resize(
            (state_size[-2], state_size[-1]))(next_state).float()
        rewards_of_episode = gamma * rewards_of_episode + reward
        # # Store results
        states.append(state)
        actions.append(action)
        rewards_to_gos.append(0)
        rewards.append(reward)
        next_states.append(next_state)
        logprobs.append(prob.log_prob(action))

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
            n = len(rewards_to_gos)
            for i in reversed(range(n)):
                rewards_to_gos[i] = rewards[i] + \
                    gamma * (rewards_to_gos[i + 1] if i + 1 < n else 0)

            store_episode(model, memory, list(states), list(actions), list(rewards), list(rewards_to_gos), list(next_states),
                          list(logprobs))

            # Reset the transition stores
            states = deque([])
            rewards = deque([])
            rewards_to_gos = deque([])
            actions = deque([])
            next_states = deque([])
            logprobs = deque([])
            #########
            # Episode base
            #########
            episodes_rewards.append(rewards_of_episode)
            rewards_of_episode = 0

            # Add episode
            episode_num += 1

            # Start a new episode
            state = env.reset()
            state = torch.as_tensor(state).moveaxis(-1, 0)

            state = Resize((state_size[-2], state_size[-1])
                           )(state).float()

        else:
            # If not done, the next_state become the current state
            # run_steps += 1

            state = next_state

    # plt.close()
    return episodes_rewards


def make_mini_batch(data, batch_size, mini_batch_size):
    mini_batch = []
    sample_number = int(batch_size / mini_batch_size)
    for (start_idx, end_idx) in zip(np.linspace(0, batch_size - mini_batch_size, sample_number, endpoint=True),
                                    np.linspace(mini_batch_size, batch_size, sample_number, endpoint=True)):
        mini_batch.append([mini_data[int(start_idx):int(end_idx)]
                          for mini_data in data])

    return mini_batch


def transitions_to_batch(transitions):
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))
    states_mb = batch.state
    actions_mb = batch.action
    rewards_mb = batch.reward
    reward_to_go_mb = batch.reward_to_go
    next_states_mb = batch.next_state
    logp_mb = batch.logp
    return states_mb, actions_mb, rewards_mb, reward_to_go_mb, next_states_mb, logp_mb


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


def calculate_gradient(model, optimizer, states, actions, rewards, next_states, logps, gamma):
    return 0


def train(model, memory, batch_size):
    return 0


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
    env_list = {'Breakout': 'Breakout-v0', 'DeepRacer': 'DeepRacer-v1'}
    game_env = 'Breakout'
    env = gym.make(env_list[game_env])
    env = env.unwrapped
    # %%
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
    max_episodes = 15
    min_episodes = 3  # Total episodes for sampling
    explore_rate = 0.5
    min_explore_rate = 0.05
    explore_decay_rate = 0.95
    explore_decay_step = 50
    gamma = 0.999  # Discounting rate
    min_clip_value = -1
    max_clip_value = 1
    # Starting Epoch
    epoch = 0  # tune this > 51 to reduce exploration rate
    batch_size = 128 * 2  # Each 1 is AN EPISODE
    mini_batch_size = 64

    writer = SummaryWriter('logs/DeepRacer-v1')

    config = {'state_size': state_size,
              'action_size': action_size,
              'learning_rate': learning_rate,
              'max_episodes': max_episodes,
              'min_episodes': min_episodes,
              'gamma': gamma,
              'explore_rate': explore_rate,
              'explore_decay_rate': explore_decay_rate,
              'batch_size': batch_size
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

    save_point = "./ckpt/{}+ppo.pt".format(game_env)
    if not os.path.exists(save_point):
        with open(save_point, "w+") as f:
            f.close()
    """
        # Set up neural models
        # Remember moving models to selected device (CPU/GPU) before exporting parameters to optimizer
        # initialize model weights with .apply()
        # change the model to float32
    """

    feature_size = 128
    encoder = Encoder() \
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

    model = Agent(encoder, policy_network,
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

    for p in model.parameters():
        p.register_hook(lambda grad: torch.clamp(
            grad, min_clip_value, max_clip_value))
    # restore saved model if available
    if not os.stat(save_point).st_size == 0:
        checkpoint = torch.load(save_point, map_location=device)

        agent_state_dict = checkpoint['agent_state_dict']
        model.load_state_dict(
            agent_state_dict, strict=False)
        optimizer.load_state_dict(
            checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']

    memory = ReplayBuffer(500)
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
    epoch += 1
    episodes = 15

    while training:
        print("==========================================")
        print("Epoch: ", epoch)
        # Gather training data
        model.train()
        print("==========================================")
        print("Agent start Playing")
        print("==========================================")
        episodes_rewards = make_batch(env, model, episodes, memory,
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

        # Feedforward, gradient and backpropagation

        mean_loss = []
        mean_total_reward = []
        mean_entropy = []
        mini_batch = transitions_to_batch(memory.samples(mini_batch_size))

        states_mb, actions_mb, rewards_mb, reward_to_gos_mb, next_states_mb, logps_mb = mini_batch
        mini_batch = torch.stack(states_mb), torch.stack(actions_mb).unsqueeze(-1), torch.stack(rewards_mb).unsqueeze(
            -1), torch.stack(reward_to_gos_mb).unsqueeze(-1), torch.stack(next_states_mb), torch.stack(logps_mb).unsqueeze(-1)

        _, _, rewards_mb, _, _, _ = mini_batch

        model.minimize(inputs=mini_batch, optimizer=optimizer)

        mean_loss.append(model.loss(
            mini_batch).detach().cpu().numpy())
        mean_total_reward.append(np.sum(rewards_mb.detach().cpu().numpy()))

        print("Total reward: {}".format(
            np.mean(mean_total_reward)))
        print("Training Loss: {}".format(np.mean(mean_loss)))

        # write summary to files

        # save checkpoint
        if epoch % 10 == 0:
            torch.save({
                'epoch':                                epoch,
                'agent_state_dict':           model.state_dict(),
                'optimizer_state_dict':                 optimizer.state_dict(),
            }, save_point)

            print('--------------------')
            print('Evaluation Process')
            print("Overall Rewards of the gameplay: {}".format(
                evaluate(model, env, config)))

        epoch += 1
        episodes = int(max_episodes * np.exp(-epoch *
                                             gamma / 30)) + min_episodes
        config['explore_rate'] = explore_rate * explore_decay_rate ** epoch

# %%
