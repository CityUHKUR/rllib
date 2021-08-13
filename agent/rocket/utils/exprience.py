from collections import deque
from heapq import heappush, heapreplace, nlargest, heapify
import itertools
import torch
import scipy.signal
from dataclasses import dataclass, field
from typing import Any
# import tensorflow as tf  # Deep Learning
import numpy as np
from functools import reduce
from rocket.ignite.types import Transition


class SumTree(object):
    """
    This SumTree code is modified version of Morvan Zhou:
    https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/5.2_Prioritized_Replay_DQN/RL_brain.py
    """
    data_pointer = 0

    """
    Here we initialize the tree with all nodes = 0, and initialize the data with all values = 0
    """

    def __init__(self, capacity):
        # Number of leaf nodes (final nodes) that contains experiences
        self.capacity = capacity

        # Generate the tree with all nodes values = 0
        # To understand this calculation (2 * capacity - 1) look at the schema above
        # Remember we are in a binary node (each node has max 2 children) so 2x kernel_size of leaf (capacity) - 1 (root node)
        # Parent nodes = capacity - 1
        # Leaf nodes = capacity
        self.tree = np.zeros(2 * capacity - 1)

        """ tree:
            0
           / \
          0   0
         / \ / \
        0  0 0  0  [Size: capacity] it's at this line that there is the priorities score (aka pi)
        """

        # Contains the experiences (so the kernel_size of data is capacity)
        self.data = np.zeros(capacity, dtype=object)

    """
    Here we add our priority score in the sumtree leaf and add the experience in data
    """

    def add(self, priority, data):
        # Look at what index we want to put the experience
        tree_index = self.data_pointer + self.capacity - 1

        """ 
        tree:
           0
          / \
         0   0
        / \ / \
        tree_index  0 0  0  We fill the leaves from left to right
        """

        # Update data frame
        self.data[self.data_pointer] = data

        # Update the leaf
        self.update(tree_index, priority)

        # Add 1 to data_pointer
        self.data_pointer += 1

        # If we're above the capacity, you go back to first index (we overwrite)
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

    """
    Update the leaf priority score and propagate the change through tree
    """

    def update(self, tree_index, priority):
        # Change = new priority score - former priority score
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority

        # then propagate the change through tree
        while tree_index != 0:  # this method is faster than the recursive loop in the reference code

            """
            Here we want to access the line above
            THE NUMBERS IN THIS TREE ARE THE INDEXES NOT THE PRIORITY VALUES

                0
               / \
              1   2
             / \ / \
            3  4 5  [6] 

            If we are in leaf at index 6, we updated the priority score
            We need then to update index 2 node
            So tree_index = (tree_index - 1) // 2
            tree_index = (6-1)//2
            tree_index = 2 (because // round the result)
            """
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    """
    Here we get the leaf_index, priority value of that leaf and experience associated with that index
    """

    def get_leaf(self, v):
        """
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for experiences
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_index = 0

        while True:  # the while loop is faster than the method in the reference code
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1

            # If we reach bottom, end the search
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break

            else:  # downward search, always search for a higher priority node

                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index

                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index

        data_index = leaf_index - self.capacity + 1

        return leaf_index, self.tree[leaf_index], self.data[data_index]

    @property
    def total_priority(self):
        return self.tree[0]  # Returns the root node


class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree
    """
    This SumTree code is modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    PER_e = 0.01  # Hyperparameter that we use to avoid some experiences to have 0 probability of being taken
    PER_a = 0.6  # Hyperparameter that we use to make a tradeoff between taking only exp with high priority and sampling randomly
    PER_b = 0.4  # importance-sampling, from initial value increasing to 1

    PER_b_increment_per_sampling = 0.001

    absolute_error_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        # Making the tree
        """
        Remember that our tree is composed of a sum tree that contains the priority scores at his leaf
        And also a data array
        We don't use deque because it means that at each timestep our experiences change index by one.
        We prefer to use a simple array and to overwrite when the memory is full.
        """
        self.tree = SumTree(capacity)

    """
    Store a new experience in our tree
    Each new experience have a score of max_prority (it will be then improved when we use this exp to train our DDQN)
    """

    def store(self, experience):
        # Find the max priority
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])

        # If the max priority = 0 we can't put priority = 0 since this exp will never have a chance to be selected
        # So we use a minimum priority
        if max_priority == 0:
            max_priority = self.absolute_error_upper

        self.tree.add(max_priority, experience)  # set the max p for new p

    """
    - First, to sample a minibatch of k kernel_size, the range [0, priority_total] is / into k ranges.
    - Then a value is uniformly sampled from each range
    - We search in the sumtree, the experience where priority score correspond to sample values are retrieved from.
    - Then, we calculate IS weights for each minibatch element
    """

    def sample(self, n):
        # Create a sample array that will contains the minibatch
        memory_b = []

        b_idx, b_ISWeights = np.empty(
            (n,), dtype=np.int32), np.empty((n, 1), dtype=np.float32)

        # Calculate the priority segment
        # Here, as explained in the paper, we divide the Range[0, ptotal] into n ranges
        priority_segment = self.tree.total_priority / n  # priority segment

        # Here we increasing the PER_b each time we sample a new minibatch
        self.PER_b = np.min(
            [1., self.PER_b + self.PER_b_increment_per_sampling])  # max = 1

        # Calculating the max_weight
        p_min = np.min(
            self.tree.tree[-self.tree.capacity:]) / self.tree.total_priority
        max_weight = (p_min * n) ** (-self.PER_b)

        for i in range(n):
            """
            A value is uniformly sample from each range
            """
            a, b = priority_segment * i, priority_segment * (i + 1)
            value = np.random.uniform(a, b)

            """
            Experience that correspond to each value is retrieved
            """
            index, priority, data = self.tree.get_leaf(value)

            # P(j)
            sampling_probabilities = priority / self.tree.total_priority

            #  IS = (1/N * 1/P(i))**b /max wi == (N*P(i))**-b  /max wi
            b_ISWeights[i, 0] = np.power(
                n * sampling_probabilities, -self.PER_b) / max_weight

            b_idx[i] = index

            experience = [data]

            memory_b.append(experience)

        return b_idx, memory_b, b_ISWeights

    """
    Update the priorities on the tree
    """

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.PER_e  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)
        ps = np.power(clipped_errors, self.PER_a)

        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)


class ExperienceBuffer(object):  # stored as ( s, a, r, s_ ) in Experience Buffer
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque([], maxlen=capacity)
        self.priority = deque([], maxlen=capacity)
        self.position = 0

    def store(self, experience, priority):
        self.memory.append(experience)
        self.priority.append(priority.detach().cpu())
        self.position = (self.position + 1) % self.capacity

    def count(self):  # number of experience stored
        return np.size(self.priority)

    def weight(self, randomness=0.5):  # priority distribution amount of experience
        reg_pro = list(map(lambda prob: np.power(
            np.abs(prob), randomness), self.priority))
        return np.divide(reg_pro, np.sum(reg_pro))

    def samples(self, batch_size, randomness=0.5):
        return list(map(lambda index: self.memory[index],
                        np.random.choice(
            self.count(),
            batch_size,
            p=self.weight(randomness=randomness))))

    def __len__(self):
        return len(self.memory)


class SequentialBuffer(object):  # stored as ( s, a, r, s_ ) in Experience Buffer
    def __init__(self, capacity, gamma, lamb):
        self.capacity = capacity
        self.memory = deque([])
        self.episodes = deque([], maxlen=capacity)
        self.position = 0
        self.gamma = gamma
        self.lamb = lamb

    def count(self):  # number of experience stored
        return len(self.episodes)

    def add(self, transistion):  # add experience
        self.memory.append(transistion)
        self.position = (self.position + 1) % self.capacity

    def discount_cumsum(self, x, discount):  # discount experience
        return scipy.signal.lfilter([1], [1, -discount], x[::-1], axis=0)[::-1]

    def pack_episodes(self):  # pack episodes
        s, a, r, _, st, logp = [*zip(*self.memory)]
        rg = self.discount_cumsum(r, self.gamma).copy()
        self.episodes.append(Transition(
            s, a, r, torch.as_tensor(rg).split(1), st, logp))
        self.memory.clear()

    def unpack_episodes(self):
        return [reduce(lambda x, y: x + y, t)
                for t in zip(*self.episodes)]

    def samples(self):
        return self.episodes

    def __len__(self):
        return len(self.memory)


""" 
Priotize Experience Replay
"""


@dataclass(order=True)
class Experience:
    priority: float
    count: int
    item: Any = field(compare=False)


class ReplayBuffer(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.counter = itertools.count()
        self.size = 0

    def store(self, item, priority=0) -> None:
        # using counter as tierbreaker for equal priorities consideration
        count = next(self.counter)
        entry = Experience(priority, count, item)

        # use heap replace when capacity is full,
        # which efficiently maintains the FIFO policy for fixed size heap
        if self.size >= self.capacity and self.size != 0:
            heapreplace(self.buffer, entry)
        else:
            heappush(self.buffer, entry)

        self.size += 1

    def samples(self, num_samples):
        heapify(self.buffer)
        return [exp.item for exp in nlargest(num_samples, self.buffer)]
