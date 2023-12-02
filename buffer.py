import random
from collections import deque
from utils.dataStructure import Max_Heap, Deque
import numpy as np
import torch

class SILReplayBuffer(object):
    def __init__(self, capacity, gamma=0.99):
        self.buffer = deque(maxlen=capacity)
        self.gamma=gamma

    def push(self, pixel_states, symbolic_states, actions, rewards, next_states, dones):
        returns = self.compute_returns(rewards)
        for i in range(len(pixel_states)):
            self.buffer.append((pixel_states[i], symbolic_states[i], actions[i], rewards[i], next_states[i], dones[i], returns[i]))

    def sample(self, batch_size):
        pixel_states, symbolic_states, actions, rewards, next_states, dones, returns = zip(
            *random.sample(self.buffer, batch_size))
        return pixel_states, symbolic_states, actions, rewards, next_states, dones, returns

    def __len__(self):
        return len(self.buffer)

    def compute_returns(self, rewards):
        returns = [0] * len(rewards)
        cumulative_return = 0
        for i in reversed(range(len(rewards))):
            cumulative_return = rewards[i] + self.gamma * cumulative_return
            returns[i] = cumulative_return
        return returns


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, pixel_state, symbolic_state, action, reward, next_state, done):
        # symbolic_state is for the rule learner to choose action
        self.buffer.append((pixel_state, symbolic_state, action, reward, next_state, done))

    def sample(self, batch_size):
        pixel_states, symbolic_states, actions, rewards, next_states, dones = zip(
            *random.sample(self.buffer, batch_size))
        return pixel_states, symbolic_states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class MixReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, pixel_state, symbolic_state, action, reward, next_pixel_state, next_symbolic_state, done):
        self.buffer.append(
            (pixel_state, symbolic_state, action, reward, next_pixel_state, next_symbolic_state, done))

    def sample(self, batch_size):
        pixel_states, symbolic_states, actions, rewards, next_pixel_state, next_symbolic_state, dones = zip(
            *random.sample(self.buffer, batch_size))
        return pixel_states, symbolic_states, actions, rewards, next_pixel_state, next_symbolic_state, dones

    def __len__(self):
        return len(self.buffer)


class IntrinsicReplayBuffer(ReplayBuffer):
    def sample(self, batch_size, KB, coeff):
        pixel_state, symbolic_state, action, reward, next_state, done = zip(
            *random.sample(self.buffer, batch_size))
        intrinic_reward = coeff * (action == KB.advice(symbolic_state))
        return pixel_state, symbolic_state, action, reward + intrinic_reward, next_state, done


class SILIntrinsicReplayBuffer(SILReplayBuffer):
    def sample(self, batch_size, KB, coeff):
        pixel_state, symbolic_state, action, reward, next_state, done, returns = zip(
            *random.sample(self.buffer, batch_size))
        intrinic_reward = coeff * (action == KB.advice(symbolic_state))
        return pixel_state, symbolic_state, action, reward + intrinic_reward, next_state, done, returns


class PrioritisedReplayBuffer(Max_Heap, Deque):
    def __init__(self, capacity, alpha, beta, incremental_td_error, device, game_type, seed=0):
        Max_Heap.__init__(
            self, capacity, dimension_of_value_attribute=5, default_key_to_use=0)
        Deque.__init__(self, capacity, dimension_of_value_attribute=5)
        self.max_size = capacity
        np.random.seed(seed)

        self.deques_td_errors = np.zeros(self.max_size)

        self.heap_index_to_overwrite_next = 1
        self.number_experiences_in_deque = 0
        self.adapted_overall_sum_of_td_errors = 0

        self.alpha = alpha
        self.beta = beta
        self.incremental_td_error = incremental_td_error

        self.heap_indexes_to_update_td_error_for = None

        self.indexes_in_node_value_tuple = {
            "state": 0,
            "action": 1,
            "reward": 2,
            "next_state": 3,
            "done": 4
        }

        self.device = device
        self.game_type = game_type

    def push(self, pixel_state, symbolic_state, action, reward, next_state, done):
        if self.game_type == "frame":
            state = np.expand_dims(pixel_state, axis=0)
            next_state = np.expand_dims(next_state, axis=0)
        else:
            state = symbolic_state
        td_error = (abs(self.give_max_key()) +
                    self.incremental_td_error) ** self.alpha
        self.adapted_overall_sum_of_td_errors += td_error - \
            self.deque[self.deque_index_to_overwrite_next].key
        self.update_deque_and_deque_td_errors(
            td_error, state, action, reward, next_state, done)
        self.update_heap_and_heap_index_to_overwrite()
        self.update_number_experiences_in_deque()
        self.update_deque_index_to_overwrite_next()

    def update_deque_and_deque_td_errors(self, td_error, state, action, reward, next_state, done):
        self.deques_td_errors[self.deque_index_to_overwrite_next] = td_error
        self.update_deque_node_key_and_value(
            self.deque_index_to_overwrite_next, td_error, (state, action, reward, next_state, done))

    def update_heap_and_heap_index_to_overwrite(self):
        """Updates the heap by rearranging it given the new experience that was just incorporated into it. If we haven't
        reached max capacity then the new experience is added directly into the heap, otherwise a pointer on the heap has
        changed to reflect the new experience so there's no need to add it in"""
        if not self.reached_max_capacity:
            self.update_heap_element(
                self.heap_index_to_overwrite_next, self.deque[self.deque_index_to_overwrite_next])
            self.deque[self.deque_index_to_overwrite_next].heap_index = self.heap_index_to_overwrite_next
            self.heap_index_to_overwrite_next += 1

        heap_index_change = self.deque[self.deque_index_to_overwrite_next].heap_index
        self.reorganise_heap(heap_index_change)

    def swap_heap_elements(self, index1, index2):
        self.heap[index1], self.heap[index2] = self.heap[index2], self.heap[index1]
        self.heap[index1].heap_index = index1
        self.heap[index2].heap_index = index2

    def sample(self, batch_size):
        experiences, deque_sample_indexes = self.pick_experiences_based_on_proportional_td_error(
            batch_size)
        states, actions, rewards, next_states, dones = self.separate_out_data_types(
            experiences)
        actions = torch.tensor(actions.cpu().numpy(), dtype=torch.int64)
        self.deque_sample_indexes_to_update_td_error_for = deque_sample_indexes

        if self.game_type == "frame":
            return states, [None], actions, rewards, next_states, dones
        else:
            return [None], states, actions, rewards, next_states, dones

    def pick_experiences_based_on_proportional_td_error(self, batch_size):
        probabilities = self.deques_td_errors / self.adapted_overall_sum_of_td_errors
        deque_sample_indexes = np.random.choice(range(
            len(self.deques_td_errors)), size=batch_size, replace=False, p=probabilities)
        experiences = self.deque[deque_sample_indexes]

        return experiences, deque_sample_indexes

    def separate_out_data_types(self, experiences):
        states = torch.from_numpy(np.vstack(
            [e.value[self.indexes_in_node_value_tuple["state"]] for e in experiences])).float()
        actions = torch.from_numpy(np.vstack(
            [e.value[self.indexes_in_node_value_tuple["action"]] for e in experiences])).float().squeeze()
        rewards = torch.from_numpy(np.vstack(
            [e.value[self.indexes_in_node_value_tuple["reward"]] for e in experiences])).float().squeeze()
        next_states = torch.from_numpy(np.vstack(
            [e.value[self.indexes_in_node_value_tuple["next_state"]] for e in experiences])).float()
        dones = torch.from_numpy(np.vstack(
            [int(e.value[self.indexes_in_node_value_tuple["done"]]) for e in experiences])).float().squeeze()
        return states, actions, rewards, next_states, dones

    def calculate_importance_sampling_weights(self, experiences):
        td_errors = [experience.key for experience in experiences]
        importance_sampling_weights = [((1.0 / self.number_experiences_in_deque) * (
            self.adapted_overall_sum_of_td_errors / td_error)) ** self.beta for td_error in td_errors]
        sample_max_importance_weight = max(importance_sampling_weights)
        importance_sampling_weights = [
            is_weight / sample_max_importance_weight for is_weight in importance_sampling_weights]
        importance_sampling_weights = torch.tensor(
            importance_sampling_weights).float().to(self.device)
        return importance_sampling_weights

    def update_td_errors(self, td_errors):
        for raw_td_error, deque_index in zip(td_errors, self.deque_sample_indexes_to_update_td_error_for):
            td_error = (abs(raw_td_error) +
                        self.incremental_td_error) ** self.alpha
            corresponding_heap_index = self.deque[deque_index].heap_index
            self.adapted_overall_sum_of_td_errors += td_error - \
                self.heap[corresponding_heap_index].key
            self.heap[corresponding_heap_index].key = td_error
            self.reorganise_heap(corresponding_heap_index)
            self.deques_td_errors[deque_index] = td_error

    def __len__(self):
        return self.number_experiences_in_deque
