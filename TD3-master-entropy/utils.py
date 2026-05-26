import numpy as np
import torch


class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	def add(self, state, action, next_state, reward, done):
		"""Add single transition (kept for backward compatibility)"""
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def add_batch(self, states, actions, next_states, rewards, dones):
		"""
		Add batch of transitions efficiently (vectorized operation).
		
		Args:
			states: np.ndarray of shape [batch_size, state_dim]
			actions: np.ndarray of shape [batch_size, action_dim]
			next_states: np.ndarray of shape [batch_size, state_dim]
			rewards: np.ndarray of shape [batch_size] or [batch_size, 1]
			dones: np.ndarray of shape [batch_size] or [batch_size, 1]
		"""
		batch_size = len(states)
		
		# Ensure rewards and dones have correct shape
		if rewards.ndim == 1:
			rewards = rewards.reshape(-1, 1)
		if dones.ndim == 1:
			dones = dones.reshape(-1, 1)
		
		# Calculate indices for circular buffer
		end_ptr = self.ptr + batch_size
		
		if end_ptr <= self.max_size:
			# Simple case: no wraparound
			indices = slice(self.ptr, end_ptr)
			self.state[indices] = states
			self.action[indices] = actions
			self.next_state[indices] = next_states
			self.reward[indices] = rewards
			self.not_done[indices] = 1. - dones
		else:
			# Wraparound case: split into two parts
			first_part_size = self.max_size - self.ptr
			second_part_size = batch_size - first_part_size
			
			# First part: from ptr to end of buffer
			self.state[self.ptr:self.max_size] = states[:first_part_size]
			self.action[self.ptr:self.max_size] = actions[:first_part_size]
			self.next_state[self.ptr:self.max_size] = next_states[:first_part_size]
			self.reward[self.ptr:self.max_size] = rewards[:first_part_size]
			self.not_done[self.ptr:self.max_size] = 1. - dones[:first_part_size]
			
			# Second part: from start of buffer
			self.state[:second_part_size] = states[first_part_size:]
			self.action[:second_part_size] = actions[first_part_size:]
			self.next_state[:second_part_size] = next_states[first_part_size:]
			self.reward[:second_part_size] = rewards[first_part_size:]
			self.not_done[:second_part_size] = 1. - dones[first_part_size:]
		
		# Update pointer and size
		self.ptr = (self.ptr + batch_size) % self.max_size
		self.size = min(self.size + batch_size, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)