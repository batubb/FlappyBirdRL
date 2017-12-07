import sys
sys.path.append("game/")
import flappy_bird_utils as utils
import wrapped_flappy_bird as game
import numpy as np
import math
import collections
import random
import dill 

class FlappyBirdLearner(object):
	def __init__(self):
		self.gamma = 1
		self.learning_rate = 1
		self.history = []
		self.s_t = (19, 19, 0, 10)
		self.a_t = 0
		self.alpha = 1
		self.grid_dim = 20
		self.Q = collections.defaultdict(lambda: np.zeros(2))
		self.constants = {}

	def setConstants(self, glossary):
		self.constants = glossary

	def save(self):
		
		
	def learn(self):
		if (len(self.history) == 1):
			return

		s_t, a_t, sp_t = self.history[0]
		r_t = 1
		for timestep in range(1, len(self.history)):
			s_tpo, a_tpo, sp_tpo = self.history[timestep]
			r_tpo = 1 if timestep < len(self.history)-1 else -1000
			if (r_tpo < 0):
				self.Q[s_t][a_t] = self.Q[s_t][a_t] + self.alpha * (r_tpo - self.Q[s_t][a_t])
			else:
				self.Q[s_t][a_t] = self.Q[s_t][a_t] + self.alpha * (r_t + self.alpha * self.Q[s_tpo][a_tpo]-self.Q[s_t][a_t])
			s_t, a_t, r_t, sp_t = s_tpo, a_tpo, r_tpo, sp_tpo

	# Discretize into a 20 x 20 grid
	def discretizeState(self, player_x, player_y, pipe_x, pipe_y):
		x_distance = pipe_x - player_x - self.constants['PIPE_WIDTH']
		y_distance = pipe_y - player_y

		x_coord = int(x_distance+60)/12

		# print pipe_x, player_x, x_distance 
		y_coord = int(y_distance+384)/40

		return x_coord, y_coord

	def formState(self, player_x, player_y, pipe_x, pipe_y, player_v):
		x_coord, y_coord = self.discretizeState(player_x, player_y, pipe_x, pipe_y)
		return (x_coord, y_coord, player_v)

	def bestAction(self, state):
		return np.argmax(self.Q[state])

	def takeAction(self, player_x, player_y, pipe_x, pipe_y, player_v):
		state = self.formState(player_x, player_y, pipe_x, pipe_y, player_v)
		self.history.append((self.s_t, self.a_t, state))

		self.s_t = state
		self.a_t = np.argmax(self.Q[state])

		print (state, self.Q[state])

		return self.a_t

	def takeRandomAction(self, player_x, player_y, pipe_x, pipe_y, player_v, action):
		state = self.formState(player_x, player_y, pipe_x, pipe_y, player_v)
		self.history.append((self.s_t, self.a_t, state))

		self.s_t = state
		self.a_t = action

		return action

class FlappyBirdGamePlayer():

	def __init__(self):
		self.game_state = game.GameState()
		self.learner = FlappyBirdLearner()
		self.learner.setConstants(self.game_state.getConstantsGlossary())
		self.random_threshold = 200
		self.flap = np.array([0, 1])
		self.drop = np.array([1, 0])
		self.drop_idx = 0
		self.flap_idx = 1

	def getNextPipe(self):
		if (self.game_state.playerx < self.game_state.lowerPipes[0]['x'] + self.game_state.getConstants('PIPE_WIDTH')):
			return self.game_state.lowerPipes[0]
		else:
			return self.game_state.lowerPipes[1]

	def idxToAction(self, idx):
		return self.flap if idx == self.flap_idx else self.drop

	def playTrainingGame(self, idx):
		print "### GAME", idx, "######"
		terminal = False
		beh = False
		while(not terminal):
			closest_pipe = self.getNextPipe()
			# if (idx < self.random_threshold and random.uniform(0,1) < 0.02):
			# 	beh = True
			# 	action = random.randint(0, 1)
			# 	self.learner.takeRandomAction(self.game_state.playerx, self.game_state.playery, closest_pipe['x'], closest_pipe['y'], self.game_state.playerVelY, action)
			# else:
			action = self.learner.takeAction(self.game_state.playerx, self.game_state.playery, closest_pipe['x'], closest_pipe['y'], self.game_state.playerVelY)
			x_t, score, terminal = self.game_state.frame_step(self.idxToAction(action))
		self.learner.learn()

	def train(self, n_iters = 1000):
		for i in range(0, n_iters):
			self.playTrainingGame(i)

	def playGame(self):
		return

	def save(self):
		self.learner.save()

def main():
	flappy_bird = FlappyBirdGamePlayer()
	flappy_bird.train(1500)
	flappy_bird.save()

if __name__ == '__main__':
	main()













