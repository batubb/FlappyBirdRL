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
		self.gamma = 0.9
		self.history = []
		self.s_t = (19, 19, 0, 10)
		self.a_t = 0
		self.alpha =  1
		self.grid_dim = 20
		self.Q = collections.defaultdict(lambda: np.zeros(2))
		self.N = collections.defaultdict(lambda: np.zeros(2))
		self.constants = {}
		self.lamb = 0.7
		self.flap = np.array([0, 1])
		self.drop = np.array([1, 0])
		self.drop_idx = 0
		self.flap_idx = 1
		self.num_actions = 2

	def setConstants(self, glossary):
		self.constants = glossary

	def save(self, alg):
		filename = 'Q/' + str(alg)
		with open (filename, "wb") as out_strm:
			dill.dump(self.Q, out_strm)

	def load(self, alg):
		filename = 'Q/' + str(alg)
		with open (filename, "rb") as in_strm:
			self.Q = dill.load(in_strm)

	############################## LEARNING ALGORITHMS ##############################

	def learnSarsa(self):
		if (len(self.history) == 1):
			return

		s_tpo, a_tpo, sp_tpo = self.history[len(self.history)-1]
		# r_tpo = -1000

		died_by_upper = sp_tpo[1] > 10

		for timestep in reversed(range(0, len(self.history))):
			s_t, a_t, sp_t = self.history[timestep]
			a_tpo = np.argmax(self.Q[sp_t])

			if ((len(self.history)-2)-timestep<2):
				self.Q[s_t][a_t] = self.Q[s_t][a_t] + self.alpha * (-1000 + self.gamma*self.Q[sp_t][a_tpo]- self.Q[s_t][a_t])
			elif (died_by_upper and a_t == self.flap_idx):
				print ("DIED BY UPPER PIPE...")
				self.Q[s_t][a_t] = self.Q[s_t][a_t] + self.alpha * (-1000 + self.gamma*self.Q[sp_t][a_tpo]- self.Q[s_t][a_t])
				died_by_upper = False
			else:
				self.Q[s_t][a_t] = self.Q[s_t][a_t] + self.alpha * (1 + self.gamma*self.Q[sp_t][a_tpo]- self.Q[s_t][a_t])

			s_tpo, a_tpo, sp_tpo = s_t, a_t, sp_t 
		self.history = []

	def learnSarsaLambda(self):
		if (len(self.history) == 1):
			return
		self.N = collections.defaultdict(lambda: np.zeros(2))
		# s_tpo, a_tpo, sp_tpo = self.history[len(self.history)-1]

		# died_by_upper = sp_tpo[1] > 10
		# delta_flag = died_by_upper
		# total_steps = len(self.history)
		# num_times_penalized = 0
		# for timestep in reversed(range(0, len(self.history)-1)):
		# 	s_t, a_t, sp_t = self.history[timestep]
		# 	self.N[s_t][a_t] = self.N[s_t][a_t] + 1

		# 	if ((len(self.history)-2)-timestep < 10):
		# 		delta = -1000 + self.gamma*self.Q[s_tpo][a_tpo]-self.Q[s_t][a_t]
		# 	elif(died_by_upper and a_t == self.flap_idx):
		# 		delta = -1000 + self.gamma*self.Q[s_tpo][a_tpo]-self.Q[s_t][a_t]
		# 		died_by_upper = False
		# 	else:
		# 		delta = 1 + self.gamma*self.Q[s_tpo][a_tpo]-self.Q[s_t][a_t]

		# 	print s_t, a_t, delta
		# 	last_two_jumps = 2
		# 	for step_idx in reversed(range(1, timestep+1)):
		# 		for a in range(0, self.num_actions):
		# 			s = self.history[step_idx][0]
		# 			# if (delta_flag and a == self.flap_idx and last_two_jumps > 2 and (timestep == total_steps-2) or (timestep == total_steps-1)):
		# 			# 	print "upper"
		# 			# 	delta_penalty = -1000 + self.gamma*self.Q[s_tpo][a_tpo]-self.Q[s_t][a_t]
		# 			# 	self.Q[s][a] = self.Q[s][a]+self.alpha*delta_penalty*self.N[s][a]
		# 			# 	self.N[s][a] = self.gamma * self.lamb * self.N[s][a]
		# 			# else:
		# 			print self.Q[s][a], delta
		# 			self.Q[s][a] = self.Q[s][a]+self.alpha*delta*self.N[s][a]
		# 			self.N[s][a] = self.gamma * self.lamb * self.N[s][a]
		# 			# print self.Q[s]
		# 	s_tpo, a_tpo, sp_tpo = s_t, a_t, sp_t
		# self.history = []
		# s_t, a_t, sp_t = self.history[0]
		# died_by_upper = sp_tpo[1] > 10
		# delta_flag = died_by_upper
		died_by_upper = self.history[len(self.history)-1][2][1] > 10
		total_steps = len(self.history)
		num_times_penalized = 0
		for timestep in range(1, len(self.history)):
			s_t, a_t, sp_t = self.history[timestep]
			a_tpo = np.argmax(self.Q[sp_t])
			self.N[s_t][a_t] = self.N[s_t][a_t] + 1

			if (timestep >= total_steps-2):
				delta = -1000 + self.gamma*self.Q[sp_t][a_tpo]-self.Q[s_t][a_t]
			elif(died_by_upper and a_tpo == self.flap_idx):
				f_left = 0
				for rest in range(timestep, len(self.history)):
					s_t, a_t, sp_t = self.history[rest]
					if a_t == self.flap_idx:
						f_left += 1 
				if f_left != 0:
					delta = -1000 + self.gamma*self.Q[sp_t][a_tpo]-self.Q[s_t][a_t]
					died_by_upper = False
				print "died by upper pipe"
			else:
				delta = 1 + self.gamma*self.Q[sp_t][a_tpo]-self.Q[s_t][a_t]

			for step_idx in range(0, len(self.history)):
				for a in range(0, self.num_actions):
					s = self.history[step_idx][0]
					self.Q[s][a] = self.Q[s][a]+self.alpha*delta*self.N[s][a]
					self.N[s][a] = self.gamma * self.lamb*self.N[s][a]
		self.history = []

	def learnQ(self):
		if (len(self.history) == 1):
			return
		total_steps = len(self.history)
		died_by_upper = self.history[total_steps-1][2][1] > 10

		for timestep in reversed(range(0,total_steps)):
			s_t, a_t, s_tpo = self.history[timestep]
			r_t = -1000 if timestep == total_steps-1 else 1

			if ((len(self.history)-2)-timestep<4):
				self.Q[s_t][a_t] = self.Q[s_t][a_t] + self.alpha * (-1000 + (self.gamma)*np.amax(self.Q[s_tpo]))
			elif(died_by_upper and a_t == self.flap_idx):
				self.Q[s_t][a_t] = self.Q[s_t][a_t] + self.alpha * (-1000 + (self.gamma)*np.amax(self.Q[s_tpo]))
				died_by_upper = False
			else:
				self.Q[s_t][a_t] = self.Q[s_t][a_t] + self.alpha * (1 + (self.gamma)*np.amax(self.Q[s_tpo]))
		self.history = []


	def learnQLambda(self):
		if (len(self.history) == 1):
			return
		self.N = collections.defaultdict(lambda: np.zeros(2))

		total_steps = len(self.history)
		died_by_upper = self.history[total_steps-1][2][1] > 10

		for timestep in reversed(range(0,total_steps)):
			s_t, a_t, sp_t = self.history[timestep]
			self.N[s_t][a_t] = self.N[s_t][a_t] + 1

			a_tpo = np.argmax(self.Q[sp_t])
			print (s_t, a_t, sp_t, self.Q[s_t], self.Q[sp_t])

			if ((len(self.history)-2) - timestep <= 1):
				delta = -1000 + self.gamma * self.Q[sp_t][a_tpo]-self.Q[s_t][a_t]
			elif(died_by_upper and a_t == self.flap_idx):
				delta = -1000 + self.gamma * self.Q[sp_t][a_tpo]-self.Q[s_t][a_t]
				died_by_upper = False
			else:
				delta = 1 + self.gamma * self.Q[sp_t][a_tpo]-self.Q[s_t][a_t]

			for step_idx in reversed(range(1, timestep+1)):
				for a in range(0, self.num_actions):
					s = self.history[step_idx][0]
					self.Q[s][a] = self.Q[s][a]+self.alpha*delta*self.N[s][a]
					if a == np.argmax(self.Q[s]):
						self.N[s][a] = self.gamma * self.lamb * self.N[s][a]
					else:
						self.N[s][a] = 0
		self.history = []

	############################## GENERALIZATION ##############################

	# def handleUnseen(self):
	# 	total = 30 * 30 * 19
	# 	seen_states self.Q.keys()
	# 	for sidx in range(0, total):
	# 		for action in range(0, self.num_actions):
	# 			running_total = 0
	# 			normalizer = 0
	# 			if (self.Q[sidx][action] == 0): 
	# 				shouldContinue = True
	# 				distance = 1
	# 				while shouldContinue:
	# 					neighbors = self.get_neighbors(seen_states[state], distance)
	# 					for s in neighbors:
	# 						q_val = self.Q[s][action]
	# 						if q_val != 0:
	# 							shouldContinue = False
	# 							running_total += q_val
	# 							normalizer += 1
	# 					distance += 1
	# 				if (normalizer == 0):
	# 					normalizer += 1
	# 				self.Q[][action] = running_total/float(normalizer)
	# 	print len(self.Q.keys())
	def handleUnseen(self):
		total = 20 * 20 * 18
		seen_states = self.Q.keys()
		print len(seen_states)
		count = 0
		for x in range(0, 20):
			for y in range(0, 20):
				for v in range(-8, 11):
					count += 1
					state = (x, y, v)
					# print state
					for action in range(0, self.num_actions):
						running_total = 0
						normalizer = 0
						if (self.Q[state][action] == 0):
							shouldContinue = True
							distance = 1
							while shouldContinue:
								neighbors = self.get_neighbors(state, distance)
								for s in neighbors:
									q_val = self.Q[s][action]
									if q_val != 0:
										shouldContinue = False
										running_total += q_val
										normalizer += 1
								distance += 1
							if (normalizer == 0):
								normalizer += 1
							self.Q[state][action] = running_total/float(normalizer)
		x_vals = [s[0] for s in self.Q.keys()]
		y_vals = [s[1] for s in self.Q.keys()]
		v_vals = [s[2] for s in self.Q.keys()]

		print max(x_vals), max(y_vals), max(v_vals)
		print min(x_vals), min(y_vals), min(v_vals)
		
	def get_neighbors(self, state, distance):
		neighbors = []
		# print state
		x_neighbors = self.get_candidates(state[0], distance, 0, 19)
		y_neighbors = self.get_candidates(state[1], distance, 0, 19)
		v_neighbors = self.get_candidates(state[2], distance, -8, 10)

		neighbors = set()
		for x in x_neighbors:
			for y in y_neighbors:
				for v in v_neighbors:
					neighbors.add((x, y, v))
		return neighbors

	def get_candidates(state, basis, distance, floor, ceiling):
		neighbors = set()
		for delta in range(0, distance):
			if (basis-delta >= floor):
				neighbors.add(basis-delta)
			if (basis+delta <= ceiling):
				neighbors.add(basis+delta)
		return neighbors

	############################## CONSTRUCTING THE STATE SPACE ##############################

	# Discretize into a 20 x 20 grid
	def discretizeState(self, player_x, player_y, pipe_x, pipe_y):
		x_distance = pipe_x - player_x

		# pipex starts at 288, player always at 57.6
		# max = 230.6
		# min = -52
		# minimum is -PIPE_WIDTH


		# can range from 200-271
		# lowest = highest_pipe-(BASEY-player_height) = -133.48 ---> 200-(404.48)
		# HIGHEST = lowest_pipe-player_height
		# highest = pipe_y-player_x = 270
		# total: 405
		# round to 420
		# print self.constants['PLAYER_HEIGHT']

		y_distance = pipe_y - player_y
		# print pipe_y
		x_coord = int(x_distance+59)/15
		# x_coord = int(x_distance+60)/8
		# height = 512
		# pipe_height = 320
		# highest possible = 270
		# lowest possible = 200
		# print pipe_x, player_x, x_distance 
		# y_coord = int(y_distance+384)/40
		# so there are 70, should disc into 
		# y_coord = int(y_distance+384)/40
		y_coord = int(y_distance+214)/25
		# y_coord = int(y_distance+135)/21
		# if (x_coord > 19)
		if (x_coord < 0 or y_coord < 0 or x_coord > 19 or y_coord > 19):
			print (x_coord, y_coord)
			print (player_x, pipe_x, x_distance)
			print (player_y, pipe_y, y_distance)
			print "\n"

		return x_coord, y_coord

	def formState(self, player_x, player_y, pipe_x, pipe_y, player_v):
		x_coord, y_coord = self.discretizeState(player_x, player_y, pipe_x, pipe_y)
		return (x_coord, y_coord, player_v)

	def takeAction(self, player_x, player_y, pipe_x, pipe_y, player_v):
		state = self.formState(player_x, player_y, pipe_x, pipe_y, player_v)
		self.history.append((self.s_t, self.a_t, state))

		self.s_t = state
		self.a_t = np.argmax(self.Q[state])
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
		self.scores = []

	def getNextPipe(self):
		if (self.game_state.playerx < self.game_state.lowerPipes[0]['x'] + self.game_state.getConstants('PIPE_WIDTH')-4):
			return self.game_state.lowerPipes[0]
		else:
			return self.game_state.lowerPipes[1]

	def idxToAction(self, idx):
		return self.flap if idx == self.flap_idx else self.drop

	def getRandomThreshold(self, idx):
		if (0 <= idx and 250 > idx):
			return 0.1
		if (250 <= idx and 500 > idx):
			return 0.025
		if (500 <= idx and 500 > idx):
			return 0.01
		if (500 <= idx and 1000 > idx):
			return 0.005
		if (1000 <= idx and 2000 > idx):
			return 0.005
		return -1	

	def playTrainingGame(self, idx):
		print "### GAME", idx, "######"
		terminal = False
		score = 0
		while(not terminal):
			closest_pipe = self.getNextPipe()
			if (idx < self.random_threshold and random.uniform(0,1) < 0.08):
				action = random.randint(0, 1)
				self.learner.takeRandomAction(self.game_state.playerx, self.game_state.playery, closest_pipe['x'], closest_pipe['y'], self.game_state.playerVelY, action)
			else:
				action = self.learner.takeAction(self.game_state.playerx, self.game_state.playery, closest_pipe['x'], closest_pipe['y'], self.game_state.playerVelY)
			x_t, score, terminal = self.game_state.frame_step(self.idxToAction(action))
		print "SCORE:", score
		self.scores.append(score)
		self.learner.learnSarsa()
		# self.learner.learnSarsaLambda()
		# self.learner.learnQ()
		# self.learner.learnQLambda()

	def train(self, n_iters = 5000):
		for i in range(0, n_iters):
			self.playTrainingGame(i)
			print "HIGH SCORE", max(self.scores)
		self.learner.save('sarsa')
		# self.learner.save('sarsa_lambda')
		# self.learner.save('q_learn')
		# self.learner.save('q_learn_lambda')

	def playGame(self):
		terminal = False
		score = 0
		while(not terminal):
			closest_pipe = self.getNextPipe()
			state = self.learner.formState(self.game_state.playerx, self.game_state.playery, closest_pipe['x'], closest_pipe['y'], self.game_state.playerVelY)
			action = np.argmax(self.learner.Q[state])
			x_t, score, terminal = self.game_state.frame_step(self.idxToAction(action))
		print "SCORE:", score

def main():
	flappy_bird = FlappyBirdGamePlayer()
	flappy_bird.train(5000)
	# flappy_bird.learner.save("sarsa")
	# flappy_bird.learner.load("sarsa")
	# flappy_bird.learner.handleUnseen()
	# flappy_bird.playGame()

if __name__ == '__main__':
	main()













