import sys
sys.path.append("game/")
import flappy_bird_utils as utils
import wrapped_flappy_bird as game
import numpy as np
import math
import collections
import random

#Pipes move 4 pixels to the left each frame
#Player moves 9 pixels up when jumping (next frame)
#Player moves 9 pixels up when jumping (next frame)
#State: (Player X, Player Y, Player Y Velocity, Upper Pipe 1, Upper Pipe 2, Lower pipe 1, Lower pipe 2)

# CONSTANTS
FPS = 30
SCREENWIDTH = 288
SCREENHEIGHT = 512
BASEY = SCREENHEIGHT * 0.79
IMAGES, SOUNDS, HITMASKS = utils.load()
PIPE_HEIGHT = IMAGES['pipe'][0].get_height();
PLAYER_WIDTH = IMAGES['player'][0].get_width()
PLAYER_HEIGHT = IMAGES['player'][0].get_height()
PIPE_WIDTH = IMAGES['pipe'][0].get_width()

# Q LEARNING

NUM_ACTIONS = 2
N_ITERS = 1000000
GAMMA = 1
EPSILON = random.uniform(0, 1)

# Actions
FLAP = np.array([0, 1])
DROP = np.array([1, 0])
DROP_INDEX = 0
FLAP_INDEX = 1

# Initializing Game
game_state = game.GameState()
print game_state.playerx
initial_state = (game_state.playerx, game_state.playery, game_state.playerVelY, (game_state.upperPipes[0]['x'], game_state.upperPipes[0]['y'] + game.PIPE_HEIGHT), \
	(-1, -1), (game_state.lowerPipes[0]['x'], game_state.lowerPipes[0]['y']), (-1, -1), False)

def update_upper_pipes(first_pipe, second_pipe):
	if (first_pipe[0] < 0):
		first_pipe = (second_pipe[0]-4, second_pipe[1])
		second_pipe = (-1, -1) if game_state.upperPipes[1]['x'] > 400 else (game_state.upperPipes[1]['x'], game_state.upperPipes[1]['y'])
	elif (second_pipe[0] == -1 and second_pipe[1] == -1):
		if (game_state.upperPipes[1]['x'] < 400):
			second_pipe = (game_state.upperPipes[1]['x'], game_state.upperPipes[1]['y'])
		first_pipe = (first_pipe[0]-4, first_pipe[1])
	else:
		first_pipe = (first_pipe[0]-4, first_pipe[1])
		second_pipe = (second_pipe[0]-4, second_pipe[1])
	return first_pipe, second_pipe

def update_lower_pipes(first_pipe, second_pipe):
	if (first_pipe[0] < 0):
		first_pipe = (second_pipe[0]-4, second_pipe[1])
		second_pipe = (-1, -1) if game_state.lowerPipes[1]['x'] > 400 else (game_state.lowerPipes[1]['x'], game_state.lowerPipes[1]['y'])
	elif (second_pipe[0] == -1 and second_pipe[1] == -1):
		if (game_state.lowerPipes[1]['x'] < 400):
			second_pipe = (game_state.lowerPipes[1]['x'], game_state.lowerPipes[1]['y'])
		first_pipe = (first_pipe[0]-4, first_pipe[1])
	else:
		first_pipe = (first_pipe[0]-4, first_pipe[1])
		second_pipe = (second_pipe[0]-4, second_pipe[1])
	return first_pipe, second_pipe

def calculate_reward(state, action):
	if (state[7]):
		return -1000
	first_upper_pipe, second_upper_pipe, first_lower_pipe, second_lower_pipe = state[3], state[4], state[5], state[6]

	bird_midpoint = np.array([state[0]+PLAYER_WIDTH/2, state[1]+PLAYER_HEIGHT/2])
	first_pipe = np.array([first_upper_pipe[0]+PIPE_WIDTH/2, (first_lower_pipe[1]-first_upper_pipe[1])/2])

	# first_pipe = np.array([SCREENWIDTH/2, BASEY/2])
	# first_dot_product = bird_midpoint.dot(first_pipe_gap_midpoint)
	# first_denominator = math.sqrt(sum([x ** 2 for x in bird_midpoint]))*math.sqrt(sum([y ** 2 for y in first_pipe_gap_midpoint]))
	
	# reward += (1-first_dot_product/first_denominator)


	AC = abs(bird_midpoint[1]-first_pipe[1])
	BC = abs(bird_midpoint[0]-first_pipe[0])

	if (BC == 0):
		angle = 0
	else:
		angle = -abs(math.atan(AC/BC))

	dist = np.linalg.norm(bird_midpoint-first_pipe)

	prize = 20 if bird_midpoint[0] > first_pipe[0] else 0

	if (prize != 0):
		print "State: ", state
		print "Action: ", "FLAP" if action[0] == 0 else "DROP"
		print "Angle: ", angle
		print "Dist: ", dist
		print angle + prize
	# print "Reward: ", bird_midpoint, first_pipe, angle

	# if (second_upper_pipe[0] == -1 and second_upper_pipe[1] == -1):
	# 	reward += (1-first_dot_product/first_denominator)
	# else:
	# 	second_pipe_gap_midpoint = np.array([second_upper_pipe[0]+PIPE_WIDTH/2, (second_lower_pipe[1]-second_upper_pipe[1])/2])
	# 	second_dot_product = bird_midpoint.dot(second_pipe_gap_midpoint)
	# 	second_denominator = math.sqrt(sum([x ** 2 for x in bird_midpoint]))*math.sqrt(sum([y ** 2 for y in second_pipe_gap_midpoint]))

	# 	if (state[0] > (first_upper_pipe[0]+PIPE_WIDTH)):
	# 		reward += 0.1 * ((1-first_dot_product/first_denominator)) + 0.9 * ((1-second_dot_product/second_denominator))
	# 	else:
	# 		reward += 0.9 * ((1-first_dot_product/first_denominator)) + 0.1 * ((1-second_dot_product/second_denominator))
	
	# if (first_upper_pipe[0]+PIPE_WIDTH < state[0]):
	# 	reward -= 100

	# reward -= 0.05 if action == 1 else 0

	return angle
	# return angle * 0.9 + dist * 0.3 + penalty

def update_state(cur_state, action, terminal):
	new_action = np.zeros(2)
	if(action[0] == 1): #Dropping
		new_playery = cur_state[1] + min(game_state.playerVelY, game.BASEY - cur_state[1] - game.PLAYER_HEIGHT)
		if(new_playery < 0):
			new_playery = 0
		new_velocity = min(cur_state[2] + game_state.playerAccY, game_state.playerMaxVelY)
	else:
		new_playery = max(cur_state[1] + min(game_state.playerVelY, game.BASEY - cur_state[1] - game.PLAYER_HEIGHT), 0)
		new_velocity = game_state.playerFlapAcc

	upper_one, upper_two = update_upper_pipes(cur_state[3], cur_state[4])
	lower_one, lower_two = update_lower_pipes(cur_state[5], cur_state[6])
	return (cur_state[0], new_playery, new_velocity, upper_one, upper_two, lower_one, lower_two, terminal)

def play_game():
	state = initial_state
	terminal = False
	count = 0

	while(not terminal):	
		game_state_now = (game_state.playerx, game_state.playery, game_state.playerVelY, (game_state.upperPipes[0]['x'], game_state.upperPipes[0]['y'] + game.PIPE_HEIGHT), \
		(-1, -1) if game_state.upperPipes[1]['x'] >= 400 else (game_state.upperPipes[1]['x'], game_state.upperPipes[1]['y']), (game_state.lowerPipes[0]['x'],\
		game_state.lowerPipes[0]['y'] + game.PIPE_HEIGHT), (-1, -1) if game_state.lowerPipes[1]['x'] >= 400 else (game_state.lowerPipes[1]['x'], game_state.lowerPipes[1]['y']))

		print game_state_now, "\n"
		x_t, r_0, terminal = game_state.frame_step(FLAP) 

		state = update_state(state, [0,1])
		print "Score:", calculateReward(state)
		count += 1

def training():
	states = set()
	Q = collections.defaultdict(lambda: np.zeros(NUM_ACTIONS))
	for i in range(0, N_ITERS):
		terminal = False
		cur_state = initial_state
		if (i % 20 == 0):
			print "Iteration", i
		while (not terminal):
			if (i < 200 and random.uniform(0,1) < 0.08):
				# print("Randomly chosen action!")
				action_index = random.randint(0,1)
			else:
				action_index = np.argmax(Q[cur_state])

			action = DROP if action_index == 0 else FLAP

			image_data, reward, terminal = game_state.frame_step(action)
			new_state = update_state(cur_state, action, terminal)

			reward = calculate_reward(new_state, action)

			Q[cur_state][action_index] = Q[cur_state][action_index] +  (reward + GAMMA * np.amax(Q[new_state]))
			states.add(cur_state)
			cur_state = new_state
		print ("### DIED ####\n")
	print "Q:", Q

def handleUnseen(Q):
	seen_states = Q.keys()
	for state in range(0, len(seen_states)):
		for action in range(0, NUM_ACTIONS):
			running_total = 0
			normalizer = 0
			if (seen_states[i][action] == 0):
				shouldContinue = True
				distance = 1
				while shouldContinue:
					neighbors = get_neighbors(state, action, distance)
					for neighbor in neighbors:
						neighbor_value = Q[neighbor[1]][neighbor[0]]
						if neighbor_value != 0:
							shouldContinue = False
							running_total += neighbor_value
							normalizer += 1
					distance += 1
				if (normalizer == 0):
					normalizer += 1
				Q[state][action] = running_total/float(normalizer)
	return Q

def get_scalar_candidates(basis, distance, multiplier, floor, ceiling):
	neighbors = []
	for delta in range(0, distance):
		if (basis-delta*multiplier >= floor):
			neighbors.append(basis-delta)
		if (basis+delta*multiplier <= ceiling):
			neighbors.append(basis+delta)
	return neighbors

def get_pipe_candidates(x, y, distance):
	neighbors = []
	multiplier = 10
	for delta in range(0, distance):
		if (y-delta*multiplier >= 0):
			neighbors.append((x, y-delta))
			if (x-delta*multiplier >= 0):
				neighbors.append((x-delta*multiplier, y-delta*multiplier))
			if (x+delta*multiplier <= 400):
				neighbors.append((x+delta*multiplier, y-delta*multiplier))
		if (y+delta*multiplier <= BASEY):
			neighbors.append((x, y+delta))
			if (x-delta*multiplier >= 0):
				neighbors.append((x-delta*multiplier, y+delta*multiplier))
			if (x+delta*multiplier <= 400):
				neighbors.append((x+delta*multiplier, y+delta*multiplier))
		if (x-delta*multiplier >= 0):
			neighbors.append((x-delta, y))
		if (x+delta*multiplier <= 400):
			neighbors.apend((x+delta*multplier, y))
	return neighbors

def get_neighbors(state, action, distance):
	if (action == DROP_INDEX):
		y_neighbors = get_candidates(state[1], distance, 1, 0, BASEY)
	else:
		y_neighbors = get_candidates(state[1], distance, 9, 0, BASEY)
	v_neighbors = get_candidates(state[2], distance, 1, game_state.playerMinVelY, game_state.playerMaxVelY)
	up1_neighbors = get_pipe_candidates(state[3][0], state[3][1], distance)
	up2_neighbors = get_pipe_candidates(state[4][0], state[4][1], distance)
	lp1_neighbors = get_pipe_candidates(state[5][0], state[5][1], distance)
	lp2_neighbors = get_pipe_candidates(state[6][0], state[6][1], distance)

	result = []
	for y in y_neighbors:
		for v in v_neighbors:
			for up1 in up1_neighbors:
				for up2 in up2_neighbors:
					for lp1 in lp1_neighbors:
						for lp2 in lp2_neighbors:
							result.append((action, (state[0], y, v, up1, up2, lp1, lp2)))
	return result

def main():
	# print initial_state
	print PLAYER_WIDTH
	# print PLAYER_HEIGHT
	# print PIPE_WIDTH
	# if (len(sys.argv) > 1 and sys.argv[1] == '--training'):
	# 	training()
	# else:
	# 	play_game()
	print PIPE_HEIGHT

if __name__ == '__main__':
	main()

