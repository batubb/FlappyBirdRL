import sys
sys.path.append("game/")
import flappy_bird_utils as utils
import wrapped_flappy_bird as game
import numpy as np


#Pipes move 4 pixels to the left each frame
#Player moves 9 pixels up when jumping (next frame)
#Player moves 9 pixels up when jumping (next frame)
#State: (Player Y, Player Y Velocity, Upper Pipe 1, Upper Pipe 2, Lower pipe 1, Lower pipe 2)

FPS = 30
SCREENWIDTH = 288
SCREENHEIGHT = 512

BASEY = SCREENHEIGHT * 0.79


IMAGES, SOUNDS, HITMASKS = utils.load()
# PIPE_HEIGHT = IMAGES['pipe'][0].get_height();

# print(PIPE_HEIGHT)

game_state = game.GameState()
do_nothing = np.zeros(2)
do_nothing[0] = 1
do_nothing[1] = 0


initial_state = (game_state.playerx, game_state.playery, game_state.playerVelY, (game_state.upperPipes[0]['x'], game_state.upperPipes[0]['y'] + game.PIPE_HEIGHT), \
	(-1, -1), (game_state.lowerPipes[0]['x'], game_state.lowerPipes[0]['y'] + game.PIPE_HEIGHT), (-1, -1))

def update_upper_pipes(first_pipe, second_pipe):
	if (first_pipe[0] < 0):
		first_pipe = (second_pipe[0]-4, second_pipe[1])
		second_pipe = (game_state.upperPipes[1]['x'], game_state.upperPipes[1]['y'])
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
		second_pipe = (game_state.lowerPipes[1]['x'], game_state.lowerPipes[1]['y'])
	elif (second_pipe[0] == -1 and second_pipe[1] == -1):
		if (game_state.lowerPipes[1]['x'] < 400):
			second_pipe = (game_state.lowerPipes[1]['x'], game_state.lowerPipes[1]['y'])
		first_pipe = (first_pipe[0]-4, first_pipe[1])
	else:
		first_pipe = (first_pipe[0]-4, first_pipe[1])
		second_pipe = (second_pipe[0]-4, second_pipe[1])
	return first_pipe, second_pipe

def calculateScore(state):
	reward = 0
	playerMidPos = state[0] + game_state.PLAYER_WIDTH / 2
	for pipe in [state[3], state[4]]:
		pipeMidPos = pipe['x'] + game_state.PIPE_WIDTH / 2
		if pipeMidPos <= playerMidPos < pipeMidPos + 4:
			reward += 1
	return reward

def update_state(cur_state, action):
	new_action = np.zeros(2)
	if(action[0] == 1): #Dropping
		if(new_playery < 0):
			new_playery = 0
		new_playery = cur_state[1] + min(game_state.playerVelY, game.BASEY - cur_state[1] - game.PLAYER_HEIGHT)
		new_velocity = min(cur_state[2] + game_state.playerAccY, game_state.playerMaxVelY)
	else:
		new_playery = max(cur_state[1] + min(game_state.playerVelY, game.BASEY - cur_state[1] - game.PLAYER_HEIGHT), 0)
		new_velocity = game_state.playerFlapAcc

	upper_one, upper_two = update_upper_pipes(cur_state[3], cur_state[4])
	lower_one, lower_two = update_lower_pipes(cur_state[5], cur_state[6])
	return (cur_state[0], new_playery, new_velocity, upper_one, upper_two, lower_one, lower_two)

print ("INITIAL: ", initial_state) 
state = initial_state
terminal = False
count = 0

while(not terminal):
	do_nothing = np.zeros(2)
	do_nothing[0] = 1
	do_nothing[1] = 0
	
	# game_state_now = (game_state.playery, game_state.playerVelY, (game_state.upperPipes[0]['x'], game_state.upperPipes[0]['y'] + game.PIPE_HEIGHT), \
	# (-1, -1) if game_state.upperPipes[1]['x'] >= 400 else (game_state.upperPipes[1]['x'], game_state.upperPipes[1]['y']), (game_state.lowerPipes[0]['x'],\
	# game_state.lowerPipes[0]['y'] + game.PIPE_HEIGHT), (-1, -1) if game_state.lowerPipes[1]['x'] >= 400 else (game_state.lowerPipes[1]['x'], game_state.lowerPipes[1]['y']))

	# print "GAME AFTER: ", game_state_now, "\n"
	x_t, r_0, terminal = game_state.frame_step([0,1]) 

	state = update_state(state, [0,1])
	print "Update ", count, ":", state
	count += 1

