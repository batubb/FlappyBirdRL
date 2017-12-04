import sys
sys.path.append("game/")
import flappy_bird_utils
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


IMAGES, SOUNDS, HITMASKS = flappy_bird_utils.load()
PIPE_HEIGHT = IMAGES['pipe'][0].get_height();

print(PIPE_HEIGHT)

game_state = game.GameState()
do_nothing = np.zeros(2)
do_nothing[0] = 1
do_nothing[1] = 0


initial_state = (game_state.playery, game_state.playerVelY, (game_state.upperPipes[0]['x'], game_state.upperPipes[0]['y'] + PIPE_HEIGHT), \
	(-1, -1), (game_state.lowerPipes[0]['x'], game_state.lowerPipes[0]['y'] + PIPE_HEIGHT), (-1, -1))

def update_state(cur_state, action):
	new_action = np.zeros(2)
	if(action[0] == 1): #Dropping
		new_playery = cur_state[0] + min(self.playerVelY, BASEY - cur_state[0] - PLAYER_HEIGHT)
		if(new_playery < 0):
			new_playery = 0
		new_velocity = min(cur_state[1] + game_state.playerAccY, game_state.playerMaxVelY)
		
		new_state = ()


while(True):
	x_t, r_0, terminal = game_state.frame_step(do_nothing) 
	#print(game_state.upperPipes[0]['y'] + PIPE_HEIGHT)
	print(game_state.playerVelY) 