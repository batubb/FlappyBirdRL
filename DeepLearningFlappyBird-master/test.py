import dill 
import collections
import numpy as np

d = collections.defaultdict(lambda: np.zeros(2))

d[0] = np.array([1,0])
d[1] = np.array([2,0])
d[2] = np.array([3,0])
d[3] = np.array([4,0])
d[4] = np.array([5,0])
d[5] = np.array([6,0])
d[6] = np.array([7,0])

with open("Q/try", 'wb') as out_strm:
	dill.dump(d, out_strm)

with open("Q/try", 'rb') as in_strm:
	new_d = dill.load(in_strm)
	print new_d
# # SCREENHEIGHT = 512
# # BASEY = SCREENHEIGHT * 0.79
# # grid_dim = {'x': 20, 'y':20}
# # max_x_distance = 800
# # min_x_distance = 

# # x_distance = [60, 70, 80, 90, 100]
# # y_distance = 80

# # for i in x_distance:
# # 	x_cell = 
# # 	print x_cell

# start = -135
# ranges = []
# for i in range(1, 21):
# 	ranges.append((start, start+20))
# 	print (start, start+20)
# 	start = start+ 21

# # print "\n\n"

# # lol = [-55.8, -59.4, -52.88283, -47.3]

# # for i in lol:
# # 	bucket = int(i+60)/12
# # 	print i
# # 	print ranges[bucket]

# # # lol = range(1, 10)
# # # for i in lol:
# # # 	bucket = (i+60)/15

# # start = -384

# # ranges = []
# # for i in range(1, 21):
# # 	ranges.append((start, start+39))
# # 	print (start, start+39)
# # 	start = start + 40

# # lol = range(376, 416)

# # print "\n\n"
# # for i in lol:
# # 	bucket = int(i+384)/40
# # 	# print i
# # 	if (ranges[bucket] != ranges[19]):
# # 		print i
# # 		print ranges[bucket]









