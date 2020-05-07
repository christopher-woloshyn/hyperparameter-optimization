import generation
import numpy as np

X = np.array([
[0, 0],
[0, 1],
[1, 0],
[1, 1],
])

y = np.array([
[0],
[1],
[1],
[0],
])

G = generation.Generation(X, y, pop_size=4)
G.calc_fitness()
for i in range(4):
    G.next_generation()
    G.calc_fitness()
G.print_gen_info()
