import numpy as np
from numpy.random import randint
import collections
from collections import Counter

def quadratic_kappa_cost(e_a, e_b):
	N = 5
	O = np.zeros((N, N), dtype=np.float)
	w = np.zeros((N, N))
	for i, j in zip(e_a, e_b):
	    O[i, j] += 1
	O = np.array(O / np.sum(O), np.float)
	for i in range(0, N):
	    for j in range(0, N):
		w[i, j] = ((i-j)**2) / float(((N-1)**2))
	e_a_hist = np.array(collections.OrderedDict(sorted(Counter(e_a).items())).values())
	e_b_hist = np.array(collections.OrderedDict(sorted(Counter(e_b).items())).values())
	E = np.array(np.outer(e_a_hist, e_b_hist), dtype=np.float32)
	E = E / np.sum(E)
	k = 1 - np.sum(np.multiply(w, O)) / np.sum(np.multiply(w,E))
	return k
