import sys

filename = sys.argv[1]

error = []
with open(filename) as fp:
    for line in fp:
        if line.startswith('	valid_y_misclass: '):
        	error.append(float(line[22:]))
print error
