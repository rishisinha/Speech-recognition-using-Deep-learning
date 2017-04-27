import sys

with open(sys.argv[1],'r') as f:
	fcon = f.read()
	lines = fcon.split('\n')
	col = []
	for l in lines:
		col.append(l.split(' '))
	print col
		
