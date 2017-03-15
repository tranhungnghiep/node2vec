import numpy as np

# String id.
with open('/Users/mac/PythonProjects/node2vec/graph/karate.edgelist', 'r') as f, \
        open('/Users/mac/PythonProjects/node2vec/graph/karate_str.edgelist', 'w') as f2:
    for line in f:
        d = line.split()
        d1 = 'J' + str(d[0])
        d2 = 'J' + str(d[1])
        f2.write(d1 + ' ' + d2 + '\n')

# Weighted.
with open('/Users/mac/PythonProjects/node2vec/graph/karate.edgelist', 'r') as f, \
        open('/Users/mac/PythonProjects/node2vec/graph/karate_w.edgelist', 'w') as f2:
    for line in f:
        f2.write(line.strip() + ' ' + str(np.random.randint(1, 10)) + '\n')

# Weighted and string id.
with open('/Users/mac/PythonProjects/node2vec/graph/karate_w.edgelist', 'r') as f, \
        open('/Users/mac/PythonProjects/node2vec/graph/karate_w_str.edgelist', 'w') as f2:
    for line in f:
        d = line.split()
        d1 = 'J' + str(d[0])
        d2 = 'J' + str(d[1])
        d3 = str(d[2])
        f2.write(d1 + ' ' + d2 + ' ' + d3 + '\n')
