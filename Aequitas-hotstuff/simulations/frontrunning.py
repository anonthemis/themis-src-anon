from itertools import permutations
import csv
import random

n = 100
f = 50
print(n,f)
ping_dataset = {}

with open('pings.csv') as datafile:
    reader = csv.reader(datafile)
    data = list(reader)[1:]

nodes = set()

for line in data:
    source, dest = int(line[0]), int(line[1])
    nodes.add(source)
    val = float(line[4])

    if (source, dest) in ping_dataset:
        ping_dataset[(source,dest)] = (ping_dataset[(source,dest)][0] + val, ping_dataset[(source,dest)][1] + 1)
    else:
        ping_dataset[(source, dest)] = (val, 1)


for x in ping_dataset:
    ping_dataset[x] = ping_dataset[x][0] / ping_dataset[x][1]

print("Read Dataset")


def test_separability(prot_nodes):
    perm = permutations(prot_nodes,3)
    osdi_failures = set()
    osdi_normalized = set()

    for (A,B,C) in perm:
        # D can be equal to C
        for D in set(prot_nodes).difference({A,B}):
            if (A,B) not in ping_dataset or (B,C) not in ping_dataset or (A,D) not in ping_dataset:
                continue
            elif ping_dataset[(A,B)] + ping_dataset[(B,C)] < ping_dataset[(A,D)]:
                osdi_failures.add((A,B,C,D))
                osdi_normalized.add((A,B))

    return (len(osdi_failures), len(osdi_normalized))


def test_order_fairness(prot_nodes):
    triangle_failures = set()
    ordfair_failures = set()

    perm = permutations(prot_nodes,3)

    fails_per_pair = {}

    for (A,B,K) in perm:
        if (A,B) not in ping_dataset or (B,K) not in ping_dataset or (A,K) not in ping_dataset:
            continue
        elif ping_dataset[(A,B)] + ping_dataset[(B,K)] < ping_dataset[(A,K)]:
            if (A,B) in fails_per_pair:
                fails_per_pair[(A,B)] += 1
            else:
                fails_per_pair[(A,B)] = 1

            triangle_failures.add((A,B,K))

    for (A,B) in fails_per_pair:
        if fails_per_pair[(A,B)] > f:
            ordfair_failures.add((A,B))

    return (len(triangle_failures), len(fails_per_pair), len(ordfair_failures))

# Run the iterations
iters = 10
sep_vals = []
of_vals = []

for i in range(iters):
    prot_nodes = random.sample(nodes, n)
    sep_vals.append(test_separability(prot_nodes))
    of_vals.append(test_order_fairness(prot_nodes))


sep_ind_fails = sum([x for x, _ in sep_vals]) / len(sep_vals)
sep_pair_fails = sum([x for _, x in sep_vals]) / len(sep_vals)
print("Avg OSDI:", sep_ind_fails)
print("Avg pair OSDI:", sep_pair_fails)


tri_fails = sum([x for x, _, __ in of_vals]) / len(of_vals)
tri_pair_fails = sum([x for _, x, __ in of_vals]) / len(of_vals)
of_pair_fails = sum([x for _, __, x in of_vals]) / len(of_vals)

print("Avg Triangle:", tri_fails)
print("Avg Triangle Pairs:", tri_pair_fails)
print("Avg Pair Order-fairness:", of_pair_fails)








# pings =  [[0,159.931 , 131.609 , 51.166 , 98.827],
# [159.561 ,0 , 32.397 , 134.018 , 222.829],  
# [131.359 , 31.723 ,0 , 106.684 , 228.356], 
# [51.407 , 133.771 , 105.733 ,0 , 147.308],
# [97.972 , 234.641 , 228.862 , 145.962 ,0]]

# n = 5
# temp_pairs = set()
# comb = permutations(range(5),4)
# print(comb)
# sep_counter = 0
# for (A,B,C,D) in comb:
#     if pings[A][B] + pings[B][C] < pings[A][D]:
#         print(A,B,C,D)
#         sep_counter += 1
#         temp_pairs.add((A,B))
#     if pings[A][B] + pings[B][C] < pings[A][C]:
#         print(A,B,C,C)
#         sep_counter += 1
#         temp_pairs.add((A,B))
# print(sep_counter)                #16
# print(len(temp_pairs))            #10    
