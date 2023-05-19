import numpy as np
import copy
from itertools import combinations, permutations
import random
import math

num_txs = 1000
n = 100        # number of nodes

# Parameters
gen_dist = "exponential"
gen_param = 1
network_dist = "exponential"


def sample_nums(dist, param, num_vals):
    if dist == "uniform":
        return np.random.uniform(low=0, high=param*2, size=num_vals)
    elif dist == "exponential":
        return np.random.exponential(scale=param, size=num_vals)


def generate_all_timestamps():
    # Generate Transaction Send timestamps
    send_time_diffs = sample_nums(gen_dist, gen_param, num_txs)
    send_times = np.cumsum(send_time_diffs)

    # Generate Transaction Network delays
    network_delays = sample_nums(network_dist, network_param, (n,num_txs))

    # Receive orderings for all nodes
    honest_node_orderings = send_times + network_delays

    return send_times, honest_node_orderings #, node_orderings



def setup_buckets(honest_node_orderings):
    # Count how close the transactions are initially
    buckets = [set() for _ in range(n//2+1)]
    bucket_map = {}

    for first in range(num_txs):
        for second in range(first+1, num_txs):
            first_counter = 0
            second_counter = 0
            for node in honest_node_orderings:
                if node[first] < node[second]:
                    first_counter += 1
                elif node[second] < node[first]:
                    second_counter += 1

            min_counter = min(first_counter, second_counter)

            buckets[min_counter].add((first,second))
            bucket_map[(first,second)] = min_counter
            bucket_map[(second,first)] = min_counter

    bucket_counts = [len(bucket) for bucket in buckets]

    return bucket_counts, bucket_map


def compare_honest_and_adv(honest_indices, adv_indices, bucket_map):
    honest_indices_map = {}
    adv_indices_map = {}

    for i in range(num_txs):
        honest_indices_map[honest_indices[i]] = i
        adv_indices_map[adv_indices[i]] = i

    # Counters by bucket
    counters = [0 for _ in range(n//2+1)]
    for first in range(num_txs):
        for second in range(first+1, num_txs):
            map_index = bucket_map[(first,second)]
            if honest_indices_map[first] < honest_indices_map[second] and adv_indices_map[second] < adv_indices_map[first]:
                counters[map_index] += 1
            elif honest_indices_map[second] < honest_indices_map[first] and adv_indices_map[first] < adv_indices_map[second]:
                counters[map_index] += 1

    return counters


def initialize():
    send_times, honest_node_orderings = generate_all_timestamps()
    bucket_counts, bucket_map = setup_buckets(honest_node_orderings)
    print("Bucket counts: ", bucket_counts)
    return send_times, honest_node_orderings, bucket_counts, bucket_map



def runonce(ratio, gammas):
    global gen_param
    global network_param
    global actual_f


    network_param = gen_param * ratio
    
    send_times, honest_node_orderings, bucket_counts, bucket_map = initialize()


    f_gamma = [math.floor((0.5 * n*(2*gamma - 1)) / (gamma + 1) - 0.01) for gamma in gammas]

    step_time = 100

    threshs = [0 for _ in range(len(gammas))]
    for index in range(len(gammas)):
        gamma, fval = gammas[index], f_gamma[index]
        threshs[index] = n * (1-gamma) + gamma * fval + 1

    solid_thres = [n - 2*f for f in f_gamma]


    counts = {}
    unique_txs = set()

    honest_node_orderings_bkp = copy.deepcopy(honest_node_orderings)
    honest_node_orderings_step = copy.deepcopy(honest_node_orderings)

    for node in range(len(honest_node_orderings_step)):
        for tx in range(len(honest_node_orderings_step[node])):
            if honest_node_orderings_step[node][tx] <= step_time:
                if tx not in counts:
                    counts[tx] = 1
                    unique_txs.add(tx)
                else:
                    counts[tx] += 1
            else:
                honest_node_orderings_step[node][tx] = math.inf

    num_unique_txs = len(counts)
    print("Unique Txs ", num_unique_txs)

    finalized_nums = [0 for i in range(len(threshs))]
    edges_done_nums = [0 for i in range(len(threshs))]

    for tx in counts:
        for i in range(len(threshs)):
            if counts[tx] >= threshs[i]:
                edges_done_nums[i] += 1

        for i in range(len(solid_thres)):
            if counts[tx] >= solid_thres[i]:
                finalized_nums[i] += 1


    np_full_node_orderings = np.transpose(honest_node_orderings_bkp)

    index_pairs = np.array(list(permutations(unique_txs, 2)))
    full_index_pairs = np.array(list(permutations(range(np_full_node_orderings.shape[0]), 2)))
    # print(len(index_pairs), len(full_index_pairs))

    fullweights = np.count_nonzero((np_full_node_orderings[full_index_pairs[:,0],:] < np_full_node_orderings[full_index_pairs[:,1],:]), axis=1)


    full_weights_map = {}


    for i in range(len(full_index_pairs)):
        first, second = full_index_pairs[i]
        full_weights_map[(first,second)] = fullweights[i]


    revs = [0 for _ in gammas]
    revs_txs = [set() for _ in gammas]

    edges_gamma = [set() for _ in gammas]
    fulledges_gamma = [set() for _ in gammas]
    num_edges = [0 for _ in gammas]

    for index in range(len(gammas)):
        gamma, fval = gammas[index], f_gamma[index]
        threshold = n * (1-gamma) + gamma * fval + 1

        edges = set()
        fulledges = set()

        np_node_orderings = np.transpose(copy.deepcopy(honest_node_orderings[:n-fval][:]))

        weights = np.count_nonzero((np_node_orderings[index_pairs[:,0],:] < np_node_orderings[index_pairs[:,1],:]), axis=1)
        weights_map = {}

        for i in range(len(index_pairs)):
            first, second = index_pairs[i]
            weights_map[(first,second)] = weights[i]


        for first, second in index_pairs:
            if weights_map[(first,second)] >= threshold and weights_map[(second, first)] >= threshold:
                if weights_map[(first,second)] >= weights_map[(second,first)]:
                    if (second, first) not in edges:
                        edges.add((first,second))
                else:
                    edges.add((second,first))
            elif weights_map[(first,second)] >= threshold:
                    edges.add((first,second))
            elif weights_map[(second,first)] >= threshold:
                    edges.add((second,first))
        
        edges_gamma[index] = edges


        for first, second in full_index_pairs:
            if full_weights_map[(first,second)] >= threshold and full_weights_map[(second, first)] >= threshold:
                if full_weights_map[(first,second)] >= full_weights_map[(second,first)]:
                    if (second, first) not in fulledges:
                        fulledges.add((first,second))
                else:
                    fulledges.add((second,first))
            elif full_weights_map[(first,second)] >= threshold:
                    fulledges.add((first,second))
            elif full_weights_map[(second,first)] >= threshold:
                    fulledges.add((second,first))

        fulledges_gamma[index] = fulledges


        num_edges[index] = len(edges)


        for (a,b) in edges:
            if (b,a) in fulledges:

                revs[index] += 1
                revs_txs[index].add(a)
                revs_txs[index].add(b)


        revs_txs[index] = len(revs_txs[index])

    return num_unique_txs, edges_done_nums, finalized_nums, num_edges, revs, revs_txs
    # Unique Txs, Shaded, Solid, Edges, Reversed Edges, Reversed Txs



def main():
    gammas = [0.51, 0.6, 0.8, 1]
    ratio = 100

    running_num_unique_txs = 0
    running_edges_done_nums = [0 for _ in gammas]
    running_finalized_nums = [0 for _ in gammas]

    running_num_edges = [0 for _ in gammas]
    running_revs = [0 for _ in gammas]
    running_rev_txs = [0 for _ in gammas]


    for i in range(100):
        num_unique_txs, edges_done_nums, finalized_nums, num_edges, revs, rev_txs = runonce(ratio, gammas)

        running_num_unique_txs += num_unique_txs
        
        running_edges_done_nums = [sum(x) for x in zip(running_edges_done_nums, edges_done_nums)]
        running_finalized_nums = [sum(x) for x in zip(running_finalized_nums, finalized_nums)]
        
        running_num_edges = [sum(x) for x in zip(running_num_edges, num_edges)]
        running_revs = [sum(x) for x in zip(running_revs, revs)]

        running_rev_txs = [sum(x) for x in zip(running_rev_txs, rev_txs)]


    print("r", ratio)
    print("Total Txs ", running_num_unique_txs)

    print("Shaded Txs ", running_edges_done_nums)
    print("Solid Txs ", running_finalized_nums)

    print("Edges Added ", running_num_edges)
    print("Edges Reversed ", running_revs)
    print("Txs Reversed ", running_rev_txs)



if __name__ == '__main__':
    main()
