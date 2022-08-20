import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
from cycler import cycler
import networkx as nx
import copy
from itertools import combinations, permutations
import random

num_txs = 1000
n = 101        # number of nodes
f = 25
actual_f = 2

# Parameters
gen_dist = "exponential"
gen_param = 100
network_dist = "exponential"
network_param = 10000


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

    # honest_node_orderings = copy.deepcopy(node_orderings)

    # # Reverse orderings
    # for node in range(actual_f):
    #     sorts = np.sort(node_orderings[node]) 
    #     matches = {}
    #     for i in range(num_txs):
    #         if sorts[i] in matches:
    #             print("Failure")
    #         matches[sorts[i]] = sorts[num_txs-i-1]
    #     for i in range(num_txs):
    #         node_orderings[node][i] = matches[node_orderings[node][i]] 

    return send_times, honest_node_orderings #, node_orderings


def reverse_orderings(honest_node_orderings):
    node_orderings = copy.deepcopy(honest_node_orderings)
    # Reverse orderings
    for node in range(actual_f):
        sorts = np.sort(node_orderings[node]) 
        matches = {}
        for i in range(num_txs):
            if sorts[i] in matches:
                print("Failure")
            matches[sorts[i]] = sorts[num_txs-i-1]
        for i in range(num_txs):
            node_orderings[node][i] = matches[node_orderings[node][i]] 

    return node_orderings




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









def median_timestamp_protocol(node_orderings):
    medians = np.median(node_orderings[:n-f], axis=0)
    sort_indices = np.argsort(medians)
    # print(sort_indices)
    return sort_indices




def aequitas_protocol(gammas, node_orderings):
    graphs = []
    possible_pairs_all_gammas = []

    for gamma in gammas:
        G = nx.DiGraph()
        G.add_nodes_from(list(range(num_txs)))
        graphs.append(G)

        possible_pairs_all_gammas.append(set())


    # Reshape so now first index is tx instead of node
    np_node_orderings = np.transpose(node_orderings)

    # Pairs of txs
    pair_indices = np.array(list(permutations(range(np_node_orderings.shape[0]), 2)))
    
    # count number of times tx < tx'
    counts = np.count_nonzero((np_node_orderings[pair_indices[:,0],:] < np_node_orderings[pair_indices[:,1],:]), axis=1)

    for index in range(len(gammas)):
        gamma = gammas[index]
        order_indices = np.where(counts >= gamma * n - 2*f)

        for first,second in pair_indices[order_indices]:
            graphs[index].add_edge(*(first,second))


    orderings_per_gamma = []

    for gamma_index in range(len(gammas)):
        G = graphs[gamma_index]
        SCC = list(nx.strongly_connected_components(G))
        # print(SCC)
        H = nx.algorithms.components.condensation(G, SCC)

        ordering = nx.algorithms.dag.topological_sort(H)
        ordering_list = list(ordering)
        new_order_list = []
 
        for i in range(len(ordering_list)):
            ordering_list[i] = SCC[ordering_list[i]]
    
        for s in ordering_list:
            for val in sorted(s):
                new_order_list.append(val)

        orderings_per_gamma.append(new_order_list)

    # print(orderings_per_gamma)
    return orderings_per_gamma






def themis_protocol(gammas, node_orderings):
    graphs = []
    possible_pairs_all_gammas = []

    for gamma in gammas:
        G = nx.DiGraph()
        G.add_nodes_from(list(range(num_txs)))
        graphs.append(G)

        possible_pairs_all_gammas.append(set())


    # Reshape so now first index is tx instead of node
    np_node_orderings = np.transpose(node_orderings)

    # Pairs of txs
    pair_indices = np.array(list(permutations(range(np_node_orderings.shape[0]), 2)))
    
    # count number of times tx < tx'
    counts = np.count_nonzero((np_node_orderings[pair_indices[:,0],:] < np_node_orderings[pair_indices[:,1],:]), axis=1)

    for index in range(len(gammas)):
        gamma = gammas[index]
        threshold = n * (1-gamma) + f + 1
        total_counts = {}

        for i in range(len(pair_indices)):
            first, second = pair_indices[i]
            total_counts[(first,second)] = counts[i]
        
        for first, second in pair_indices:
            assert(total_counts[(first,second)] + total_counts[(second,first)] == n)
            if total_counts[(first,second)] >= threshold and total_counts[(second, first)] >= threshold:
                if total_counts[(first,second)] >= total_counts[(second,first)]:
                    graphs[index].add_edge(*(first,second))
                else:
                    graphs[index].add_edge(*(second,first))
            elif total_counts[(first,second)] >= threshold:
                graphs[index].add_edge(*(first,second))
            elif total_counts[(second,first)] >= threshold:
                graphs[index].add_edge(*(second,first))
            else:
                # This needs to be fixed since for small gamma, the bound on f is also smaller
                print("Error")
                print(threshold, total_counts[(first,second)], total_counts[(second,first)])


    orderings_per_gamma = []

    for gamma_index in range(len(gammas)):
        G = graphs[gamma_index]
        SCC = list(nx.strongly_connected_components(G))
        # print(SCC)
        H = nx.algorithms.components.condensation(G, SCC)

        ordering = nx.algorithms.dag.topological_sort(H)
        ordering_list = list(ordering)
        new_order_list = []
 
        for i in range(len(ordering_list)):
            ordering_list[i] = SCC[ordering_list[i]]
    
        for s in ordering_list:
            for val in sorted(s):
                new_order_list.append(val)

        orderings_per_gamma.append(new_order_list)

    # print(orderings_per_gamma)
    return orderings_per_gamma







def initialize():
    send_times, honest_node_orderings = generate_all_timestamps()
    bucket_counts, bucket_map = setup_buckets(honest_node_orderings)
    print("Bucket counts: ", bucket_counts)
    return send_times, honest_node_orderings, bucket_counts, bucket_map


def run(send_times, honest_node_orderings, node_orderings, bucket_counts, bucket_map):
    # Median timestamp protocol
    honest_indices_median = median_timestamp_protocol(honest_node_orderings)
    adv_indices_median = median_timestamp_protocol(node_orderings)

    median_counters = compare_honest_and_adv(honest_indices_median, adv_indices_median, bucket_map)
    print("Median")
    print(median_counters)



    # Aequitas
    print("Aequitas")
    gamma_honest_indices_aequitas = aequitas_protocol([1], honest_node_orderings)
    gamma_adv_indices_aequitas = aequitas_protocol([1], node_orderings)

    aequitas_counters = []
    for i in range(len(gamma_honest_indices_aequitas)):
        aequitas_counter = compare_honest_and_adv(gamma_honest_indices_aequitas[i], gamma_adv_indices_aequitas[i], bucket_map)
        print(aequitas_counter)
        aequitas_counters.append(aequitas_counter)

    print('Themis')
    gamma_honest_indices_themis = themis_protocol([1], honest_node_orderings)
    gamma_adv_indices_themis = themis_protocol([1], node_orderings)

    themis_counters = []
    for i in range(len(gamma_honest_indices_themis)):
        themis_counter = compare_honest_and_adv(gamma_honest_indices_themis[i], gamma_adv_indices_themis[i], bucket_map)
        print(themis_counter)
        themis_counters.append(themis_counter)
     
    return bucket_counts, median_counters, aequitas_counters, themis_counters



def main():
    global gen_param
    global network_param
    global actual_f

    fig, ax = plt.subplots()

    # custom_cycler = (cycler(color=['#377eb8','#ff7f00','#4daf4a','#f781bf']) +
    #              cycler(hatch=['/', '*', '+', '|'])
    #              )
    # custom_cycler = (cycler(color=['#377eb8','#377eb8','#ff7f00','#ff7f00']) +
    #              cycler(hatch=['/', '/','/','/'])
    #              )
    custom_cycler = (cycler(color=['#377eb8','#377eb8','#377eb8', '#ff7f00','#ff7f00','#ff7f00']) +
                 cycler(marker=['.','.','.','.','.','.']) + 
                 cycler(linestyle=['solid', 'dotted', 'dashdot', 'solid', 'dotted', 'dashdot']))
    ax.set_prop_cycle(custom_cycler)

    # bins = range(0,100,10)
    # histbins = bins#range(num_txs)
   
    # labels = np.array([i for i in range(n//2+1)])
    # honest = [43899, 342, 167, 106, 70, 63, 45, 43, 48, 33, 34]
    # median = [0, 0, 0, 0, 1, 0, 0, 1, 8, 6, 7]
    # aequitas = [0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 8]
    # themis = [0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 8]



    # plt.bar(labels-0.4,honest, alpha=0.75,width=0.20)
    # plt.bar(labels-0.2,median, alpha=0.75,width=0.20)
    # plt.bar(labels+0,aequitas, alpha=0.75,width=0.20)
    # plt.bar(labels+0.2,themis, alpha=0.75,width=0.20)


    # plt.yscale('log')
    # plt.show()



    # exit()


    num_pairs = int((num_txs * (num_txs - 1)) / 2)
    # ratios = np.concatenate((np.logspace(-2,0,num=3), np.logspace(0,3,num=12)))
    ratios = [1,10,100]
    # labels = np.array([i for i in range(n//2+1)])

    labels = np.array(list(range(n,0, -2)))

    ratio = 25
    advs = [5,15,25]#[i for i in range(1,f+1)]
    network_param = gen_param * ratio

    send_times, honest_node_orderings, bucket_counts, bucket_map = initialize()

    medians = []
    themis_all = []


    for adv in advs:
        actual_f = adv
        node_orderings = reverse_orderings(honest_node_orderings)
        bucket_counts, median_counters, aequitas_counters, themis_counters = run(send_times, honest_node_orderings, node_orderings, bucket_counts, bucket_map)
        
        bucket_counts = np.array(bucket_counts)
        median_counters = np.array(median_counters)/bucket_counts
        aequitas_counters = np.array(aequitas_counters)/bucket_counts
        themis_counters = np.array(themis_counters)/bucket_counts
        
        medians.append(median_counters)
        themis_all.append(themis_counters[0])
        # network_param = gen_param * ratio
        # bucket_counts, median_counters, aequitas_counters, themis_counters = run()

        # all_counters = [bucket_counts, median_counters, aequitas_counters[0], themis_counters[0]]
        # print(bucket_counts)
        # print(median_counters)
        # print(aequitas_counters[0])
        # print(themis_counters[0])

        # fig, ax = plt.subplots()

        # custom_cycler = (cycler(color=['#377eb8','#ff7f00','#4daf4a','#f781bf']) +
        #          cycler(hatch=['/', '*', '+', '|'])
        #          )
        # ax.set_prop_cycle(custom_cycler)

        # plt.clear()
        # plt.bar(labels-0.4,bucket_counts, alpha=0.75,width=0.25)
        # plt.bar(labels-0.45,median_counters, alpha=0.75,width=0.30)
        # plt.bar(labels-0.15,aequitas_counters[0], alpha=0.75,width=0.30)
        # # plt.bar(labels+0.15,themis_counters[0], alpha=0.75,width=0.30)
        # plt.xticks(np.arange(1, n+1, step=2))

        # # plt.yscale('log')

        # plt.legend(['Median', 'Aequitas/Themis'], fontsize=15)

        # # bins_labels(bins)

        # plt.xlabel('Txs',fontsize=15)
        # plt.show()
    
    plt.plot(labels, medians[0])
    plt.plot(labels, medians[1])
    plt.plot(labels, medians[2])

    plt.plot(labels, themis_all[0])
    plt.plot(labels, themis_all[1])
    plt.plot(labels, themis_all[2])

    # labels = labels[25:] 
    # plt.bar(labels-0.45,medians[0][25:], alpha=0.75,width=0.20)
    # plt.bar(labels-0.23,medians[1][25:], alpha=0.75,width=0.20, hatch='//')

    # plt.bar(labels+0.02,themis_all[0][25:], alpha=0.75,width=0.20)

    # plt.bar(labels+0.25,themis_all[1][25:], alpha=0.75,width=0.20, hatch='//')

    # plt.bar(labels+0.15,themis_counters[0], alpha=0.75,width=0.30)
    # plt.xticks(np.arange(1, 27, step=2))

    # plt.yscale('log')

    # plt.legend(['Median, actual\_f = 2', 'Median, actual\_f = 5', 'Aequitas/Themis, actual\_f = 2', 'Aequitas/Themis, actual\_f = 5'], fontsize=12)

    # bins_labels(bins)

    # plt.xlabel(r"$\left|\textnormal{\#}[Recv(\textnormal{tx}) < Recv(\textnormal{tx}')]-\textnormal{\#}[Recv(\textnormal{tx}') < Recv(\textnormal{tx})]\right|$",fontsize=15)
    plt.xlabel(r"$\textnormal{Dist}(\textnormal{tx}, \textnormal{tx}')$", fontsize=15)
    plt.ylabel(r"Fraction of transaction pairs $(\textnormal{tx}, \textnormal{tx}') \textnormal{ reversed}$",fontsize=15)
    plt.title(r'$n=101, f=25$', fontsize=15)

    plt.legend(['Median, actual\_f = 5', 'Median, actual\_f = 15', 'Median, actual\_f = 25', 'Aequitas/Themis, actual\_f = 5', 'Aequitas/Themis, actual\_f = 15', 'Aequitas/Themis, actual\_f = 25'], fontsize=12)
    plt.xscale('log')
    plt.xticks(np.arange(1, n+1, step=10))


    # plt.show()
    plt.savefig('adv_reorder.pdf',bbox_inches='tight')

        # plt.ylabel("Number of transactions",fontsize=15)


if __name__ == '__main__':
    main()
