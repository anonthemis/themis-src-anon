# Comparison to Send ordering

import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
from cycler import cycler
import networkx as nx


num_txs = 100
n = 5                    # number of nodes
f = 1

# Parameters
gen_dist = "uniform"
gen_param = 30
network_dist = "uniform"
network_param = 450


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
    node_orderings = send_times + network_delays

    return send_times, node_orderings


# def receive_fairness(gamma, send_times, node_orderings):
#     counter = 0
#     # Probably can optimize
#     for first in range(len(send_times)):
#         for second in range(first+1, len(send_times)):
#             first_counter = 0
#             second_counter = 0
#             for node in node_orderings:
#                 if node[first] < node[second]:
#                     first_counter += 1
#                 elif node[second] < node[first]:
#                     second_counter += 1
    
#             if send_times[first] < send_times[second] and first_counter >= (gamma * n):
#                 counter += 1
#             elif send_times[second] < send_times[first] and second_counter >= (gamma * n):
#                 counter += 1
#     return counter


def fast_aequitas_graph(gamma, node_orderings):
    print(node_orderings)
    G = nx.DiGraph()
    G.add_nodes_from(list(range(num_txs)))

    threshold = n * (1-gamma) + f + 1
    for first in range(len(node_orderings[0])):
        for second in range(first+1, len(node_orderings[0])):
            first_counter = 0
            second_counter = 0
            for node in node_orderings:
                if node[first] < node[second]:
                    first_counter += 1
                elif node[second] < node[first]:
                    second_counter += 1


            print(first,second, first_counter, second_counter)
            if first_counter >= threshold and second_counter >= threshold:
                if first_counter >= second_counter:
                    G.add_edge(*(first,second))
                else:
                    G.add_edge(*(second,first))
            elif first_counter >= threshold:
                G.add_edge(*(first,second))
            elif second_counter >= threshold:
                G.add_edge(*(second,first))
    # print(G.edges)
    SCC = list(nx.strongly_connected_components(G))
    print(SCC)
    lens = [len(k) for k in SCC]
    print(max(lens), np.median(lens))
    # nx.draw(G)
    # plt.show()



# Ignore
def receive_and_batch_fairness(gammas, send_times, node_orderings):
    graphs = []
    possible_pairs_all_gammas = []

    for gamma in gammas:
        G = nx.DiGraph()
        G.add_nodes_from(list(range(num_txs)))
        graphs.append(G)

        possible_pairs_all_gammas.append(set())

    # Probably can optimize
    for first in range(len(send_times)):
        for second in range(first+1, len(send_times)):
            first_counter = 0
            second_counter = 0
            for node in node_orderings:
                if node[first] < node[second]:
                    first_counter += 1
                elif node[second] < node[first]:
                    second_counter += 1
            
            for index in range(len(gammas)):
                gamma = gammas[index]
                if first_counter >= (gamma * n):
                    graphs[index].add_edge(*(first,second))
                elif second_counter >= (gamma * n):
                    graphs[index].add_edge(*(second,first))


                if send_times[first] < send_times[second] and first_counter >= (gamma * n):
                    possible_pairs_all_gammas[index].add((first,second))
                elif send_times[second] < send_times[first] and second_counter >= (gamma * n):
                    possible_pairs_all_gammas[index].add((second,first))
    # print("Edges ", G.number_of_edges())

    receive_counters = []
    batch_counters = []

    for gamma_index in range(len(gammas)):
        batch_counter = 0

        G = graphs[gamma_index]
        SCC = list(nx.strongly_connected_components(G))
        node_map = {}
        index = 0
        for S in SCC:
            for val in S:
                node_map[val] = index
            index += 1
        for (first, second) in possible_pairs_all_gammas[gamma_index]:
            if node_map[first] != node_map[second]:
                # Not in same cycle. 
                batch_counter += 1
            else:
                # Randomly ordering them has 1/2 probability of being wrong
                # This is naive. Might be some other better strategy for breaking ties
                batch_counter += 0.5

        receive_counters.append(len(possible_pairs_all_gammas[gamma_index]))
        batch_counters.append(batch_counter)
    return receive_counters, batch_counters


#Ignore
def linear_separability(send_times, node_orderings):
    max_times = np.max(node_orderings, axis=0)
    min_times = np.min(node_orderings, axis=0)

    counter = 0
    for first in range(len(send_times)):
        for second in range(first+1, len(send_times)):
            if (send_times[first] < send_times[second] and max_times[first] < min_times[second]):
                counter += 1
            elif (send_times[second] < send_times[first] and max_times[second] < min_times[first]):
                counter += 1
    return counter



def run():
    send_times, node_orderings = generate_all_timestamps()
    fast_aequitas_graph(1,node_orderings)



    # linear_counter = linear_separability(send_times, node_orderings)
    # print(linear_counter)

    
    # receive_counters,batch_counters = receive_and_batch_fairness([1,0.8,0.6,0.51], send_times, node_orderings)
    # print(receive_counters,batch_counters)

    # r2,b2 = receive_and_batch_fairness(0.8, send_times, node_orderings)
    # print(r2,b2)

    # r3,b3 = receive_and_batch_fairness(0.6, send_times, node_orderings)
    # print(r3,b3)

    # r4,b4 = receive_and_batch_fairness(0.51, send_times, node_orderings)
    # print(r4,b4)

    # receive_counters = [r1,r2,r3,r4]
    # batch_counters = [b1,b2,b3,b4]

    # return (linear_counter, receive_counters, batch_counters)


def main():
    global gen_param
    global network_param

    # num_pairs = int((num_txs * (num_txs - 1)) / 2)
    # #ratios = np.concatenate((np.logspace(-2,0,num=6), np.logspace(1,3,num=12)))
    # ratios = np.concatenate((np.logspace(-2,0,num=3), np.logspace(0,3,num=12)))
    # linear, r_one, r_pteight, r_ptsix, r_lowest = [], [], [], [], []
    # b_one, b_pteight, b_ptsix, b_lowest = [], [], [], []

    # for ratio in ratios:
    #     network_param = gen_param * ratio
    #     l, r, b = run()
    #     linear.append(l/num_pairs)
    #     r_one.append(r[0]/num_pairs)
    #     r_pteight.append(r[1]/num_pairs)
    #     r_ptsix.append(r[2]/num_pairs)
    #     r_lowest.append(r[3]/num_pairs)

    #     b_one.append(b[0]/num_pairs)
    #     b_pteight.append(b[1]/num_pairs)
    #     b_ptsix.append(b[2]/num_pairs)
        # b_lowest.append(b[3]/num_pairs)

    run()
    exit(1)


    # Plotting. Ignore
    fig, ax = plt.subplots()

    custom_cycler = (cycler(color=['#377eb8','#ff7f00','#ff7f00','#4daf4a','#4daf4a','#f781bf','#f781bf', "#a65628","#a65628"]) +
                 cycler(linestyle=['solid', 'dotted', 'dashdot','dotted', 'dashdot','dotted', 'dashdot','dotted', 'dashdot' ]) +
                 cycler(linewidth=[1,2,1,2,1,2,1,2,1]))
    ax.set_prop_cycle(custom_cycler)

    plt.plot(ratios, linear)
    
    plt.plot(ratios, r_one)
    plt.plot(ratios, b_one)

    plt.plot(ratios, r_pteight)
    plt.plot(ratios, b_pteight)

    plt.plot(ratios, r_ptsix)
    plt.plot(ratios, b_ptsix)

    plt.plot(ratios, r_lowest)
    plt.plot(ratios, b_lowest)

    plt.xlabel(r'$r=\frac{\textnormal{Mean Network Delay}}{\textnormal{Mean Generation Time}}$', fontsize=12)
    plt.xscale('log')
    plt.ylabel("Fraction of correctly ordered transaction pairs",fontsize=12)
    # plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0)
     
    plt.legend(["Linear Separability", 
        r'Receive-Order-Fairness $\gamma = 1$',r'Batch-Order-Fairness $\gamma = 1$',
        r'Receive-Order-Fairness $\gamma = 0.8$',r'Batch-Order-Fairness $\gamma = 0.8$',
        r'Receive-Order-Fairness $\gamma = 0.6$',r'Batch-Order-Fairness $\gamma = 0.6$',
        r'Receive-Order-Fairness $\gamma = 0.51$',r'Batch-Order-Fairness $\gamma = 0.51$'], fontsize=10)
    plt.savefig('send_comparison.pdf',bbox_inches='tight')

    #plt.show()


if __name__ == '__main__':
    main()
