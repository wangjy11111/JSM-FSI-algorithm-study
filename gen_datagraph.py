import logging
import time

import numpy as np
import networkx as nx
import random
import sys
from itertools import combinations, permutations
from tqdm import tqdm
import common_util as cu

# Generate 100 types of distributions, ensuring the total number of nodes is 10,000
def gen_type_distribution(type_mean, type_std, total_types, total_nodes):
    type_distribution = np.sort(np.random.normal(type_mean, type_std, total_types).astype(int))
    type_distribution = np.clip(type_distribution, 1, None)  # Ensure there are no negative numbers
    type_distribution = type_distribution / type_distribution.sum() * total_nodes  # Normalize to 10,000 nodes
    type_distribution = type_distribution.astype(int)
    
    # Ensure the distribution is exactly 10,000
    difference = total_nodes - type_distribution.sum()
    if difference > 0:
        type_distribution[:difference] += 1
    elif difference < 0:
        type_distribution[:abs(difference)] -= 1
    return type_distribution
def gen_frequent_subgraph(frequent_count, type_rates, subgraph_node_mean, subgraph_node_std):
    frequent_subgraphs = []
    subgraph_node_counts = []
    subgraph_type_distribution = []
    
    for _ in range(frequent_count):
        subgraph_node_count = int(np.clip(np.random.normal(subgraph_node_mean, subgraph_node_std), 3, 20))
        #logging.debug(f"sub count:{subgraph_node_count}")
        subgraph = nx.gnm_random_graph(subgraph_node_count, subgraph_node_count * 2)
        frequent_subgraphs.append(subgraph)
        subgraph_node_counts.append(subgraph_node_count)


    # logging.debug(f"Generated {frequent_count} frequent subgraphs, node counts follow a normal distribution (10, 5^2)")
    
    # 为频繁子图的节点分配类型，基于类型分布的概率进行选择
    for i in range(frequent_count):
        subgraph_node_types = np.random.choice(np.arange(total_types), subgraph_node_counts[i], p=type_rates)
        subgraph_type_distribution.append(subgraph_node_types)
        # 给每个节点添加 'node_type' 属性
        for j, node in enumerate(frequent_subgraphs[i].nodes()):
            frequent_subgraphs[i].nodes[node]['node_type'] = int(subgraph_node_types[j])



    return frequent_subgraphs, subgraph_node_counts, subgraph_type_distribution


def build_FSI_index(frequent_subgraphs, M, N, max_path):
    # Initialize node type to frequent subgraph index
    FSI_node_index = build_FSI_node_index(frequent_subgraphs)

    FSI_linear_index = {}
    FSI_cycle_index = {}
    for subgraph_index, subgraph in enumerate(frequent_subgraphs):  # Traverse each frequent subgraph
        #logging.debug(f"subgraph_index {subgraph_index}: {subgraph.nodes()}")
        build_single_FSI_index("FSI", subgraph_index, subgraph, M, N, max_path, FSI_linear_index, FSI_cycle_index, True)

    return FSI_node_index, FSI_linear_index, FSI_cycle_index

def build_single_FSI_index(tag, subgraph_index, subgraph, M, N, max_path, FSI_linear_index, FSI_cycle_index, is_build_cycle):
    for v in tqdm(subgraph.nodes(), desc=f"build {tag} index, subgraph_index:{subgraph_index}, max_path:{max_path}"):
        build_by_dfs(subgraph_index, subgraph, v, [], M, N, max_path, FSI_linear_index, FSI_cycle_index,
                     is_build_cycle, 0)  # Initial Path is empty

def build_single_SGI_index(subgraph_index, subgraph, M, N, max_path, FSI_linear_index, FSI_cycle_index, is_build_cycle):
    call_times = 0
    nodes_with_dash = {node for node in subgraph.nodes() if str(node).startswith("-")}
    for v in tqdm(nodes_with_dash, desc=f"build SGI index, subgraph_index:{subgraph_index}, max_path:{max_path}"):
        #logging.info(f"v:{v}")
        call_t = build_by_dfs(subgraph_index, subgraph, v, [], M, N, max_path, FSI_linear_index, FSI_cycle_index,
                              is_build_cycle, 0)  # Initial Path is empty
        call_times += call_t
    return call_times

# DFS traversal and build index
def build_by_dfs(subgraph_id, subgraph, v, Path, M, N, max_path, FSI_linear_index, FSI_cycle_index, is_build_cycle,
                 call_times):
    if len(Path) > max_path:
        return call_times
    if v not in Path:  # It means it's not a circle, it's a straight path
        if M >=2 and len(Path) >= 2:
            for m in range(2, min(M + 1, len(Path) + 1)):
                build_linear_index(subgraph_id, subgraph, v, Path, m, FSI_linear_index)  # Construct a linear index
        Path.append(v)  # Add the current node to Path
        for neighbor in subgraph.neighbors(v):  # Traverse all neighbor nodes
            call_times = build_by_dfs(subgraph_id, subgraph, neighbor, Path, M, N, max_path, FSI_linear_index,
                                  FSI_cycle_index, is_build_cycle, call_times)  # Recursively perform DFS
        Path.pop()  # Backtrack, restore Path to the state before the call
        return call_times + 1
    elif is_build_cycle and v == Path[0]:  # Indicates that a cycle has formed
        if N >=3 and len(Path) >= 3:
            for n in range(3, min(N + 1, len(Path) + 1)):
                build_cycle_index(subgraph_id, subgraph, Path, n, FSI_cycle_index)  # Build cyclical index
        return call_times + 1
    return call_times


# Build linear index
def build_linear_index(subgraph_id, subgraph, v, Path, m, FSI_linear_index):
    # Get the subpath of Path (part excluding the first node
    subPath = Path[1:]
    # Generate combinations of length m-2, nodes in the combination need to be arranged in the order of Path
    for group in combinations(subPath, m - 2):
        group = list(group)
        linear_path = [Path[0]] + group + [v]  # Concatenate to form a linear path
        #logging.debug(f"linear_path: {len(linear_path)}, {linear_path}")
        type_linear_path = [subgraph.nodes[x]['node_type'] for x in linear_path]
        type_str = "_".join(map(str, type_linear_path))
        # Record the m linear index of this link to the subgraph
        if m not in FSI_linear_index:
            FSI_linear_index[m] = {}
        if type_str not in FSI_linear_index[m]:
            FSI_linear_index[m][type_str] = {}
        if subgraph_id not in FSI_linear_index[m][type_str]:
            FSI_linear_index[m][type_str][subgraph_id] = set()
        FSI_linear_index[m][type_str][subgraph_id].add(tuple(linear_path))
        # logging.debug(f"Linear index: {type_linear_path} to the index position of frequent subgraph {subgraph_id}")

# 构建环性索引
def build_cycle_index(subgraph_id, subgraph, Path, n, FSI_cycle_index):
    # Get the subpath of Path (part excluding the first node)
    subPath = Path[1:]
    # Generate combinations of length n-2, nodes in the combination need to be arranged in the order of Pat
    for group in combinations(subPath, n - 2):
        group = list(group)
        cycle_path = [Path[0]] + group  # Concatenate to form a circular path
        type_cycle_path = [subgraph.nodes[x]['node_type'] for x in cycle_path]
        type_str = "_".join(map(str, type_cycle_path))
        # Record the n-circular index of this link to the subgraph
        if n not in FSI_cycle_index:
            FSI_cycle_index[n] = {}
        if type_str not in FSI_cycle_index[n]:
            FSI_cycle_index[n][type_str] = {}
        if subgraph_id not in FSI_cycle_index[n][type_str]:
            FSI_cycle_index[n][type_str][subgraph_id] = set()
        FSI_cycle_index[n][type_str][subgraph_id].add(tuple(cycle_path))
        # logging.debug(f"Cycle index: {type_linear_path} to the index position of frequent subgraph {subgraph_id}")


def build_FSI_node_index(frequent_subgraphs):
    # Initialize the index from node type to frequent subgraph
    
    FSI_node_index = {}
    for subgraph_index, subgraph in enumerate(frequent_subgraphs):  # Traverse each frequent subgraph
        # Traverse the node types in the current frequent subgraph
        for node_id in subgraph.nodes():
            node_type = subgraph.nodes[node_id]['node_type']
            if node_type not in FSI_node_index:
                FSI_node_index[node_type] = []
    
            # Record the relative position (node_id) of this node type in which subgraph (subgraph_index)
            FSI_node_index[node_type].append((subgraph_index, node_id))
    
    # Output the index from node type to frequent subgraph
    #for node_type, locations in FSI_node_index.items():
        #logging.debug(f"Node type {node_type} appears at the following locations: {locations}")
    return FSI_node_index
def build_FSO_linear_index(frequent_subgraphs, L):
    index = {}  # Initialize the index
    all_nodes = []  # Store all nodes in the frequent subgraphs
    
    # Traverse all frequent subgraphs
    for subgraph_index, subgraph in enumerate(frequent_subgraphs):
        # Traverse each node in the subgraph
        for node_id in subgraph.nodes():
            node_type = subgraph.nodes[node_id]['node_type']
            # Package the node and its type with the id of the frequent subgraph as a tuple and add it to the array
            all_nodes.append((node_id, node_type, subgraph_index))
    
    # Generate combinational links composed of any m nodes (m ranging from 2 to L)
    for m in range(2, L + 1):
        if m not in index:
            index[m] = {}
        
        # Generate combinations of m elements from all_nodes
        for node_combination in combinations(all_nodes, m):
            # Generate all possible ordered permutations for the combinations
            for permuted_combination in permutations(node_combination):
                # Extract node types
                type_list = [str(node[1]) for node in permuted_combination]  # node[1] is node type
                type_str = "_".join(type_list)  # Concatenate node type strings with underscore

                # Initialize a set to store unique values
                if type_str not in index[m]:
                    index[m][type_str] = set()  # Use a set for deduplication
                
                # Use the combination of node information (including frequent subgraph id) as the value
                val = tuple(node[2] for node in permuted_combination)  # node[0] is node id, node[2] is subgraph ID
                
                # Add the path to the index
                if type_str not in index[m]:
                    index[m][type_str] = []
                index[m][type_str].add(val)
    
    return index
def gen_simplified_graph(G, embeddings, embedding_nodes, embedding_subgraph_map):

    G1 = nx.Graph()  # New simplified graph G1
    embedding_to_aggregate = {}  # Store the aggregate node corresponding to each embedding
    
    # 1. Add all non-embedding nodes and their edges to G1
    for node in G.nodes():
        if node not in embedding_nodes:
            # Add all non-embedding nodes to G1
            G1.add_node(node, **G.nodes[node])  # Copy node attributes
    
    #logging.debug(f"Original graph G node count: {len(G.nodes())}, edge count: {len(G.edges())}")
    # Add edges between non-embedding nodes
    for u, v in G.edges():
        if u not in embedding_nodes and v not in embedding_nodes:
            G1.add_edge(u, v)  # Directly add edges between non-embedding nodes
    
    logging.debug(f"embedding_nodes: {len(embedding_nodes)}")
    #logging.debug(f"Simplified graph G1 node count: {len(G1.nodes())}, edge count: {len(G1.edges())}")

    # 2. Generate an aggregate node for each embedding and add it to G1
    for embedding_id, embedding in enumerate(embeddings):
        # Generate a new aggregate node
        agg_node = str(-1 * (embedding_id+1)).zfill(5)
        #logging.debug(f"agg_node:{agg_node}")
        subgraph_type = int(embedding_subgraph_map[embedding_id])
        G1.add_node(agg_node, node_type=int(-1 * (subgraph_type+1)))
        embedding_to_aggregate[embedding_id] = agg_node  # Record aggregate node

    # 3. Traverse all internal nodes of the embedding, replace edges with the edges of the aggregate node
    for embedding_id, embedding in enumerate(embeddings):
        agg_node = embedding_to_aggregate[embedding_id]
    
        # Traverse each node of the subgraph
        for node in embedding.values():
            # Find all neighbors of this embedding node
            for neighbor in G.neighbors(node):
                # If the neighbor belongs to the embedding, find the corresponding aggregate node
                if neighbor in embedding_nodes:
                    # Find the subgraph where the neighbor is located
                    for other_embedding_id, other_embedding in enumerate(embeddings):
                        if neighbor in other_embedding.values():
                            agg_neighbor = embedding_to_aggregate[other_embedding_id]
                            # Add an edge between the two aggregate nodes
                            if not G1.has_edge(agg_node, agg_neighbor):
                                G1.add_edge(agg_node, agg_neighbor)
                            break
                else:
                    # If neighbor is non-embedding node, add edge between the aggregate and the external node
                    if not G1.has_edge(agg_node, neighbor):
                        G1.add_edge(agg_node, neighbor)
    
    #logging.debug(f"Simplified graph G1 number of nodes: {len(G1.nodes())}, number of edges: {len(G1.edges())}")

    return G1
def build_ori_index(G, M, N, max_path):
    node_index = build_graph_node_index(G)
    linear_index = {}
    build_time = {}
    for i in range(2, max_path + 1):
        linear_this = {}
        cycle_this = {}
        logging.debug(f"nodes: {len(G.nodes)}")
        t1 = time.time()
        build_single_FSI_index("ori", -1, G, M, N, i, linear_this, cycle_this, False)
        t2 = time.time()
        build_time[i] = t2 - t1
        linear_index[i] = linear_this
    return node_index, linear_index, build_time

def build_SGI_index(G, M, N, max_path):
    node_index = build_graph_node_index(G)
    linear_index = {}
    build_time = {}
    for i in range(2, max_path + 1):
        linear_this = {}
        cycle_this = {}
        logging.debug(f"nodes: {len(G.nodes)}")
        t1 = time.time()
        call_times = build_single_SGI_index(-1, G, M, N, i, linear_this, cycle_this, False)
        t2 = time.time()
        logging.info(f"max:{i}, call_times:{call_times}")
        build_time[i] = t2 - t1
        linear_index[i] = linear_this
    return node_index, linear_index, build_time


def build_graph_node_index(G):
    # Initialize the index of the node type to the position in G
    node_index = {}
    for node in G.nodes(data=True):
        node_id = node[0]  # node ID
        node_type = node[1].get('node_type')  # Get node type
        if node_type not in node_index:
            node_index[node_type] = []

        # Add the position of this type of node to the index
        node_index[node_type].append(node_id)
    #for node_type, locations in node_index.items():
        #logging.debug(f"Node type {node_type} appears at the following locations：{locations}")
    return node_index
# Initialize graph G, and add the embedding nodes and corresponding edges to the graph
def gen_original_graph(type_distribution, frequent_subgraphs, embedding_count, subgraph_type_distribution, degree_mean, degree_std, total_nodes):
    G = nx.Graph() # Original graph
    current_node = 0
    embeddings = []
    embedding_nodes = []  # Store embedding nodes
    embedding_subgraph_map = {} # The mapping relationship between embedding and frequent subgraphs
    
    for subgraph_index, subgraph in enumerate(frequent_subgraphs):
        for _ in range(embedding_count):
            node_mapping = {}
            # Assign nodes to each embedding and connect edges
            for i, node_type in enumerate(subgraph_type_distribution[subgraph_index]):
                node_id = str(current_node).zfill(5)  # Format the node number into four digits
                G.add_node(node_id, node_type=int(node_type))
                node_mapping[i] = node_id 
                embedding_nodes.append(node_id)
                current_node += 1
            # Add edges in the subgraph
            for u, v in subgraph.edges():
                G.add_edge(node_mapping[u], node_mapping[v])
            embeddings.append(node_mapping)
            embedding_subgraph_map[len(embeddings)-1] = subgraph_index
    
    #logging.debug(f"Already add embedding node to graph")
    if len(embedding_nodes) >= total_nodes:
        logging.debug(f"embedding_nodes {len(embedding_nodes)} > total_nodes {total_nodes}")
        exit()

    # Set the type of non-embedding nodes to ensure that the distribution of 10,000 node types conforms to (100, 10^2)
    remaining_type_distribution = type_distribution.copy()
    
    # Subtract the assigned embedding node types
    for i in range(frequent_count):
        for t in subgraph_type_distribution[i]:
            remaining_type_distribution[t] -= embedding_count
    
    # Ensure that the remaining node types also follow the overall distribution
    remaining_node_types = []
    for t in range(total_types):
        remaining_node_types.extend([t] * remaining_type_distribution[t])
    
    random.shuffle(remaining_node_types)

    # Add non-embedding nodes to the graph
    for remaining_node_type in remaining_node_types:
        node_id = str(current_node).zfill(5)  # Format the node number as a four-digit number
        G.add_node(node_id, node_type=int(remaining_node_type))
        current_node += 1
    
    # Traverse all nodes, add edges, ensure that the degree distribution conforms to the normal distribution of (4,2^2)
    for node in G.nodes():
        # Calculate the target degree
        degree = int(np.clip(np.random.normal(degree_mean, degree_std), 1, 10))
        
        # Get the number of edges currently established for this node
        current_degree = len(list(G.neighbors(node)))
        
        # Calculate the edge counts to ensure that the total number of edges conforms to the target degree
        remaining_degree = degree - current_degree
        
        # If the number of edges to be established is greater than 0, continue to add edges
        if remaining_degree > 0:
            # Obtain all potential neighbors (remove oneself and connected nodes)
            potential_edges = [n for n in G.nodes() if n != node and not G.has_edge(node, n)]
            
            # If the number of potential neighbors is less than remaining_degree, prevent it from exceeding the range
            if remaining_degree > len(potential_edges):
                remaining_degree = len(potential_edges)
            
            # Randomly shuffle potential neighbors and select remaining_degree neighbors
            random.shuffle(potential_edges)
            neighbors = random.sample(potential_edges, remaining_degree)
            
            # Establish edges between this node and new neighbors
            for neighbor in neighbors:
                G.add_edge(node, neighbor)
    return G, embeddings, embedding_nodes, embedding_subgraph_map

def run(directory, total_nodes, type_distribution, frequent_subgraphs, subgraph_type_distribution, embedding_count, degree_mean,
        degree_std, max_path, FSI_max_path, M, N):

    name_prefix = str(M) + "_" + str(N) + "_" + str(embedding_count)

    logging.debug("build FSI index")
    t1 = time.time()
    FSI_node_index, FSI_linear_index, FSI_cycle_index = build_FSI_index(frequent_subgraphs, M, N, FSI_max_path)
    t2 = time.time()
    FSI_build_time = t2 - t1
    cu.save_single_obj(directory, name_prefix + "_FSIn.pkl", FSI_node_index)
    FSI_node_index.clear()
    cu.save_single_obj(directory, name_prefix + "_FSIl.pkl", FSI_linear_index)
    FSI_linear_index.clear()
    cu.save_single_obj(directory, name_prefix + "_FSIc.pkl", FSI_cycle_index)
    FSI_cycle_index.clear()
    cu.save_single_obj(directory, name_prefix + "_FSIt.pkl", FSI_build_time)

    logging.debug("generate G")
    G, embeddings, embedding_nodes, embedding_subgraph_map =(
        gen_original_graph(type_distribution, frequent_subgraphs, embedding_count, subgraph_type_distribution,
                           degree_mean, degree_std, total_nodes))

    cu.save_multi_graph(directory, name_prefix + "_freq", frequent_subgraphs)

    logging.debug("generate G1")
    G1 = gen_simplified_graph(G, embeddings, embedding_nodes, embedding_subgraph_map)
    node_types = {G1.nodes[node]['node_type'] for node in G1.nodes()}
    logging.debug(f"node_type:{sorted(node_types)}")

    logging.debug("build G1 index")
    G1_node_index, G1_linear_index, G1_build_time = build_SGI_index(G1, M, N, max_path)
    logging.info(f"G1_build_time:{G1_build_time}")

    cu.save_single_graph(directory, name_prefix + "_simp.graphml", G1)
    G1.clear()
    cu.save_single_obj(directory, name_prefix + "_simpn.pkl", G1_node_index)
    G1_node_index.clear()
    cu.save_multi_obj(directory, name_prefix, "simpl.pkl", G1_linear_index)
    G1_linear_index.clear()
    cu.save_single_obj(directory, name_prefix + "_simpt.pkl", G1_build_time)

    logging.debug("build G index")
    G_node_index, G_linear_index, G_build_time = build_ori_index(G, M, N, max_path)
    logging.info(f"G1_build_time:{G_build_time}")

    cu.save_single_graph(directory, name_prefix + "_ori.graphml", G)
    G.clear()
    cu.save_single_obj(directory, name_prefix + "_orin.pkl", G_node_index)
    G_node_index.clear()
    cu.save_multi_obj(directory, name_prefix, "oril.pkl", G_linear_index)
    G_linear_index.clear()
    cu.save_single_obj(directory, name_prefix + "_orit.pkl", G_build_time)

#exit()
# Set random seed to ensure reproducibility
np.random.seed(42)
random.seed(42)
cu.custom_logging()

is_test = False

# Parameter Definition
total_nodes = 10000  # Total number of nodes
total_types = 100 # Total number of node types
#frequent_count = 3  # Number of frequent subgraphs generated
#embedding_count = 10  # Number of embeddings per frequent subgraph
degree_mean, degree_std = 3, 2  # Distribution of degrees of general nodes
type_mean, type_std = 100, 50  # Normal distribution (100, 50^2) of node type distribution
subgraph_node_mean, subgraph_node_std = 10, 5  # Distribution of the number of nodes in frequent subgraphs
#max_path = 4 # Establish a multi-hop index within the longest how large path
FSI_max_path = 10

if is_test == True:
    #=================================
    # 参数定义
    total_nodes = 200  # Total number of nodes
    total_types = 10 # Total number of node types
    #frequent_count = 3  # Number of frequent subgraphs generated
    #embedding_count = 10  # Number of embeddings per frequent subgraph
    degree_mean, degree_std = 3, 2  # Distribution of degrees of general nodes
    type_mean, type_std = 10, 5  # Normal distribution (10, 5^2) of node type distribution
    subgraph_node_mean, subgraph_node_std = 5, 3  # Distribution of the number of nodes in frequent subgraphs
    #max_path = 4 # Establish a multi-hop index within the longest how large path
    FSI_max_path = 10
    #=================================


#exit()

dir_in = str(sys.argv[1]) + "/"
max_path = int(sys.argv[2])

#logging.debug("Type distribution has been generated, conforming to the normal distribution (100, 50^2)")
type_distribution = gen_type_distribution(type_mean, type_std, total_types, total_nodes)
logging.debug(f"len type:{len(type_distribution)}, sum type:{sum(type_distribution)}")
#cu.plt_his(type_distribution)

frequent_count = 20
if is_test == True:
    #=================================
    frequent_count = 3
    #=================================
# Generate frequent subgraphs, the number of nodes in each subgraph conforms to the normal distribution (10, 5^2)
frequent_subgraphs, subgraph_node_counts, subgraph_type_distribution\
    = gen_frequent_subgraph(frequent_count, type_distribution/ total_nodes, subgraph_node_mean, subgraph_node_std)
 
M = 3
N = 3

y = np.array(range(1, 21, 2))
for embedding_count in y:
    run(dir_in, total_nodes, type_distribution, frequent_subgraphs, subgraph_type_distribution, embedding_count,
        degree_mean, degree_std, max_path, FSI_max_path, M, N)

    #cu.save_info(dir_in, G, G1, frequent_subgraphs, FSI_node_index, FSI_linear_index, FSI_cycle_index,
    #                      FSI_build_time, G_node_index, G_linear_index, G_build_time, G1_node_index,
    #                      G1_linear_index, G1_build_time,
    #                      str(M) + "_" + str(N) + "_" + str(embedding_count))

