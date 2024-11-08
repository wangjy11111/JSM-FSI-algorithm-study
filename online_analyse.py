import numpy as np
import networkx as nx
from networkx.algorithms.isomorphism import GraphMatcher
import random
import time
import logging
import copy
import itertools
import sys
from tqdm import tqdm

import common_util as cu
import plt_util

def gen_querys(G, query_num, query_mean, query_std, degree_mean, degree_std):
    query_graphs = []
    for _ in range(query_num):
        # Generate the number of nodes following a normal distribution N(5, 2^2), limited to a range of 3 to 7.
        num_nodes = int(np.clip(np.random.normal(query_mean, query_std), 3, 7))
    
        # Randomly select nodes from G and create a query graph
        query_graph = nx.Graph()
        selected_nodes = random.choices(list(G.nodes()), k=num_nodes)  # Nodes can be selected multiple times

        # Used to track visited nodes
        added_nodes = []

        # Add nodes to the query graph and establish edges
        for i, original_node in enumerate(selected_nodes):
            # Renumber to consecutive integers starting from 0
            new_node_id = i  # 新的节点 ID
            query_graph.add_node(new_node_id, node_type=G.nodes[original_node]['node_type'])
            #query_graph.add_node(new_node_id, node_type=random.choice([74, 90, 98, 93, 90, 68, 92, 99, 71]))

            # Check if there is an edge between the current node and the visited nodes
            if added_nodes:  # Ensure that the visited nodes list is not empty
                # Check if there is an edge between the current node and the visited nodes
                connected = any(query_graph.has_edge(new_node_id, target_node) for target_node in added_nodes)

                if not connected:  # If there is no connection
                    # Randomly select a visited node to connect
                    target_node = random.choice(added_nodes)
                    query_graph.add_edge(new_node_id, target_node)  # 添加边

            # Add the current node to the list of visited nodes
            added_nodes.append(new_node_id)

        # Generate the required degree for each node and add edges
        node_degrees = {}
        
        for node in query_graph.nodes():
            desired_degree = int(np.clip(np.random.normal(degree_mean, degree_std), 1, num_nodes - 1))
            node_degrees[node] = desired_degree
    
        # Add edges to meet the degree requirements for each node
        for node in query_graph.nodes():
            current_degree = query_graph.degree(node)
            while current_degree < node_degrees[node]:
                # Randomly select a target node, ensuring the target node is different and has not yet met its degree requirement
                potential_targets = [
                    n for n in query_graph.nodes() 
                    if n != node 
                    and query_graph.degree(n) < node_degrees[n]
                    and not query_graph.has_edge(node, n)  # Check if the edge already exists
                ]
                if not potential_targets:
                    break  # If no suitable target is found, stop adding edges
                
                target_node = random.choice(potential_targets)
                query_graph.add_edge(node, target_node)
                current_degree += 1
    
        node_types_array = [query_graph.nodes[node]['node_type'] for node in query_graph.nodes()]
        logging.info(f"query graph type: {node_types_array}")
        query_graphs.append(query_graph)
    return query_graphs

def match_by_ORI(start_time, max_query_time, query, M, N, G, G_node_index, G_linear_index, max_matches):
    return do_ORI2(start_time, max_query_time, query, G, G_linear_index, max_matches)

def do_ORI(query, linears, linears_to_nodes, G, G_node_index, G_linear_index, max_matches):
    ORI_pos = set()
    ORI_linear_cand = get_ORI_cand(linears, G_linear_index)
    logging.debug(f"size of ORI_linear_cand is:{sum(len(cands) for cands in ORI_linear_cand.values())}")

    if len(ORI_linear_cand) > 0:
        type_str, cands = next(iter(ORI_linear_cand.items()))
        query_arr = list(linears_to_nodes[type_str])
        query_start_node = query_arr[0]
        logging.debug(f"before ORI_hip_match:{cands}")
        ORI_pos = ORI_hip_match(-1, G, query, G_linear_index, type_str, cands, query_start_node, 10, max_matches)

    # Only one type_str needs to be checked, as the other type_strs are guaranteed to be connected to it, so there is no need for repeated checks
    return ORI_pos

def time_exceed(start_time, max_run_time):
    if time.time() - start_time > max_run_time:
        logging.info("exceed")
        return True
    return False

def do_ORI2(start_time, max_query_time, query, G, G_linear_index, max_matches):
    ORI_pos = set()
    ORI_linear_cand = get_ORI_cand2(query, G_linear_index)
    logging.debug(f"size of ORI_linear_cand is:{sum(len(cands) for cands in ORI_linear_cand.values())}")
    if time_exceed(start_time, max_query_time):
        return ORI_pos

    if len(ORI_linear_cand) > 0:
        edge_str, cands = next(iter(ORI_linear_cand.items()))

        first_node_index = edge_str.find('_')
        first_node = int(edge_str[:first_node_index])

        logging.debug(f"before ORI_hip_match:{cands}")
        ORI_pos = ORI_hip_match(start_time, max_query_time, -1, G, query, G_linear_index, edge_str, cands, first_node,
                                10, max_matches)

    # Only one type_str needs to be checked, as the other type_strs are guaranteed to be connected to it, so there is no need for repeated checks
    return ORI_pos

def get_single_linear_cand(linears, G_index, Gs_linear_index):
    cand_type2list = {}
    for m, linear in tqdm(linears.items(), desc="Processing linears"):
        if m not in Gs_linear_index:
            continue
        logging.debug(f"Gs_linear_index keys:{Gs_linear_index[m].keys()}")
        for elem_id, elem in enumerate(linear):  # Iterate through each frequent subgraph
            type_str = "_".join(map(str, elem))
            if type_str not in Gs_linear_index[m]:
                logging.debug(f"no type_str:{type_str} found")
                return {}
            if G_index not in Gs_linear_index[m][type_str]:
                logging.debug(f"G_index not exist, G_index:{G_index}, type_str:{type_str}")
                return {}
            else:
                logging.debug(f"G_index exist, G_index:{G_index}, type_str:{type_str}")
            if not mix_to_linear_cand(cand_type2list, type_str, Gs_linear_index[m][type_str][G_index]):
                logging.debug(f"mix_to_linear_cand fail, G_index:{G_index}, type_str:{type_str}")
                return {}
            else:
                logging.debug(f"mix_to_linear_cand succ, G_index:{G_index}, type_str:{type_str}")
    return cand_type2list

def get_linear_cand(linears, Gs_linear_index):
    cand_type2list = {}
    logging.debug(f"len linears:{len(linears)}")
    for m, linear in linears.items():
        if m not in Gs_linear_index:
            continue
        logging.debug(f"linear size:{len(linear)}")
        for elem_id, elem in enumerate(linear):  # Iterate through each frequent subgraph
            type_str = "_".join(map(str, elem))
            if type_str in Gs_linear_index[m]:
                cand_type2list[type_str] = set(Gs_linear_index[m][type_str].keys())
    embeddings_elem = set()
    if len(cand_type2list) > 0:
        embeddings_elem = set.intersection(*cand_type2list.values())
    return embeddings_elem
def get_ORI_cand2(query, Gs_linear_index):
    cand_type2list = {}
    for u, v in query.edges():
        edge_str = str(u) + "_" + str(v)
        type_str = str(query.nodes[u]['node_type']) + "_" + str(query.nodes[v]['node_type'])
        if type_str not in Gs_linear_index[2]:
            logging.debug(f"no type_str:{type_str} found")
            return {}
        #logging.info(f"cand:{Gs_linear_index[2][type_str][-1]}")
        cand_type2list[edge_str] = Gs_linear_index[2][type_str][-1]

        if not mix_to_linear_cand(cand_type2list, type_str, Gs_linear_index[2][type_str][-1]):
            logging.debug(f"mix_to_linear_cand fail, type_str:{type_str}")
            return {}
        else:
            logging.debug(f"mix_to_linear_cand succ, type_str:{type_str}")
    return cand_type2list

def get_ORI_cand(linears, Gs_linear_index):
    embeddings_elem = get_single_linear_cand(linears, -1, Gs_linear_index)
    return embeddings_elem

def get_cycle_cand(cycles, Gs_cycle_index):
    cand_type2list = {}
    for m, cycle in cycles.items():
        if m not in Gs_cycle_index:
            continue
        logging.debug(f"cycle size:{len(cycle)}")
        for elem_id, elem in enumerate(cycle):  # Iterate through each frequent subgraph
            type_str = "_".join(map(str, elem))
            if type_str in Gs_cycle_index[m]:
                cand_type2list[type_str] = set(Gs_cycle_index[m][type_str].keys())
    embeddings_elem = set()
    if len(cand_type2list) > 0:
        embeddings_elem = set.intersection(*cand_type2list.values())
    return embeddings_elem

def mix_to_linear_cand(cand_type2list, val_type_str, vals_in):
    #logging.debug(f"vals_in:{vals_in}")
    vals = vals_in.copy()
    if len(cand_type2list) == 0:
        cand_type2list[val_type_str] = vals
        return True

    first_val_index = val_type_str.find('_')
    last_val_index = val_type_str.rfind('_')
    prefix_val_str = val_type_str[:first_val_index]
    suffix_val_str = val_type_str[last_val_index + 1:]
    # Iterate through cand and trim elements in vals_in
    for cand_type, cand_list in cand_type2list.items():
        first_index = cand_type.find('_')
        last_index = cand_type.rfind('_')

        prefix_cand_str = cand_type[:first_index]
        suffix_cand_str = cand_type[last_index+1:]
    
        #logging.debug(f"filter val, prefix_val_str:{prefix_val_str}, suffix_val_str:{suffix_val_str}, "
        #              f"prefix_cand_str:{prefix_cand_str}, suffix_cand_str:{suffix_cand_str}")
        if suffix_val_str == prefix_cand_str:
            vals = filter_vals(cand_list, vals, "suffix_prefix")
        if prefix_val_str == suffix_cand_str:
            vals = filter_vals(cand_list, vals, "prefix_suffix")
        #logging.debug(f"final vals:{vals}")
    #logging.debug(f"len vals in:{len(vals_in)}, len vals:{len(vals)}")

    # 遍历vals, 裁剪cand中的元素
    for cand_type, cand_list in cand_type2list.items():
        first_index = cand_type.find('_')
        last_index = cand_type.rfind('_')

        prefix_cand_str = cand_type[:first_index]
        suffix_cand_str = cand_type[last_index+1:]
    
        #logging.debug(f"filter cand, prefix_val_str:{prefix_val_str}, suffix_val_str:{suffix_val_str}, "
        #              f"prefix_cand_str:{prefix_cand_str}, suffix_cand_str:{suffix_cand_str}")
        len_before = len(cand_list)
        if suffix_val_str == prefix_cand_str:
            filter_cands(cand_type2list, cand_type, vals, "suffix_prefix")
        if prefix_val_str == suffix_cand_str:
            filter_cands(cand_type2list, cand_type, vals, "prefix_suffix")
        len_after = len(cand_type2list[cand_type])
        #logging.debug(f"len before:{len_before}, len after:{len_after}")
    for k, v in cand_type2list.items():
        if isinstance(v, set) and len(v) == 0:
            return False

    cand_type2list[val_type_str] = vals
    return True

def filter_cands(cand_type2list, cand_type, vals, flag):
    new_cand = set()
    if flag == "suffix_prefix":
        val_dict = set()
        for val_elem in vals:
            val_dict.add(list(val_elem)[-1])
        for elem in cand_type2list[cand_type]:
            if list(elem)[0] in val_dict:
                new_cand.add(elem)
    elif flag == "prefix_suffix":
         val_dict = set()
         for val_elem in vals:
             val_dict.add(list(val_elem)[0])
         for elem in cand_type2list[cand_type]:
             if list(elem)[-1] in val_dict:
                 new_cand.add(elem)
    cand_type2list[cand_type] = new_cand

def filter_vals(cand_list, vals, flag):
    new_vals = set()
    if flag == "suffix_prefix":
        cand_dict = set()
        for cand_elem in cand_list:
            cand_dict.add(list(cand_elem)[0])
        for elem in vals:
            if list(elem)[-1] in cand_dict:
                new_vals.add(elem)
    elif flag == "prefix_suffix":
        cand_dict = set()
        for cand_elem in cand_list:
            cand_dict.add(list(cand_elem)[-1])
        for elem in vals:
            if list(elem)[0] in cand_dict:
                new_vals.add(elem)
    return new_vals

def node_match(n1, n2):
    return n1['node_type'] == n2['node_type']


def merge_FSI(start_time, max_query_time, FSI_linear_cand, FSI_cycle_cand, subgraphs, query, G1_node_index):
    FSI_pos = {}
    graph_ids = FSI_linear_cand & FSI_cycle_cand
    for graph_id in graph_ids:
        graph = subgraphs[graph_id]
        subgraph_pos = FSI_hip_match(start_time, max_query_time, graph_id, graph, query, G1_node_index)
        FSI_pos = cu.dict_union(FSI_pos, subgraph_pos)
        logging.debug(f"subgraph_pos:{subgraph_pos}")
        if (time_exceed(start_time, max_query_time)):
            break

    return FSI_pos

def FSI_hip_match(start_time, max_query_time, graph_id, graph, query, G1_node_index):
    matches = set()  # Store the final matching results
    matched_nodes = set()  # Record the matched nodes

    # Create a function for finding paths
    def find_path(start_node, end_node, path=[]):
        path = path + [start_node]  # Add the starting node to the path
        if start_node == end_node:
            return path
        for neighbor in graph.neighbors(start_node):
            if neighbor not in path and neighbor not in matched_nodes:  # Check if the path intersects
                new_path = find_path(neighbor, end_node, path)
                if new_path:  # If a path is found, return it
                    return new_path
        return []

    # Create a recursive function to find complete query matches
    def find_matching(start_time, max_query_time, query_index, current_path, matched_nodes):
        logging.debug(f"query:{query.nodes}, query_index:{query_index}, current_path:{current_path}")
        if (time_exceed(start_time, max_query_time)):
            return
        if query_index == len(query.nodes()):  # All query nodes have been matched
            matches.add(tuple(current_path))
            logging.debug(f"find one embedding:{current_path}")
            return

        query_node = list(query.nodes())[query_index]
        query_node_type = query.nodes[query_node]['node_type']

        # Find the matched nodes in the graph
        for graph_node in graph.nodes():
            if graph.nodes[graph_node]['node_type'] == query_node_type and graph_node not in matched_nodes:
                # 添加当前匹配的节点到已匹配集合
                matched_nodes.add(graph_node)
                current_path.append(graph_node)
                logging.debug(f"query:{query.nodes}, current_path:{current_path}")
                logging.debug(f"query_type:{[query.nodes[x]['node_type'] for x in query.nodes()]}, path_type:{[graph.nodes[x]['node_type'] for x in current_path]}")

                # If it is not the last node, find the path to the next query node
                if query_index < len(query.nodes()) - 1:
                    next_query_node = list(query.nodes())[query_index + 1]
                    next_query_node_type = query.nodes[next_query_node]['node_type']

                    for next_graph_node in graph.nodes():
                        if graph.nodes[next_graph_node]['node_type'] == next_query_node_type:
                            path = find_path(graph_node, next_graph_node)
                            if len(path) > 0:  # Find the path and continue searching for the next match
                                find_matching(start_time, max_query_time, query_index + 1, current_path, matched_nodes)
                else:
                    # Handle the case of the last node
                    find_matching(start_time, max_query_time, query_index + 1, current_path, matched_nodes)

                # Backtrack
                matched_nodes.remove(graph_node)
                current_path.pop()

    # Start from the first node to find the complete match
    matched_nodes = set()
    find_matching(start_time, max_query_time, 0, [], matched_nodes)

    embeddings = G1_node_index[-1 * (graph_id+1)]
    logging.debug(f"embeddings:{embeddings}, matches is:{matches}")
    res = {}
    if len(matches) > 0:
        res[graph_id] = set()
        for embedding_id in embeddings:
            for match in matches:
                res[graph_id].add((embedding_id, tuple(match)))
    return res

def ORI_hip_match(start_time, max_query_time, graph_id, graph, query, G_linear_index, tag, cands, query_start_node,
                  max_search_len, max_matches):

    # Create a function to find paths, limiting the path length to max_search_len
    def find_path(start_node, end_node, path, length, matches, matched_nodes):
        if length > max_search_len:  # If it exceeds the maximum length limit, return empty list
            return []
        path = path + [start_node]  # Add the starting node to the path
        if start_node == end_node:
            return path
        for neighbor in graph.neighbors(start_node):
            if neighbor not in path and neighbor not in matched_nodes:  # Check if the path intersects
                new_path = find_path(neighbor, end_node, path, length + 1, matches, matched_nodes)
                if len(new_path) > 0:  # If a path is found, return it
                    return new_path
        return []

    # Create a recursive function to find complete query matches, limiting the distance to max_search_len
    def find_matching(start_time, max_query_time, query_node, G_linear_index, current_path, current_len, matches, matched_nodes, matched_query_nodes):
        #logging.debug(f"query:{query.nodes}, query_node:{query_node}, current_path:{current_path}")
        #logging.debug(f"matched_nodes:{matched_nodes}, matched_query:{matched_query_nodes}, current_path:{current_path}")
        if (time_exceed(start_time, max_query_time)):
            return
        if (len(matches) > max_matches):
            return
        if len(current_path) == len(query.nodes()):  # All query nodes have been matched
            logging.debug(f"find one embedding: {current_path}")
            matches.add(tuple(current_path))
            return
        #logging.info(f"query_node:{query_node}")
        #logging.info(f"query nodes:{query.nodes()}")
        query_node_type = query.nodes[query_node]['node_type']
        #logging.debug(f"G_node_index:{G_node_index.keys()}")
        #logging.debug(f"query_node_type:{query_node_type}")

        # If it is not the last node, find the path to the next query node
        for next_query_node in query.neighbors(query_node):
            if next_query_node in matched_query_nodes:
                continue
            next_query_node_type = query.nodes[next_query_node]['node_type']
            graph_node = current_path[-1]

            linear_str = str(query_node_type) + "_" + str(next_query_node_type)
            if linear_str not in G_linear_index[2]:
                logging.debug(f"linear_str:{linear_str} not in G_linear_index keys:{G_linear_index.keys()}")
                continue
            for linears in G_linear_index[2][linear_str][graph_id]:
                if linears[0] == graph_node and linears[1] not in matched_nodes:
                    next_graph_node = linears[1]
                    current_path.append(next_graph_node)
                    matched_nodes.add(next_graph_node)
                    matched_query_nodes.add(next_query_node)
                    find_matching(start_time, max_query_time, next_query_node, G_linear_index, current_path, current_len,
                                  matches, matched_nodes, matched_query_nodes)
                    matched_nodes.remove(next_graph_node)
                    matched_query_nodes.remove(next_query_node)
                    current_path.pop()

                if (len(matches) > max_matches):
                    return

    ORI_pos = set()
    for val in tqdm(cands, desc=f"Processing {tag}"):
        matches = set()  # Store the final matching results
        matched_query_nodes = set() # Record the matched query nodes
        matched_nodes = set()  # Record the matched graph nodes
        graph_start_node = list(val)[0]
        if graph_start_node == "-":
            logging.debug(f"cands:{cands}")
            logging.debug(f"val:{val}")
            exit()
        # Start finding matches from the specified starting point
        matched_nodes.add(graph_start_node)
        matched_query_nodes.add(query_start_node)
        find_matching(start_time, max_query_time, query_start_node, G_linear_index, [graph_start_node],
                     1, matches, matched_nodes, matched_query_nodes)
        ORI_pos.update(matches)
        if len(ORI_pos) > max_matches:
            break
        if time_exceed(start_time, max_query_time):
            break
    return ORI_pos

def match_by_FSI(start_time, max_query_time, query, subgraphs, M, N, FSI_linear_index, FSI_cycle_index, G1,
                 G1_node_index, G1_linear_index, max_matches):

    def convert_querys(query, N):
        all_possible_graphs = {}
        for i in range(N):
            removed_num = i+1
            # List all possible graphs after removing 3 nodes

            # Use itertools.combinations to generate all combinations of 3 nodes
            for nodes_to_remove in itertools.combinations(query.nodes, removed_num):
                # Copy the original graph
                new_query = query.copy()

                # Remove each node in the combination
                new_query.remove_nodes_from(nodes_to_remove)

                # Save the new graph
                if removed_num not in all_possible_graphs:
                    all_possible_graphs[removed_num] = set()
                all_possible_graphs[removed_num].add((nodes_to_remove, new_query))
        return all_possible_graphs

    def get_FSI_cand(start_time, max_query_time, query, linears, cycles):
        FSI_linear_cand = get_linear_cand(linears, FSI_linear_index)
        logging.debug(f"FSI_linear_cand:{FSI_linear_cand}")
        FSI_cycle_cand = get_cycle_cand(cycles, FSI_cycle_index)
        logging.debug(f"FSI_cycle_cand:{FSI_cycle_cand}")
        FSI_cand = merge_FSI(start_time, max_query_time, FSI_linear_cand, FSI_cycle_cand, subgraphs, query, G1_node_index)
        logging.debug(f"FSI_cand:{FSI_cand}")
        return FSI_cand
    def do_FSI(start_time, max_query_time, ori_query, query, linears, cycles, removed_nodes):

        FSI_pos = set()
        FSI_cand = get_FSI_cand(start_time, max_query_time, query, linears, cycles)
        logging.debug(f"FSI_cand:{FSI_cand}")

        for graph_id, FSI_elem in FSI_cand.items():
            FSI_type = -1*(graph_id+1)
            SGI_query = nx.Graph()
            FSI_node = -1
            logging.debug(f"FSI_elem:{FSI_elem}")
            SGI_query.add_node(FSI_node, node_type=FSI_type)
            logging.debug(f"removed_nodes:{removed_nodes}")
            for removed_node in removed_nodes:
                SGI_query.add_node(removed_node, node_type=ori_query.nodes[removed_node]['node_type'])
                SGI_query.add_edge(FSI_node, removed_node)  # 添加边
            for combination in itertools.combinations(removed_nodes, 2):
                if ori_query.has_edge(combination[0], combination[1]):
                    SGI_query.add_edge(combination[0], combination[1])

            for eid, path in FSI_elem:
                logging.debug(f"eid:{eid}, path:{path}")
            embedding_ids = {(elem[0],) for elem in FSI_elem}
            logging.debug(f"before ORI_hip_match:{embedding_ids}")
            SGI_pos = ORI_hip_match(start_time, max_query_time, -1, G1, SGI_query, G1_linear_index,
                                    "_".join(map(str, SGI_query.nodes())), embedding_ids, FSI_node, 10,
                                    max_matches)
            logging.debug(f"SGI_pos:{SGI_pos}")
            FSI_pos.update(SGI_pos)
        return FSI_pos

    FSI_total = set()
    linears, cycles, linears_to_nodes, cycles_to_nodes = parse_query(query, M, N)
    FSI_pos = get_FSI_cand(start_time, max_query_time, query, linears, cycles)
    FSI_total.update(set().union(*FSI_pos.values()))
    logging.debug(f"FSI_total:{FSI_total}")
    if time_exceed(start_time, max_query_time):
        return FSI_total

    querys_tuple = convert_querys(query, 2)
    for removed_num, query_tuple in querys_tuple.items():
        for tuples in query_tuple:
            removed_nodes = tuples[0]
            new_query = tuples[1]
            logging.debug(f"remove_nodes:{removed_nodes}, new_querys:{new_query.nodes}")
            new_linears, new_cycles, new_linears_to_nodes, new_cycles_to_nodes = parse_query(new_query, M, N)
            FSI_pos = do_FSI(start_time, max_query_time, query, new_query, new_linears, new_cycles, removed_nodes)
            logging.debug(f"FSI_pos:{FSI_pos}")
            FSI_total.update(FSI_pos)
            if len(FSI_total) > max_matches:
                return FSI_total
            if time_exceed(start_time, max_query_time):
                return FSI_total

    return FSI_total


def parse_query(query, M, N):
    #cu.draw_graph(query)
    linears = {}
    cycles = {}
    linears_to_nodes = {}
    cycles_to_nodes = {}
    for v in tqdm(query.nodes(), desc="Parsing query"):
        parse_by_dfs(query, v, [], M, N, linears, cycles, linears_to_nodes, cycles_to_nodes)  # The initial path is empty
    return linears, cycles, linears_to_nodes, cycles_to_nodes

# Perform DFS traversal and build the index
def parse_by_dfs(query, v, Path, M, N, linears, cycles, linears_to_nodes, cycles_to_nodes):
    if v not in Path:  # Indicate that it is not a cycle, but a linear path
        if len(Path) >= M:
            return
        Path.append(v)  # Add the current node to the path
        type_linear_path = [query.nodes[x]['node_type'] for x in Path]
        if len(Path) > 1:
            if len(Path) not in linears:
                linears[len(Path)] = set()
            linears[len(Path)].add(tuple(type_linear_path))
            linears_to_nodes["_".join(map(str, type_linear_path))] = tuple(Path)
        for neighbor in query.neighbors(v):  # Traverse all neighboring nodes
            parse_by_dfs(query, neighbor, Path, M, N, linears, cycles, linears_to_nodes, cycles_to_nodes)  # Recursively perform DFS
        Path.pop()  # Backtrack, restoring the path to its state before the call
    elif v == Path[0]:  # Indicate that a cycle has been formed
        if len(Path) >= N:
            return
        type_cycle_path = [query.nodes[x]['node_type'] for x in Path]
        type_str = "_".join(map(str, type_cycle_path))
        if len(Path) > 1:
            if len(Path)+1 not in cycles:
                cycles[len(Path)+1] = set()
            cycles[len(Path)+1].add(tuple(type_cycle_path))
            cycles_to_nodes["_".join(map(str, type_cycle_path))] = tuple(Path)


def main():
    # Corresponding to Fig 10.(c).(d). Analyse query performance for CSM-Index and JSM-FSI
    max_query_times = [600]
    for max_query_time in max_query_times:
        logging.info(f"begin max_query_time:{max_query_time}")
        run_by_embeeding(max_query_time)

    # Corresponding to Fig 9
    query_sizes = range(3, 8)
    for query_size in query_sizes:
        logging.info(f"begin query_size:{query_size}")
        run_by_query_size(query_size)

    # Corresponding to Fig 11.(c).(d). Analyse query performance for CSM-FSI
    max_query_times = [600]
    for max_query_time in max_query_times:
        logging.info(f"begin mix max_query_time:{max_query_time}")
        run_by_mix(max_query_time)

def run_querys(querys, subgraphs, M, N, G, G_node_index, G_linear_index, G1, G1_node_index, G1_linear_index,
               FSI_linear_index, FSI_cycle_index, max_query_time, max_matches):
    ORI_elapse = []
    FSI_elapse = []
    ORI_size = []
    FSI_size = []
    for query in tqdm(querys, desc=f"Processing querys"):
        # ORI query
        t1 = time.time()
        ORI_pos = match_by_ORI(t1, max_query_time, query, M, N, G, G_node_index, G_linear_index, max_matches)
        t2 = time.time()
        logging.info(f"ORI_pos size:{len(ORI_pos)}")
        ORI_size.append(len(ORI_pos))
        logging.info(f"ORI_elapse:{t2-t1}")
        ORI_elapse.append(t2 - t1)
        # FSI query
        FSI_pos = match_by_FSI(t2, max_query_time, query, subgraphs, M, N, FSI_linear_index, FSI_cycle_index, G1,
                               G1_node_index, G1_linear_index, max_matches)
        logging.info(f"FSI_pos size:{len(FSI_pos)}")
        t3 = time.time()
        FSI_size.append(len(FSI_pos))
        logging.info(f"FSI_elapse:{t3-t2}")
        FSI_elapse.append(t3 - t2)

    return ORI_size, FSI_size, ORI_elapse, FSI_elapse

def run_by_query_size(query_size):
    max_query_time = 600
    query_num = 100
    degree_mean, degree_std = 2, 4  # Degree distribution of regular nodes
    M = 3
    N = 3
    embedding_count = 9
    max_matches = 100
    max_path = 5

    dir_in = str(sys.argv[1]) + "/"
    (G, G1, subgraphs, FSI_node_index, FSI_linear_index, FSI_cycle_index, FSI_build_time, G_node_index,
     G_linear_index, G_build_times, G1_node_index, G1_linear_index, G1_build_times) = \
        cu.read_info_by_maxlen(dir_in, str(M) + "_" + str(N) + "_" + str(embedding_count), max_path, False)

    total_nodes = sum(len(subgraph.nodes()) for subgraph in subgraphs)
    embedding_rate = 100 * total_nodes * embedding_count/10000

    querys = gen_querys(G, query_num, query_size, 0, degree_mean, degree_std)
    ORI_sizes, FSI_sizes, ORI_elapses, FSI_elapses =(
        run_querys(querys, subgraphs, M, N, G, G_node_index, G_linear_index, G1, G1_node_index, G1_linear_index,
                   FSI_linear_index, FSI_cycle_index, max_query_time, max_matches))

    G.clear()
    G1.clear()
    subgraphs.clear()
    FSI_node_index.clear()
    FSI_linear_index.clear()
    FSI_cycle_index.clear()
    G_node_index.clear()
    G_linear_index.clear()
    G1_node_index.clear()
    G1_linear_index.clear()
    logging.info(f"embedding_rate:{embedding_rate}")
    logging.info(f"ORI_elapses:{ORI_elapses}")
    logging.info(f"FSI_elapses:{FSI_elapses}")
    logging.info(f"embedding_count:{embedding_count}")
    logging.info(f"ORI_sizes:{ORI_sizes}")
    logging.info(f"FSI_sizes:{FSI_sizes}")

    cu.save_online(dir_in, embedding_count, embedding_rate, FSI_elapses, ORI_elapses, FSI_sizes, ORI_sizes, f"query_size_{query_size}_online")

def run_by_embeeding(max_query_time):
    query_num = 100
    query_mean, query_std = 5, 2
    degree_mean, degree_std = 2, 4  # Degree distribution of regular nodes
    y = np.array(range(1, 21, 2))
    M = 3
    N = 3
    max_matches = 100
    ORI_elapses = []
    FSI_elapses = []
    ORI_sizes = []
    FSI_sizes = []
    max_path = 5

    dir_in = str(sys.argv[1]) + "/"
    embedding_rate = []
    for embedding_count in tqdm(y, desc="Processing each embedding_count"):
        (G, G1, subgraphs, FSI_node_index, FSI_linear_index, FSI_cycle_index, FSI_build_time, G_node_index,
         G_linear_index, G_build_times, G1_node_index, G1_linear_index, G1_build_times) =\
            cu.read_info_by_maxlen(dir_in, str(M) + "_" + str(N) + "_" + str(embedding_count), max_path, False)

        total_nodes = sum(len(subgraph.nodes()) for subgraph in subgraphs)
        embedding_rate.append(100 * total_nodes * embedding_count/10000)

        querys = gen_querys(G, query_num, query_mean, query_std, degree_mean, degree_std)
        ORI_size, FSI_size, ORI_elapse, FSI_elapse =(
            run_querys(querys, subgraphs, M, N, G, G_node_index, G_linear_index, G1, G1_node_index, G1_linear_index,
                       FSI_linear_index, FSI_cycle_index, max_query_time, max_matches))

        ORI_sizes.append(tuple(ORI_size))
        FSI_sizes.append(tuple(FSI_size))
        ORI_elapses.append(tuple(ORI_elapse))
        FSI_elapses.append(tuple(FSI_elapse))
        logging.info(f"ORI_sizes:{ORI_sizes}")
        logging.info(f"FSI_sizes:{FSI_sizes}")
        logging.info(f"ORI_elapses:{ORI_elapses}")
        logging.info(f"FSI_elapses:{FSI_elapses}")
        G.clear()
        G1.clear()
        subgraphs.clear()
        FSI_node_index.clear()
        FSI_linear_index.clear()
        FSI_cycle_index.clear()
        G_node_index.clear()
        G_linear_index.clear()
        G1_node_index.clear()
        G1_linear_index.clear()
    logging.info(f"embedding_rate:{embedding_rate}")
    logging.info(f"ORI_elapses:{ORI_elapses}")
    logging.info(f"FSI_elapses:{FSI_elapses}")
    logging.info(f"y:{y}")
    logging.info(f"ORI_sizes:{ORI_sizes}")
    logging.info(f"FSI_sizes:{FSI_sizes}")

    cu.save_online(dir_in, y, embedding_rate, FSI_elapses, ORI_elapses, FSI_sizes, ORI_sizes, f"max_query_time_{max_query_time}_online")
def run_by_mix(max_query_time):
    query_num = 100
    query_mean, query_std = 5, 2
    degree_mean, degree_std = 2, 4  # Degree distribution of regular nodes
    y = np.array(range(1, 21, 4))
    M = 3
    N = 3
    max_matches = 100
    ORI_elapses = []
    FSI_elapses = []
    ORI_sizes = []
    FSI_sizes = []
    max_path = 5

    dir_in = str(sys.argv[1]) + "/"
    embedding_rate = []
    for embedding_count in tqdm(y, desc="Processing each embedding_count"):
        (G, G1, subgraphs, FSI_node_index, FSI_linear_index, FSI_cycle_index, FSI_build_time, G_node_index,
         G_linear_index, G_build_times, G1_node_index, G1_linear_index, G1_build_times) = \
            cu.read_info_by_maxlen(dir_in, str(M) + "_" + str(N) + "_" + str(embedding_count), max_path, True)

        total_nodes = sum(len(subgraph.nodes()) for subgraph in subgraphs)
        embedding_rate.append(100 * total_nodes * embedding_count/10000)

        querys = gen_querys(G, query_num, query_mean, query_std, degree_mean, degree_std)
        ORI_size, FSI_size, ORI_elapse, FSI_elapse =(
            run_querys(querys, subgraphs, M, N, G, G_node_index, G_linear_index, G1, G1_node_index, G1_linear_index,
                       FSI_linear_index, FSI_cycle_index, max_query_time, max_matches))

        ORI_sizes.append(tuple(ORI_size))
        FSI_sizes.append(tuple(FSI_size))
        ORI_elapses.append(tuple(ORI_elapse))
        FSI_elapses.append(tuple(FSI_elapse))
        logging.info(f"ORI_sizes:{ORI_sizes}")
        logging.info(f"FSI_sizes:{FSI_sizes}")
        logging.info(f"ORI_elapses:{ORI_elapses}")
        logging.info(f"FSI_elapses:{FSI_elapses}")
        G.clear()
        G1.clear()
        subgraphs.clear()
        FSI_node_index.clear()
        FSI_linear_index.clear()
        FSI_cycle_index.clear()
        G_node_index.clear()
        G_linear_index.clear()
        G1_node_index.clear()
        G1_linear_index.clear()
    logging.info(f"embedding_rate:{embedding_rate}")
    logging.info(f"ORI_elapses:{ORI_elapses}")
    logging.info(f"FSI_elapses:{FSI_elapses}")
    logging.info(f"y:{y}")
    logging.info(f"ORI_sizes:{ORI_sizes}")
    logging.info(f"FSI_sizes:{FSI_sizes}")

    cu.save_online(dir_in, y, embedding_rate, FSI_elapses, ORI_elapses, FSI_sizes, ORI_sizes, f"mix_{max_query_time}_online")

if __name__ == '__main__':
    random.seed(time.time())
    cu.custom_logging()
    t1 = time.time()
    main()
    t2 = time.time()
    logging.info(f"total time:{t2-t1}")
