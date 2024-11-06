import logging
import time

import numpy as np
import networkx as nx
import random
import sys
from itertools import combinations, permutations
from tqdm import tqdm
import common_util as cu

#  生成 100 种类型的分布，确保总节点数为 10000
def gen_type_distribution(type_mean, type_std, total_types, total_nodes):
    type_distribution = np.sort(np.random.normal(type_mean, type_std, total_types).astype(int))
    type_distribution = np.clip(type_distribution, 1, None)  # 确保没有负数
    type_distribution = type_distribution / type_distribution.sum() * total_nodes  # 归一化为10000个节点
    type_distribution = type_distribution.astype(int)
    
    # 确保分布精确为 10000
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


    #logging.debug(f"生成了 {frequent_count} 个频繁子图，节点数符合 (10, 25) 的正态分布")
    
    # 为频繁子图的节点分配类型，基于类型分布的概率进行选择
    for i in range(frequent_count):
        subgraph_node_types = np.random.choice(np.arange(total_types), subgraph_node_counts[i], p=type_rates)
        subgraph_type_distribution.append(subgraph_node_types)
        # 给每个节点添加 'node_type' 属性
        for j, node in enumerate(frequent_subgraphs[i].nodes()):
            frequent_subgraphs[i].nodes[node]['node_type'] = int(subgraph_node_types[j])



    return frequent_subgraphs, subgraph_node_counts, subgraph_type_distribution


def build_FSI_index(frequent_subgraphs, M, N, max_path):
    # 初始化节点类型到频繁子图的索引
    FSI_node_index = build_FSI_node_index(frequent_subgraphs)

    FSI_linear_index = {}
    FSI_cycle_index = {}
    for subgraph_index, subgraph in enumerate(frequent_subgraphs):  # 遍历每个频繁子图
        #logging.debug(f"subgraph_index {subgraph_index}: {subgraph.nodes()}")
        build_single_FSI_index("FSI", subgraph_index, subgraph, M, N, max_path, FSI_linear_index, FSI_cycle_index, True)

    return FSI_node_index, FSI_linear_index, FSI_cycle_index

def build_single_FSI_index(tag, subgraph_index, subgraph, M, N, max_path, FSI_linear_index, FSI_cycle_index, is_build_cycle):
    for v in tqdm(subgraph.nodes(), desc=f"build {tag} index, subgraph_index:{subgraph_index}, max_path:{max_path}"):
        build_by_dfs(subgraph_index, subgraph, v, [], M, N, max_path, FSI_linear_index, FSI_cycle_index,
                     is_build_cycle, 0)  # 初始 Path 为空

def build_single_SGI_index(subgraph_index, subgraph, M, N, max_path, FSI_linear_index, FSI_cycle_index, is_build_cycle):
    call_times = 0
    nodes_with_dash = {node for node in subgraph.nodes() if str(node).startswith("-")}
    for v in tqdm(nodes_with_dash, desc=f"build SGI index, subgraph_index:{subgraph_index}, max_path:{max_path}"):
        #logging.info(f"v:{v}")
        call_t = build_by_dfs(subgraph_index, subgraph, v, [], M, N, max_path, FSI_linear_index, FSI_cycle_index,
                              is_build_cycle, 0)  # 初始 Path 为空
        call_times += call_t
    return call_times

# DFS 遍历并构建索引
def build_by_dfs(subgraph_id, subgraph, v, Path, M, N, max_path, FSI_linear_index, FSI_cycle_index, is_build_cycle,
                 call_times):
    if len(Path) > max_path:
        return call_times
    if v not in Path:  # 说明不是环，是直线通路
        if M >=2 and len(Path) >= 2:
            for m in range(2, min(M + 1, len(Path) + 1)):
                build_linear_index(subgraph_id, subgraph, v, Path, m, FSI_linear_index)  # 构建线性索引
        Path.append(v)  # 把当前节点加入到 Path
        for neighbor in subgraph.neighbors(v):  # 遍历所有邻居节点
            call_times = build_by_dfs(subgraph_id, subgraph, neighbor, Path, M, N, max_path, FSI_linear_index,
                                  FSI_cycle_index, is_build_cycle, call_times)  # 递归执行 DFS
        Path.pop()  # 回溯，恢复 Path 到调用前的状态
        return call_times + 1
    elif is_build_cycle and v == Path[0]:  # 说明形成了环
        if N >=3 and len(Path) >= 3:
            for n in range(3, min(N + 1, len(Path) + 1)):
                build_cycle_index(subgraph_id, subgraph, Path, n, FSI_cycle_index)  # 构建环性索引
        return call_times + 1
    return call_times


# 构建线性索引
def build_linear_index(subgraph_id, subgraph, v, Path, m, FSI_linear_index):
    # 获取 Path 的子路径（除第一个节点外的部分）
    subPath = Path[1:]
    # 生成 m-2 长度的组合，组合中的节点需要按照 Path 顺序排列
    for group in combinations(subPath, m - 2):
        group = list(group)
        linear_path = [Path[0]] + group + [v]  # 拼接形成线性路径
        #logging.debug(f"linear_path: {len(linear_path)}, {linear_path}")
        type_linear_path = [subgraph.nodes[x]['node_type'] for x in linear_path]
        type_str = "_".join(map(str, type_linear_path))
        # 记录该链路到 subgraph 中的 m 线性索引
        if m not in FSI_linear_index:
            FSI_linear_index[m] = {}
        if type_str not in FSI_linear_index[m]:
            FSI_linear_index[m][type_str] = {}
        if subgraph_id not in FSI_linear_index[m][type_str]:
            FSI_linear_index[m][type_str][subgraph_id] = set()
        FSI_linear_index[m][type_str][subgraph_id].add(tuple(linear_path))
        #logging.debug(f"线性索引: {type_linear_path} 到频繁子图 {subgraph_id} 的索引位置")

# 构建环性索引
def build_cycle_index(subgraph_id, subgraph, Path, n, FSI_cycle_index):
    # 获取 Path 的子路径（除第一个节点外的部分）
    subPath = Path[1:]
    # 生成 n-2 长度的组合，组合中的节点需要按照 Path 顺序排列
    for group in combinations(subPath, n - 2):
        group = list(group)
        cycle_path = [Path[0]] + group  # 拼接形成环路径
        type_cycle_path = [subgraph.nodes[x]['node_type'] for x in cycle_path]
        type_str = "_".join(map(str, type_cycle_path))
        # 记录该链路到 subgraph 中的 n 环性索引
        if n not in FSI_cycle_index:
            FSI_cycle_index[n] = {}
        if type_str not in FSI_cycle_index[n]:
            FSI_cycle_index[n][type_str] = {}
        if subgraph_id not in FSI_cycle_index[n][type_str]:
            FSI_cycle_index[n][type_str][subgraph_id] = set()
        FSI_cycle_index[n][type_str][subgraph_id].add(tuple(cycle_path))
        #logging.debug(f"环性索引: {type_cycle_path} 到频繁子图 {subgraph_id} 的索引位置")


def build_FSI_node_index(frequent_subgraphs):
    # 初始化节点类型到频繁子图的索引
    
    FSI_node_index = {}
    for subgraph_index, subgraph in enumerate(frequent_subgraphs):  # 遍历每个频繁子图
        # 遍历当前频繁子图中的节点类型
        for node_id in subgraph.nodes():
            node_type = subgraph.nodes[node_id]['node_type']
            if node_type not in FSI_node_index:
                FSI_node_index[node_type] = []
    
            # 记录该节点类型在哪个子图(subgraph_index)的哪个相对位置(node_id)
            FSI_node_index[node_type].append((subgraph_index, node_id))
    
    # 输出节点类型到频繁子图的索引
    #for node_type, locations in FSI_node_index.items():
        #logging.debug(f"节点类型 {node_type} 出现在以下位置：{locations}")
    return FSI_node_index
def build_FSO_linear_index(frequent_subgraphs, L):
    index = {}  # 初始化索引
    all_nodes = []  # 存储所有频繁子图中的节点
    
    # 遍历所有频繁子图
    for subgraph_index, subgraph in enumerate(frequent_subgraphs):
        # 遍历子图中的每个节点
        for node_id in subgraph.nodes():
            node_type = subgraph.nodes[node_id]['node_type']
            # 将节点及其类型与频繁子图的id打包为tuple并加入数组
            all_nodes.append((node_id, node_type, subgraph_index))
    
    # 生成任意m个节点组成的组合链路（m从2到L）
    for m in range(2, L + 1):
        if m not in index:
            index[m] = {}
        
        # 从 all_nodes 中生成 m 个元素的组合
        for node_combination in combinations(all_nodes, m):
            # 对组合生成所有可能的顺序排列
            for permuted_combination in permutations(node_combination):
                # 提取节点类型
                type_list = [str(node[1]) for node in permuted_combination]  # node[1]是节点类型
                type_str = "_".join(type_list)  # 使用下划线连接节点类型字符串

                # 初始化一个集合以存储唯一值
                if type_str not in index[m]:
                    index[m][type_str] = set()  # 使用集合来去重
                
                # 将组合的节点信息（包括频繁子图id）作为值
                val = tuple(node[2] for node in permuted_combination)  # node[0]是节点，node[2]是子图ID
                
                # 将这个链路存入索引
                if type_str not in index[m]:
                    index[m][type_str] = []
                index[m][type_str].add(val)
    
    return index
def gen_simplified_graph(G, embeddings, embedding_nodes, embedding_subgraph_map):

    G1 = nx.Graph()  # 新的简化图 G1
    embedding_to_aggregate = {}  # 保存每个 embedding 节点对应的聚合节点
    
    # 1. 将所有非 embedding 的节点及它们的边加入 G1
    for node in G.nodes():
        if node not in embedding_nodes:
            # 将非 embedding 节点加入到 G1
            G1.add_node(node, **G.nodes[node])  # 复制节点属性
    
    #logging.debug(f"step0 简化图 G 节点数: {len(G.nodes())}, 边数: {len(G.edges())}")
    # 添加非 embedding 节点之间的边
    for u, v in G.edges():
        if u not in embedding_nodes and v not in embedding_nodes:
            G1.add_edge(u, v)  # 非 embedding 节点之间的边直接添加
    
    logging.debug(f"embedding_nodes: {len(embedding_nodes)}")
    #logging.debug(f"step1 简化图 G1 节点数: {len(G1.nodes())}, 边数: {len(G1.edges())}")

    # 2. 对每个 embedding 生成一个聚合节点并加入到 G1
    for embedding_id, embedding in enumerate(embeddings):
        # 生成一个新的聚合节点
        agg_node = str(-1 * (embedding_id+1)).zfill(5)
        #logging.debug(f"agg_node:{agg_node}")
        subgraph_type = int(embedding_subgraph_map[embedding_id])
        G1.add_node(agg_node, node_type=int(-1 * (subgraph_type+1)))
        embedding_to_aggregate[embedding_id] = agg_node  # 记录聚合节点

    # 3. 遍历所有 embedding 内部节点，替换边为聚合节点的边
    for embedding_id, embedding in enumerate(embeddings):
        agg_node = embedding_to_aggregate[embedding_id]
    
        # 遍历每个子图的节点
        for node in embedding.values():
            # 找到这个 embedding 节点的所有邻居
            for neighbor in G.neighbors(node):
                # 如果邻居属于 embedding，找到对应的聚合节点
                if neighbor in embedding_nodes:
                    # 找到邻居所在的子图
                    for other_embedding_id, other_embedding in enumerate(embeddings):
                        if neighbor in other_embedding.values():
                            agg_neighbor = embedding_to_aggregate[other_embedding_id]
                            # 添加两个聚合节点之间的边
                            if not G1.has_edge(agg_node, agg_neighbor):
                                G1.add_edge(agg_node, agg_neighbor)
                            break
                else:
                    # 如果邻居是非 embedding 节点，则将边改为聚合节点与外部节点的边
                    if not G1.has_edge(agg_node, neighbor):
                        G1.add_edge(agg_node, neighbor)
    
    logging.debug(f"简化图 G1 节点数: {len(G1.nodes())}, 边数: {len(G1.edges())}")

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
    # 初始化节点类型到 G 中位置的索引
    node_index = {}
    for node in G.nodes(data=True):
        node_id = node[0]  # 节点 ID
        node_type = node[1].get('node_type')  # 获取节点类型
        if node_type not in node_index:
            node_index[node_type] = []

        # 将该类型的节点位置加入到索引中
        node_index[node_type].append(node_id)
    #for node_type, locations in node_index.items():
        #logging.debug(f"节点类型 {node_type} 出现在以下位置：{locations}")
    return node_index
# 5. 初始化图 G，并将 embedding 节点和对应的边加入图中
def gen_original_graph(type_distribution, frequent_subgraphs, embedding_count, subgraph_type_distribution, degree_mean, degree_std, total_nodes):
    G = nx.Graph() # 原图
    current_node = 0
    embeddings = []
    embedding_nodes = []  # 存储 embedding 的节点
    embedding_subgraph_map = {} # embedding 和频繁子图的映射关系
    
    for subgraph_index, subgraph in enumerate(frequent_subgraphs):
        for _ in range(embedding_count):
            node_mapping = {}
            # 为每个 embedding 分配节点并连接边
            for i, node_type in enumerate(subgraph_type_distribution[subgraph_index]):
                node_id = str(current_node).zfill(5)  # 将节点编号格式化为四位数字
                G.add_node(node_id, node_type=int(node_type))
                node_mapping[i] = node_id 
                embedding_nodes.append(node_id)
                current_node += 1
            # 添加子图中的边
            for u, v in subgraph.edges():
                G.add_edge(node_mapping[u], node_mapping[v])
            embeddings.append(node_mapping)
            embedding_subgraph_map[len(embeddings)-1] = subgraph_index
    
    #logging.debug(f"已将 embedding 节点及其边加入图中")
    if len(embedding_nodes) >= total_nodes:
        logging.debug(f"embedding_nodes {len(embedding_nodes)} > total_nodes {total_nodes}")
        exit()

    # 4. 设置非 embedding 节点的类型，确保 10000 个节点的类型分布符合 (100, 100)
    remaining_type_distribution = type_distribution.copy()
    
    # 减去已分配的 embedding 节点类型
    for i in range(frequent_count):
        for t in subgraph_type_distribution[i]:
            remaining_type_distribution[t] -= embedding_count
    
    # 确保剩余的节点类型也遵循总体分布
    remaining_node_types = []
    for t in range(total_types):
        remaining_node_types.extend([t] * remaining_type_distribution[t])
    
    random.shuffle(remaining_node_types)

    # 6. 将非 embedding 节点加入图中
    for remaining_node_type in remaining_node_types:
        node_id = str(current_node).zfill(5)  # 将节点编号格式化为四位数字
        G.add_node(node_id, node_type=int(remaining_node_type))
        current_node += 1
    
    # 7. 遍历所有节点，添加边，确保度的分布符合 (4, 4) 的正态分布
    for node in G.nodes():
        # 计算目标度数
        degree = int(np.clip(np.random.normal(degree_mean, degree_std), 1, 10))
        
        # 获取该节点当前已经建立的边数
        current_degree = len(list(G.neighbors(node)))
        
        # 计算还需要建立的边数，确保总边数符合目标度数
        remaining_degree = degree - current_degree
        
        # 如果需要建立的边数大于0，继续添加边
        if remaining_degree > 0:
            # 获取所有潜在的邻居（去除自身和已连接的节点）
            potential_edges = [n for n in G.nodes() if n != node and not G.has_edge(node, n)]
            
            # 如果潜在邻居的数量少于remaining_degree，防止超出范围
            if remaining_degree > len(potential_edges):
                remaining_degree = len(potential_edges)
            
            # 随机打乱潜在邻居并选择 remaining_degree 个邻居
            random.shuffle(potential_edges)
            neighbors = random.sample(potential_edges, remaining_degree)
            
            # 为该节点与新邻居建立边
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

exit()
# 设置随机种子以确保可重复性
np.random.seed(42)
random.seed(42)
cu.custom_logging()

is_test = False

# 参数定义
total_nodes = 10000  # 总节点数
total_types = 100 # 总节点类型总数
#frequent_count = 3  # 生成的频繁子图数量
#embedding_count = 10  # 每个频繁子图的嵌入次数
degree_mean, degree_std = 3, 2  # 普通节点的度数分布
type_mean, type_std = 100, 2500  # 节点类型分布的正态分布 (100, 100)
subgraph_node_mean, subgraph_node_std = 10, 5  # 频繁子图的节点数分布
#max_path = 4 # 最长在多大的路径内建立多跳索引
FSI_max_path = 10

if is_test == True:
    #=================================
    # 参数定义
    total_nodes = 200  # 总节点数
    total_types = 10 # 总节点类型总数
    #frequent_count = 3  # 生成的频繁子图数量
    #embedding_count = 10  # 每个频繁子图的嵌入次数
    degree_mean, degree_std = 3, 2  # 普通节点的度数分布
    type_mean, type_std = 10, 5  # 节点类型分布的正态分布 (100, 100)
    subgraph_node_mean, subgraph_node_std = 5, 3  # 频繁子图的节点数分布
    #max_path = 4 # 最长在多大的路径内建立多跳索引
    FSI_max_path = 10
    #=================================


exit()

dir_in = str(sys.argv[1]) + "/"
max_path = int(sys.argv[2])

#logging.debug("类型分布已生成，符合 (100, 100) 的正态分布")
type_distribution = gen_type_distribution(type_mean, type_std, total_types, total_nodes)
logging.debug(f"len type:{len(type_distribution)}, sum type:{sum(type_distribution)}")
#cu.plt_his(type_distribution)

frequent_count = 20
if is_test == True:
    #=================================
    frequent_count = 3
    #=================================
# 生成 frequent_count 个频繁子图，每个子图的节点数符合 (10, 25) 的正态分布
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

