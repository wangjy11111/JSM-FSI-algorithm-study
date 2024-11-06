import pickle
import numpy as np
import networkx as nx
import inspect
import sys
import logging
import os
import re

def custom_logging():

    # 定义自定义日志格式
    log_format = '[%(asctime)s][%(levelname)s][%(filename)s/%(funcName)s():%(lineno)d] %(message)s'

    # 配置基本的日志设置
    logging.basicConfig(level=logging.INFO, format=log_format)


def custom_print(*args, **kwargs):
    # 获取当前调用的帧
    frame = inspect.currentframe().f_back

    # 获取文件名、方法名和行号
    filename = os.path.basename(frame.f_code.co_filename)  # 获取文件名，不包括路径
    function_name = frame.f_code.co_name  # 获取方法名
    lineno = frame.f_lineno  # 获取行号

    # 格式化输出
    print(f"{filename}/{function_name} {lineno} --", *args, **kwargs)

def dict_intersaction(dict_a, dict_b):
    #custom_print(f"dict_a:{dict_a}, dict_b:{dict_b}")
    # 计算交集
    intersection = {key: dict_a[key] & dict_b[key] for key in dict_a if key in dict_b}

    # 移除空集合
    intersection = {k: v for k, v in intersection.items() if v}
    # 计算交集
    return intersection
def dict_union(dict_a, dict_b):
    return {key: dict_a.get(key, set()) | dict_b.get(key, set()) for key in dict_a.keys() | dict_b.keys()}

def get_total_size(obj):
    """递归计算对象及其元素的总字节数"""
    total_size = sys.getsizeof(obj)
    if isinstance(obj, tuple) or isinstance(obj, list) or isinstance(obj, set):
        total_size += sum(get_total_size(item) for item in obj)
    elif isinstance(obj, dict):
        total_size += sum(get_total_size(key) + get_total_size(value) for key, value in obj.items())
    return total_size

def save_single_graph(directory, filename, graph):
    logging.info(f"save graph {directory + filename}")
    nx.write_graphml(graph, directory + filename)

def save_multi_graph(directory, name_prefix, graphs):
    for i, graph in enumerate(graphs):
        logging.info(f"save graphs {directory}{name_prefix}_{i}.graphml")
        nx.write_graphml(graph, f"{directory}{name_prefix}_{i}.graphml")

def save_single_obj(directory, filename, obj):
    logging.info(f"save obj {directory + filename}")
    with open(directory + filename, "wb") as file:
        pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)

def save_multi_obj(directory, name_prefix, name_suffix, objs):
    for i, elem in objs.items():
        logging.info(f"save objs {name_prefix + "_" + str(i) + "_" + name_suffix}")
        with open(directory + name_prefix + "_" + str(i) + "_" + name_suffix, "wb") as file:
            pickle.dump(elem, file, protocol=pickle.HIGHEST_PROTOCOL)


def save_info(directory, G, G1, subgraphs, FSI_node_index, FSI_linear_index, FSI_cycle_index, FSI_build_time,
              G_node_index, G_linear_index, G_build_time, G1_node_index, G1_linear_index, G1_build_time, name_prefix):
    save_single_graph(directory, name_prefix + "_ori.graphml", G)
    save_single_graph(directory, name_prefix + "_simp.graphml", G1)
    save_multi_graph(directory, name_prefix + "_freq", subgraphs)
    save_single_obj(directory, name_prefix + "_FSIn.pkl", FSI_node_index)
    save_single_obj(directory, name_prefix + "_FSIl.pkl", FSI_linear_index)
    save_single_obj(directory, name_prefix + "_FSIc.pkl", FSI_cycle_index)
    save_single_obj(directory, name_prefix + "_FSIt.pkl", FSI_build_time)
    save_single_obj(directory, name_prefix + "_orin.pkl", G_node_index)
    save_multi_obj(directory, name_prefix, "oril.pkl", G_linear_index)
    save_single_obj(directory, name_prefix + "_orit.pkl", G_build_time)
    save_single_obj(directory, name_prefix + "_simpn.pkl", G1_node_index)
    save_multi_obj(directory, name_prefix, "simpl.pkl", G1_linear_index)
    save_single_obj(directory, name_prefix + "_simpt.pkl", G1_build_time)

    #nx.write_graphml(G, directory + name_prefix + "_ori.graphml")
    #nx.write_graphml(G1, directory + name_prefix + "_simp.graphml")
    #for i, subgraph in enumerate(subgraphs):
    #    nx.write_graphml(subgraph, f"{directory}{name_prefix}_freq_{i}.graphml")
    #with open(directory + name_prefix + "_FSIn.pkl", "wb") as file:
    #    pickle.dump(FSI_node_index, file)
    #with open(directory + name_prefix + "_FSIl.pkl", "wb") as file:
    #    pickle.dump(FSI_linear_index, file)
    #with open(directory + name_prefix + "_FSIc.pkl", "wb") as file:
    #    pickle.dump(FSI_cycle_index, file)
    #with open(directory + name_prefix + "_FSIt.pkl", "wb") as file:
    #    pickle.dump(FSI_build_time, file)
    #with open(directory + name_prefix + "_orin.pkl", "wb") as file:
    #    pickle.dump(G_node_index, file)
    #for max_path, G_linear_elem in G_linear_index.items():
    #    with open(directory + name_prefix + "_" + str(max_path) + "_oril.pkl", "wb") as file:
    #        pickle.dump(G_linear_elem, file)
    #with open(directory + name_prefix + "_orit.pkl", "wb") as file:
    #    pickle.dump(G_build_time, file)
    #with open(directory + name_prefix + "_simpn.pkl", "wb") as file:
    #    pickle.dump(G1_node_index, file)
    #for max_path, G1_linear_elem in G1_linear_index.items():
    #    with open(directory + name_prefix + "_" + str(max_path) + "_simpl.pkl", "wb") as file:
    #        pickle.dump(G1_linear_elem, file)
    #with open(directory + name_prefix + "_simpt.pkl", "wb") as file:
    #    pickle.dump(G1_build_time, file)

def read_info(directory, name_prefix):

    max_len = 6

    logging.info(f"reading {directory}{name_prefix}")
    G = nx.read_graphml(directory + name_prefix + "_ori.graphml")
    G1 = nx.read_graphml(directory + name_prefix + "_simp.graphml")

    #graph_files = [f for f in os.listdir(directory) if f.startswith(name_prefix + "_freq_") and f.endswith('.graphml')]
    graph_files = [f for f in os.listdir(directory) if f.startswith(name_prefix + "_freq_") and f.endswith('.graphml')]
    # 提取文件中的数字并按数字排序
    def extract_freq_number(file_name):
        match = re.search(r'freq_(\d+)', file_name)  # 使用正则表达式提取 freq_ 后的数字
        return int(match.group(1)) if match else float('inf')  # 返回提取的数字
    # 按 freq_ 后的数字排序文件
    graph_files.sort(key=extract_freq_number)

    # 初始化一个列表来存储读取的频繁子图
    subgraphs = []

    # 循环读取每个子图
    for file in graph_files:
        file_path = os.path.join(directory, file)  # 构建完整的文件路径
        subgraph = nx.read_graphml(file_path)  # 读取单个子图
        subgraphs.append(subgraph)  # 将子图添加到列表中

    with open(directory + name_prefix + "_FSIn.pkl", "rb") as file:
        FSI_node_index = pickle.load(file)
    with open(directory + name_prefix + "_FSIl.pkl", "rb") as file:
        FSI_linear_index = pickle.load(file)
    with open(directory + name_prefix + "_FSIc.pkl", "rb") as file:
        FSI_cycle_index = pickle.load(file)
    with open(directory + name_prefix + "_FSIt.pkl", "rb") as file:
        FSI_build_time = pickle.load(file)
    with open(directory + name_prefix + "_orin.pkl", "rb") as file:
        G_node_index = pickle.load(file)

    G_linear_index = {}
    G1_linear_index = {}
    for max_path in range(2, max_len+1):
        with open(directory + name_prefix + "_" + str(max_path) + "_oril.pkl", "rb") as file:
            G_linear_elem = pickle.load(file)
        with open(directory + name_prefix + "_" + str(max_path) + "_simpl.pkl", "rb") as file:
            G1_linear_elem = pickle.load(file)

        G_linear_index[max_path] = G_linear_elem
        G1_linear_index[max_path] = G1_linear_elem

    with open(directory + name_prefix + "_orit.pkl", "rb") as file:
        G_build_time = pickle.load(file)
    with open(directory + name_prefix + "_simpn.pkl", "rb") as file:
        G1_node_index = pickle.load(file)
    with open(directory + name_prefix + "_simpt.pkl", "rb") as file:
        G1_build_time = pickle.load(file)
    return (G, G1, subgraphs, FSI_node_index, FSI_linear_index, FSI_cycle_index, FSI_build_time, G_node_index,
            G_linear_index, G_build_time, G1_node_index, G1_linear_index, G1_build_time)

def read_info_by_maxlen(directory, name_prefix, max_len, is_mix):


    logging.info(f"reading {directory}{name_prefix}")
    G = nx.read_graphml(directory + name_prefix + "_ori.graphml")
    G1 = nx.read_graphml(directory + name_prefix + "_simp.graphml")

    #graph_files = [f for f in os.listdir(directory) if f.startswith(name_prefix + "_freq_") and f.endswith('.graphml')]
    graph_files = [f for f in os.listdir(directory) if f.startswith(name_prefix + "_freq_") and f.endswith('.graphml')]
    # 提取文件中的数字并按数字排序
    def extract_freq_number(file_name):
        match = re.search(r'freq_(\d+)', file_name)  # 使用正则表达式提取 freq_ 后的数字
        return int(match.group(1)) if match else float('inf')  # 返回提取的数字
    # 按 freq_ 后的数字排序文件
    graph_files.sort(key=extract_freq_number)

    # 初始化一个列表来存储读取的频繁子图
    subgraphs = []
    
    # 循环读取每个子图
    for file in graph_files:
        file_path = os.path.join(directory, file)  # 构建完整的文件路径
        subgraph = nx.read_graphml(file_path)  # 读取单个子图
        subgraphs.append(subgraph)  # 将子图添加到列表中

    with open(directory + name_prefix + "_FSIn.pkl", "rb") as file:
        FSI_node_index = pickle.load(file)
    with open(directory + name_prefix + "_FSIl.pkl", "rb") as file:
        FSI_linear_index = pickle.load(file)
    with open(directory + name_prefix + "_FSIc.pkl", "rb") as file:
        FSI_cycle_index = pickle.load(file)
    with open(directory + name_prefix + "_FSIt.pkl", "rb") as file:
        FSI_build_time = pickle.load(file)
    with open(directory + name_prefix + "_orin.pkl", "rb") as file:
        G_node_index = pickle.load(file)

    with open(directory + name_prefix + "_" + str(max_len) + "_oril.pkl", "rb") as file:
        G_linear_index = pickle.load(file)

    G1_len = max_len - 1 if is_mix else max_len
    with open(directory + name_prefix + "_" + str(G1_len) + "_simpl.pkl", "rb") as file:
        G1_linear_index = pickle.load(file)

    with open(directory + name_prefix + "_orit.pkl", "rb") as file:
        G_build_time = pickle.load(file)
    with open(directory + name_prefix + "_simpn.pkl", "rb") as file:
        G1_node_index = pickle.load(file)
    with open(directory + name_prefix + "_simpt.pkl", "rb") as file:
        G1_build_time = pickle.load(file)
    return (G, G1, subgraphs, FSI_node_index, FSI_linear_index, FSI_cycle_index, FSI_build_time, G_node_index,
            G_linear_index, G_build_time, G1_node_index, G1_linear_index, G1_build_time)

def save_mix_offline(directory, y, embedding_rate, FSI_sizes, SGI_sizes, ORI_sizes, FSI_times, SGI_times, ORI_times, name_prefix):
    save_offline(directory, y, embedding_rate, FSI_sizes, ORI_sizes, FSI_times, SGI_times, ORI_times, name_prefix)
    with open(directory + name_prefix + "_SGIs.pkl", "wb") as file:
        pickle.dump(SGI_sizes, file, protocol=pickle.HIGHEST_PROTOCOL)

def save_offline(directory, y, embedding_rate, FSI_sizes, ORI_sizes, FSI_times, SGI_times, ORI_times, name_prefix):
    with open(directory + name_prefix + "_y.pkl", "wb") as file:
        pickle.dump(y, file, protocol=pickle.HIGHEST_PROTOCOL)
    with open(directory + name_prefix + "_rate.pkl", "wb") as file:
        pickle.dump(embedding_rate, file, protocol=pickle.HIGHEST_PROTOCOL)
    with open(directory + name_prefix + "_FSIs.pkl", "wb") as file:
        pickle.dump(FSI_sizes, file, protocol=pickle.HIGHEST_PROTOCOL)
    with open(directory + name_prefix + "_ORIs.pkl", "wb") as file:
        pickle.dump(ORI_sizes, file, protocol=pickle.HIGHEST_PROTOCOL)
    with open(directory + name_prefix + "_FSIt.pkl", "wb") as file:
        pickle.dump(FSI_times, file, protocol=pickle.HIGHEST_PROTOCOL)
    with open(directory + name_prefix + "_SGIt.pkl", "wb") as file:
        pickle.dump(SGI_times, file, protocol=pickle.HIGHEST_PROTOCOL)
    with open(directory + name_prefix + "_ORIt.pkl", "wb") as file:
        pickle.dump(ORI_times, file, protocol=pickle.HIGHEST_PROTOCOL)

def read_mix_offline(directory, name_prefix):
    y, embedding_rate, FSI_sizes, ORI_sizes, FSI_times, SGI_times, ORI_times = read_offline(directory, name_prefix)
    with open(directory + name_prefix + "_SGIs.pkl", "rb") as file:
        SGI_sizes = pickle.load(file)
    return y, embedding_rate, FSI_sizes, SGI_sizes, ORI_sizes, FSI_times, SGI_times, ORI_times

def read_offline(directory, name_prefix):
    with open(directory + name_prefix + "_y.pkl", "rb") as file:
        y = pickle.load(file)
    with open(directory + name_prefix + "_rate.pkl", "rb") as file:
        embedding_rate = pickle.load(file)
    with open(directory + name_prefix + "_FSIs.pkl", "rb") as file:
        FSI_sizes = pickle.load(file)
    with open(directory + name_prefix + "_ORIs.pkl", "rb") as file:
        ORI_sizes = pickle.load(file)
    with open(directory + name_prefix + "_FSIt.pkl", "rb") as file:
        FSI_times = pickle.load(file)
    with open(directory + name_prefix + "_SGIt.pkl", "rb") as file:
        SGI_times = pickle.load(file)
    with open(directory + name_prefix + "_ORIt.pkl", "rb") as file:
        ORI_times = pickle.load(file)

    return y, embedding_rate, FSI_sizes, ORI_sizes, FSI_times, SGI_times, ORI_times

def save_online(directory, y, embedding_rate, FSI_elapses, ORI_elapses, FSI_sizes, ORI_sizes, name_prefix):
    with open(directory + name_prefix + "_y.pkl", "wb") as file:
        pickle.dump(y, file, protocol=pickle.HIGHEST_PROTOCOL)
    with open(directory + name_prefix + "_rate.pkl", "wb") as file:
        pickle.dump(embedding_rate, file, protocol=pickle.HIGHEST_PROTOCOL)
    with open(directory + name_prefix + "_FSIe.pkl", "wb") as file:
        pickle.dump(FSI_elapses, file, protocol=pickle.HIGHEST_PROTOCOL)
    with open(directory + name_prefix + "_ORIe.pkl", "wb") as file:
        pickle.dump(ORI_elapses, file, protocol=pickle.HIGHEST_PROTOCOL)
    with open(directory + name_prefix + "_FSIs.pkl", "wb") as file:
        pickle.dump(FSI_sizes, file, protocol=pickle.HIGHEST_PROTOCOL)
    with open(directory + name_prefix + "_ORIs.pkl", "wb") as file:
        pickle.dump(ORI_sizes, file, protocol=pickle.HIGHEST_PROTOCOL)

def read_online(directory, name_prefix):
    with open(directory + name_prefix + "_y.pkl", "rb") as file:
        y = pickle.load(file)
    with open(directory + name_prefix + "_rate.pkl", "rb") as file:
        embedding_rate = pickle.load(file)
    with open(directory + name_prefix + "_FSIe.pkl", "rb") as file:
        FSI_elapses = pickle.load(file)
    with open(directory + name_prefix + "_ORIe.pkl", "rb") as file:
        ORI_elapses = pickle.load(file)
    with open(directory + name_prefix + "_FSIs.pkl", "rb") as file:
        FSI_sizes = pickle.load(file)
    with open(directory + name_prefix + "_ORIs.pkl", "rb") as file:
        ORI_sizes = pickle.load(file)
    return y, embedding_rate, FSI_elapses, ORI_elapses, FSI_sizes, ORI_sizes
'''
print(nx.__version__)
read_info("xxx")
'''

