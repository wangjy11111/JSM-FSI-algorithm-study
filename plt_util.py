import logging

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from collections import Counter

def plt_3d(x, y, z, x_label, y_label, z_label):
    # 创建三维图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制三维曲面
    ax.plot_surface(x[:, np.newaxis], y, z, cmap='viridis')  # 添加[:, np.newaxis]以使x成为列向量
    
    # 添加标签
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    
    # 显示图形
    plt.show()

def plt_2d(x, y, x_label, y_label):
    # 创建 2D 图
    #plt.figure(figsize=(6,6))
    plt.figure(figsize=(7,6))

    # 绘制 x 和 y 的关系
    plt.plot(x, y, marker='o', linestyle='-')  # 可以选择不同的 marker 和 linestyle

    # 添加标签
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # 添加标签并设置字体大小
    plt.xlabel(x_label, fontsize=20)
    plt.ylabel(y_label, fontsize=20)

    # 设置刻度字体大小
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    # 调整边距以确保纵轴标签显示完全
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.15)

    # 显示图形
    plt.grid(True)  # 可选: 添加网格
    plt.show()

def plt_3d_comparison(x, y, z1, z2, x_label, y_label, z_label):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 创建网格
    X, Y = np.meshgrid(x, y)

    # 绘制 z1 曲面
    ax.plot_surface(X, Y, z1.T, cmap='viridis', alpha=0.7, label="Z1")  # 转置 z1 匹配 X 和 Y 的形状
    # 绘制 z2 曲面
    ax.plot_surface(X, Y, z2.T, cmap='plasma', alpha=0.7, label="Z2")  # 转置 z2 匹配 X 和 Y 的形状

    # 添加标签
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)

    # 调整视角
    ax.view_init(elev=10, azim=30)

    # 显示图形
    plt.show()

def plt_multi_line(x, datas, legends, x_label, y_label, y_bottoms, legend_label, legend_loc, fig_size):
    font_size = 20
    plt.figure(figsize=fig_size)

    logging.info(f"datas:{datas}")
    logging.info(f"legends:{legends}")
    # 遍历 result 的每个内部列表和 embedding_rate，绘制曲线
    for i, (y_values, legend) in enumerate(zip(datas, legends)):
        if (len(legend_label) > 0):
            plt.plot(x, y_values, label=f'{legend_label}={legend}')
        else:
            plt.plot(x, y_values, label=f'{legend}')

    # 添加标题和标签
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    #plt.title(title, fontsize=20)
    plt.legend(loc=legend_loc, fontsize=font_size)

    # 添加标签并设置字体大小
    plt.xlabel(x_label, fontsize=font_size)
    plt.ylabel(y_label, fontsize=font_size)

    # 设置刻度字体大小
    #plt.xticks(range(int(min(x)), int(max(x))+1), fontsize=font_size)  # 横轴刻度只显示整数
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)

    if (len(y_bottoms)>0):
        plt.ylim(bottom=y_bottoms[0])

    # 调整边距以确保纵轴标签显示完全
    plt.subplots_adjust(left=0.2, right=0.95, top=0.95, bottom=0.15)
    #plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.15)

    # 显示图
    plt.show()

def draw_graph(G):
    # 绘制图 G
    plt.figure(figsize=(8, 6))  # 设置图的大小
    nx.draw(G, with_labels=True, node_color='lightblue', node_size=700, font_size=16, font_color='black',
            font_weight='bold', edge_color='gray')

    # 添加标题
    plt.title("Graph G")

    # 显示图形
    plt.show()
def draw_embedding(G, query, mapping):
    plt.figure(figsize=(10, 6))

    # 绘制大图 G
    pos_G = nx.spring_layout(G)  # 布局
    nx.draw(G, pos_G, with_labels=True, node_color='lightblue', node_size=700, font_size=16, font_color='black',
            font_weight='bold', edge_color='gray')

    # 突出显示映射的节点
    for g_node, h_node in mapping.items():
        plt.scatter(*pos_G[g_node], color='red', s=200)  # 红色标记大图中的节点
        plt.text(pos_G[g_node][0], pos_G[g_node][1], f" {h_node}", color='white', fontsize=12, ha='center',
                 va='center')  # 显示子图节点编号

    # 绘制子图 H
    pos_H = nx.spring_layout(query)
    nx.draw(query, pos_H, with_labels=True, node_color='lightgreen', node_size=700, font_size=16, font_color='black',
            font_weight='bold', edge_color='gray')

    # 添加标题
    plt.title("Graph G with Mapping to Subgraph query")
    plt.show()
def plt_his(sorted_data):
    print(f"sorted_data:{sorted_data}")
    indices = np.arange(len(sorted_data))
    # 绘制柱状图
    plt.figure(figsize=(10, 6))
    #plt.hist(indices, sorted_data, color='lightblue', edgecolor='black', alpha=0.7)  # 设置柱状图的颜色和边缘
    plt.plot(indices, sorted_data, marker='o', color='lightblue', linestyle='-', linewidth=2, markersize=5)
    plt.title('Histogram of Normally Distributed Data (N(100, 50))')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)  # 添加网格
    plt.show()
def plt_distribution(node_types):

    # 计算 node_type 的分布
    node_type_count = Counter(node_types)

    # 打印 node_type 的分布
    print("Node type distribution:", node_type_count)

    # 可视化分布
    plt.bar(node_type_count.keys(), node_type_count.values(), color='lightblue')
    plt.xlabel('Node Type')
    plt.ylabel('Count')
    plt.title('Node Type Distribution in Graph G')
    plt.show()

def plt_viridis_by_log(FSI_elapses, ORI_elapses, legends, x_label, y_label, title):
    # 对数据取对数（以10为底）
    FSI_log = [np.log10(row) for row in FSI_elapses]
    ORI_log = [np.log10(row) for row in ORI_elapses]

    # 创建散点图
    plt.figure(figsize=(6, 6))

    colors = plt.cm.viridis(np.linspace(0, 1, len(FSI_log)))

    for i, (fsi_row, ori_row, legend) in enumerate(zip(FSI_log, ORI_log, legends)):
        plt.scatter(fsi_row, ori_row, color=colors[i], label=legend)

    # 设置坐标轴标签和标题
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    # 设置 x 轴和 y 轴范围为 -3 到 3
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)

    # 显示图例
    plt.legend(loc="best")

    # 显示图形
    plt.show()
def plt_viridis(FSI, ORI, legends, x_label, y_label, title):
    # 创建图形
    plt.figure(figsize=(8, 8))

    # 使用 viridis 颜色映射生成不同颜色
    colors = plt.cm.viridis(np.linspace(0, 1, len(FSI)))

    # 遍历每组数据并绘制散点图
    for i, (fsi_row, ori_row, legend) in enumerate(zip(FSI, ORI, legends)):
        plt.scatter(fsi_row, ori_row, color=colors[i], label=legend)

    # 设置 x 轴和 y 轴范围为一致，例如 -3 到 3（如有需要可以修改）
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.axis('equal')  # 保持 x 和 y 轴比例一致

    # 设置坐标轴标签和标题
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    # 显示图例
    plt.legend(loc="best")

    # 显示图形
    plt.show()
def plt_double_bars(datas, legends, x_label, y_label, title):
    plt.figure(figsize=(6,5))
    # 提取数据
    keys = list(datas.keys())  # 横轴
    values = list(datas.values())  # 每个值是一个 tuple

    # 将 tuple 的两个元素分别拆分为不同的列表
    val1 = [v[0] for v in values]  # 第一个元素
    val2 = [v[1] for v in values]  # 第二个元素

    # 设置柱的宽度和位置
    x = np.arange(len(keys))  # 每个 key 的位置
    bar_width = 0.35  # 每个柱子的宽度
    font_size = 20

    # 绘制柱状图
    plt.bar(x - bar_width/2, val1, width=bar_width, label=legends[0])
    plt.bar(x + bar_width/2, val2, width=bar_width, label=legends[1])

    # 添加标签和标题
    plt.xlabel(x_label, fontsize=font_size)
    plt.ylabel(y_label, fontsize=font_size)
    #plt.title(title, fontsize=font_size)
    plt.xticks(x, keys, fontsize=font_size)  # 将 x 轴刻度设置为 keys 的值
    plt.yticks(fontsize=font_size)
    plt.legend(fontsize=font_size)

    plt.subplots_adjust(left=0.2, right=0.95, top=0.95, bottom=0.15)

    # 显示图形
    plt.show()