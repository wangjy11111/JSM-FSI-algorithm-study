import logging

import pickle
import numpy as np
from tqdm import tqdm
import common_util as cu
import sys

def run_mix():
    max_len = 5

    ORI_sizes = []
    FSI_sizes = []
    SGI_sizes = []
    FSI_times = []
    SGI_times = []
    ORI_times = []
    y = np.array(range(1, 21, 4))
    M = 3
    N = 3

    dir_in = str(sys.argv[1]) + "/"
    embedding_rate = []

    for embedding_count in tqdm(y, desc="Processing nodes"):
        (G, G1, subgraphs, FSI_node_index, FSI_linear_index, FSI_cycle_index, FSI_build_time, G_node_index, G_linear_index,
         G_build_times, G1_node_index, G1_linear_index, G1_build_times) =(
            cu.read_info_by_maxlen(dir_in, str(M) + "_" + str(N) + "_" + str(embedding_count), max_len, True))

        total_nodes = sum(len(subgraph.nodes()) for subgraph in subgraphs)
        embedding_rate.append(100.0 * total_nodes * embedding_count/10000)

        FSI_size = (cu.get_total_size(FSI_node_index) + cu.get_total_size(FSI_linear_index) +
                    cu.get_total_size(FSI_cycle_index))
        SGI_size = cu.get_total_size(G1_node_index) + cu.get_total_size(G1_linear_index)
        ORI_size = cu.get_total_size(G_node_index) + cu.get_total_size(G_linear_index)
        logging.info(f"FSI size:{FSI_size}, SGI_size:{SGI_size}, ORI_size:{ORI_size}")

        FSI_sizes.append(FSI_size)
        SGI_sizes.append(SGI_size)
        ORI_sizes.append(ORI_size)

        FSI_time = FSI_build_time
        SGI_time = G1_build_times[max_len-1]
        ORI_time = G_build_times[max_len]
        logging.info(f"FSI_time:{FSI_time}, SGI_time:{SGI_time}, ORI_time:{ORI_time}")

        FSI_times.append(FSI_time)
        SGI_times.append(SGI_time)
        ORI_times.append(ORI_time)

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

    logging.info(f"FSI_sizes:{FSI_sizes}")
    logging.info(f"SGI_sizes:{SGI_sizes}")
    logging.info(f"ORI_sizes:{ORI_sizes}")
    logging.info(f"FSI_times:{FSI_times}")
    logging.info(f"SGI_times:{SGI_times}")
    logging.info(f"ORI_times:{ORI_times}")
    logging.info(f"embedding_rate:{embedding_rate}")

    cu.save_mix_offline(dir_in, y, embedding_rate, FSI_sizes, SGI_sizes, ORI_sizes, FSI_times, SGI_times, ORI_times,
                        "mix_offline")


def run_all_path_len():
    FSI_sizes = []
    ORI_sizes = []
    FSI_times = []
    SGI_times = []
    ORI_times = []
    y = np.array(range(1, 21, 2))
    M = 3
    N = 3
    max_path = 6

    dir_in = str(sys.argv[1]) + "/"
    embedding_rate = []

    for embedding_count in tqdm(y, desc="Processing nodes"):

        FSI_size_elem = []
        ORI_size_elem = []
        SGI_time_elem = []
        ORI_time_elem = []

        (G, G1, subgraphs, FSI_node_index, FSI_linear_index, FSI_cycle_index, FSI_build_time, G_node_index, G_linear_indexs,
         G_build_times, G1_node_index, G1_linear_indexs, G1_build_times) =(
            cu.read_info(dir_in, str(M) + "_" + str(N) + "_" + str(embedding_count)))

        total_nodes = sum(len(subgraph.nodes()) for subgraph in subgraphs)
        embedding_rate.append(100.0 * total_nodes * embedding_count/10000)

        for path_len in range(2, max_path+1):
            G_linear_index = G_linear_indexs[path_len]
            G_build_time = G_build_times[path_len]
            G1_linear_index = G1_linear_indexs[path_len]
            G1_build_time = G1_build_times[path_len]

            FSI_size = (cu.get_total_size(FSI_node_index) + cu.get_total_size(FSI_linear_index) +
                        cu.get_total_size(FSI_cycle_index))
            SGI_size = cu.get_total_size(G1_node_index) + cu.get_total_size(G1_linear_index)
            ORI_size = cu.get_total_size(G_node_index) + cu.get_total_size(G_linear_index)
            logging.info(f"G_node size:{cu.get_total_size(G_node_index)}")
            logging.info(f"G_linear size:{cu.get_total_size(G_linear_index)}")

            FSI_size_elem.append(FSI_size + SGI_size)
            ORI_size_elem.append(ORI_size)
            SGI_time_elem.append(G1_build_time)
            ORI_time_elem.append(G_build_time)
        logging.info(f"embedding_count:{embedding_count}, FSI_size_elem:{FSI_size_elem}")
        logging.info(f"embedding_count:{embedding_count}, ORI_size_elem:{ORI_size_elem}")
        logging.info(f"embedding_count:{embedding_count}, FSI_time_elem:{FSI_build_time}")
        logging.info(f"embedding_count:{embedding_count}, SGI_time_elem:{SGI_time_elem}")
        logging.info(f"embedding_count:{embedding_count}, ORI_time_elem:{ORI_time_elem}")
        logging.info(f"embedding_rate:{100.0 * total_nodes * embedding_count/10000}")
        FSI_sizes.append(FSI_size_elem)
        ORI_sizes.append(ORI_size_elem)
        FSI_times.append(FSI_build_time)
        SGI_times.append(SGI_time_elem)
        ORI_times.append(ORI_time_elem)

        G.clear()
        G1.clear()
        subgraphs.clear()
        FSI_node_index.clear()
        FSI_linear_index.clear()
        FSI_cycle_index.clear()
        G_node_index.clear()
        G_linear_indexs.clear()
        G1_node_index.clear()
        G1_linear_indexs.clear()

    logging.info(f"FSI_sizes:{FSI_sizes}")
    logging.info(f"ORI_sizes:{ORI_sizes}")
    logging.info(f"FSI_times:{FSI_times}")
    logging.info(f"SGI_times:{SGI_times}")
    logging.info(f"ORI_times:{ORI_times}")
    logging.info(f"embedding_rate:{embedding_rate}")

    cu.save_offline(dir_in, y, embedding_rate, FSI_sizes, ORI_sizes, FSI_times, SGI_times, ORI_times, "offline")

def save_result():
    dir_in = str(sys.argv[1]) + "/"
    y = np.array(range(1, 21, 2))
    FSI_sizes = []
    ORI_sizes = []
    FSI_times = []
    SGI_times = []
    ORI_times = []
    embedding_rates = []

    FSI_size_elem=[13700723, 21782840, 51426481, 158016032, 548003648]
    ORI_size_elem=[30992232, 454983080, 2322568179, 10661304858, 45348142129]
    FSI_time_elem=34.66221904754639
    SGI_time_elem=[0.006884098052978516, 0.09057188034057617, 0.3514981269836426, 1.5182569026947021, 6.1732869148254395]
    ORI_time_elem=[0.4969348907470703, 5.623465299606323, 29.18261408805847, 156.25924015045166, 1002.9290881156921]
    embedding_rate=1.96
    FSI_sizes.append(FSI_size_elem)
    ORI_sizes.append(ORI_size_elem)
    FSI_times.append(FSI_time_elem)
    SGI_times.append(SGI_time_elem)
    ORI_times.append(ORI_time_elem)
    embedding_rates.append(embedding_rate)

    FSI_size_elem=[14632920, 41142780, 148665304, 601593347, 2432354415]
    ORI_size_elem=[32564442, 486186508, 2521776095, 11768149194, 50583862931]
    FSI_time_elem=36.09812879562378
    SGI_time_elem=[0.027724027633666992, 0.2697288990020752, 1.6901910305023193, 7.031142234802246, 33.7378351688385]
    ORI_time_elem=[0.5394840240478516, 6.082297086715698, 32.306217193603516, 148.72833800315857, 1500.7311851978302]
    embedding_rate=5.88
    FSI_sizes.append(FSI_size_elem)
    ORI_sizes.append(ORI_size_elem)
    FSI_times.append(FSI_time_elem)
    SGI_times.append(SGI_time_elem)
    ORI_times.append(ORI_time_elem)
    embedding_rates.append(embedding_rate)

    FSI_size_elem=[15778552, 69158211, 312854780, 1460175282, 6306495583]
    ORI_size_elem=[32731886, 489220856, 2505731848, 11635042999, 49771865976]
    FSI_time_elem=36.76656198501587
    SGI_time_elem=[0.04304695129394531, 0.5977292060852051, 4.024820804595947, 22.205798149108887, 128.46034717559814]
    ORI_time_elem=[0.7752590179443359, 6.6520140171051025, 35.54920029640198, 169.8871088027954, 1165.278785943985]
    embedding_rate=9.8
    FSI_sizes.append(FSI_size_elem)
    ORI_sizes.append(ORI_size_elem)
    FSI_times.append(FSI_time_elem)
    SGI_times.append(SGI_time_elem)
    ORI_times.append(ORI_time_elem)
    embedding_rates.append(embedding_rate)

    FSI_size_elem=[16645676, 91803145, 457320548, 2269432858, 10113511072]
    ORI_size_elem=[33434636, 503685138, 2577001250, 11922006736, 50979614385]
    FSI_time_elem=35.64922213554382
    SGI_time_elem=[0.055094003677368164, 0.8745732307434082, 5.200778961181641, 32.829806089401245, 196.12366485595703]
    ORI_time_elem=[0.575901985168457, 6.5297322273254395, 35.665223121643066, 157.14483094215393, 1261.0970323085785]
    embedding_rate=13.72
    FSI_sizes.append(FSI_size_elem)
    ORI_sizes.append(ORI_size_elem)
    FSI_times.append(FSI_time_elem)
    SGI_times.append(SGI_time_elem)
    ORI_times.append(ORI_time_elem)
    embedding_rates.append(embedding_rate)

    FSI_size_elem=[17445974, 113389641, 601948502, 3134300836, 14369701826]
    ORI_size_elem=[34180112, 520386168, 2651078751, 12286281630, 52377389555]
    FSI_time_elem=38.21595811843872
    SGI_time_elem=[0.1931438446044922, 1.31864595413208, 10.25325632095337, 54.24419403076172, 332.0606110095978]
    ORI_time_elem=[0.6763958930969238, 8.338393926620483, 42.700217962265015, 193.35142016410828, 1276.5510642528534]
    embedding_rate=17.64
    FSI_sizes.append(FSI_size_elem)
    ORI_sizes.append(ORI_size_elem)
    FSI_times.append(FSI_time_elem)
    SGI_times.append(SGI_time_elem)
    ORI_times.append(ORI_time_elem)
    embedding_rates.append(embedding_rate)

    FSI_size_elem=[18715427, 152041096, 902032398, 4950858586, 23430255942]
    ORI_size_elem=[35304398, 547487004, 2820971061, 13287965777, 57166650808]
    FSI_time_elem=41.07721710205078
    SGI_time_elem=[0.13754582405090332, 1.8048937320709229, 12.070978164672852, 82.15714406967163, 675.67125415802]
    ORI_time_elem=[0.6650393009185791, 7.501895189285278, 42.7708899974823, 219.34199118614197, 1724.3787660598755]
    embedding_rate=21.56
    FSI_sizes.append(FSI_size_elem)
    ORI_sizes.append(ORI_size_elem)
    FSI_times.append(FSI_time_elem)
    SGI_times.append(SGI_time_elem)
    ORI_times.append(ORI_time_elem)
    embedding_rates.append(embedding_rate)

    FSI_size_elem=[19724952, 183109126, 1156831445, 6566053116, 31522090309]
    ORI_size_elem=[35231398, 544385071, 2773960354, 12873532773, 54780785967]
    FSI_time_elem=40.390037059783936
    SGI_time_elem=[0.14587712287902832, 2.4422922134399414, 17.866436004638672, 111.83203792572021, 883.5071959495544]
    ORI_time_elem=[0.7900311946868896, 8.516230344772339, 43.83786582946777, 218.504714012146, 1664.605626821518]
    embedding_rate=25.48
    FSI_sizes.append(FSI_size_elem)
    ORI_sizes.append(ORI_size_elem)
    FSI_times.append(FSI_time_elem)
    SGI_times.append(SGI_time_elem)
    ORI_times.append(ORI_time_elem)
    embedding_rates.append(embedding_rate)

    FSI_size_elem=[20828792, 217275575, 1450588058, 8478839402, 40986818290]
    ORI_size_elem=[35808448, 559421292, 2862852745, 13382600479, 57198139670]
    FSI_time_elem=35.03879404067993
    SGI_time_elem=[0.1298351287841797, 2.783663272857666, 18.755640983581543, 143.5684049129486, 1236.0874881744385]
    ORI_time_elem=[0.7602658271789551, 8.46761703491211, 45.71880912780762, 239.10718369483948, 1726.931918144226]
    embedding_rate=29.4
    FSI_sizes.append(FSI_size_elem)
    ORI_sizes.append(ORI_size_elem)
    FSI_times.append(FSI_time_elem)
    SGI_times.append(SGI_time_elem)
    ORI_times.append(ORI_time_elem)
    embedding_rates.append(embedding_rate)

    FSI_size_elem=[21970513, 251183688, 1745796819, 10397659661, 50488272194]
    ORI_size_elem=[35902390, 559294225, 2815647083, 12975070529, 54829036381]
    FSI_time_elem=36.9486358165741
    SGI_time_elem=[0.2122659683227539, 3.150970935821533, 25.20711612701416, 198.80004286766052, 1972.0037009716034]
    ORI_time_elem=[0.7013440132141113, 7.851774215698242, 43.97766709327698, 209.35612607002258, 1465.6170570850372]
    embedding_rate=33.32
    FSI_sizes.append(FSI_size_elem)
    ORI_sizes.append(ORI_size_elem)
    FSI_times.append(FSI_time_elem)
    SGI_times.append(SGI_time_elem)
    ORI_times.append(ORI_time_elem)
    embedding_rates.append(embedding_rate)

    FSI_size_elem=[22488550, 269542693, 1931272006, 11551662335, 56025541804]
    ORI_size_elem=[35678532, 554299237, 2748875261, 12452950983, 52259756218]
    FSI_time_elem=34.15408706665039
    SGI_time_elem=[0.20435500144958496, 3.5643579959869385, 25.628440141677856, 214.89737701416016, 1961.8176147937775]
    ORI_time_elem=[0.7357277870178223, 8.201491832733154, 44.353310346603394, 224.0219268798828, 2144.620279073715]
    embedding_rate=37.24
    FSI_sizes.append(FSI_size_elem)
    ORI_sizes.append(ORI_size_elem)
    FSI_times.append(FSI_time_elem)
    SGI_times.append(SGI_time_elem)
    ORI_times.append(ORI_time_elem)
    embedding_rates.append(embedding_rate)

    cu.save_offline(dir_in, y, embedding_rates, FSI_sizes, ORI_sizes, FSI_times, SGI_times, ORI_times, "offline")

def test():
    FSI_sizes = {}
    ORI_sizes = {}
    FSI_times = {}
    SGI_times = {}
    ORI_times = {}
    y = np.array(range(1, 21, 2))
    M = 3
    N = 3
    max_path = 6

    dir_in = str(sys.argv[1]) + "/"
    embedding_rate = []

    for embedding_count in tqdm(y, desc="Processing nodes"):

        FSI_size_elem = []
        ORI_size_elem = []
        SGI_time_elem = []
        ORI_time_elem = []

        with open(dir_in + "3_3_1_FSIn.pkl", "rb") as file:
            FSI_node_index = pickle.load(file)
        with open(dir_in + "3_3_1_FSIl.pkl", "rb") as file:
            FSI_linear_index = pickle.load(file)
        with open(dir_in + "3_3_1_FSIc.pkl", "rb") as file:
            FSI_cycle_index = pickle.load(file)
        with open(dir_in + "3_3_1_FSIt.pkl", "rb") as file:
            FSI_build_time = pickle.load(file)


        for path_len in range(2, max_path+1):

            FSI_size = (cu.get_total_size(FSI_node_index) + cu.get_total_size(FSI_linear_index) +
                        cu.get_total_size(FSI_cycle_index))
            FSI_size_elem.append(FSI_size)

        logging.info(f"embedding_count:{embedding_count}, FSI_size_elem:{FSI_size_elem}")
        logging.info(f"embedding_count:{embedding_count}, ORI_size_elem:{ORI_size_elem}")
        logging.info(f"embedding_count:{embedding_count}, FSI_time_elem:{FSI_build_time}")
        logging.info(f"embedding_count:{embedding_count}, SGI_time_elem:{SGI_time_elem}")
        logging.info(f"embedding_count:{embedding_count}, ORI_time_elem:{ORI_time_elem}")
        FSI_sizes[embedding_count] = FSI_size_elem
        ORI_sizes[embedding_count] = ORI_size_elem
        FSI_times[embedding_count] = FSI_build_time
        SGI_times[embedding_count] = SGI_time_elem
        ORI_times[embedding_count] = ORI_time_elem

    logging.info(f"FSI_sizes:{FSI_sizes}")
    logging.info(f"ORI_sizes:{ORI_sizes}")
    logging.info(f"FSI_times:{FSI_times}")
    logging.info(f"SGI_times:{SGI_times}")
    logging.info(f"ORI_times:{ORI_times}")
    logging.info(f"embedding_rate:{embedding_rate}")

    #cu.save_offline(dir_in, y, embedding_rate, FSI_sizes, ORI_sizes, FSI_times, SGI_times, ORI_times, "offline")

if __name__ == '__main__':
    cu.custom_logging()
    #run_all_path_len()
    run_mix()
    #test()
    #save_result()
