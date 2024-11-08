import logging

import numpy as np
from tqdm import tqdm
import common_util as cu
import plt_util
import sys

def plt_mix():
    max_path = 6
    dir_in = str(sys.argv[1]) + "/"
    y, embedding_rate, FSI_sizes, SGI_sizes, ORI_sizes, FSI_times, SGI_times, ORI_times = cu.read_mix_offline(dir_in, "mix_offline")
    #logging.info(f"y:{y}")
    #logging.info(f"embedding_rate:{embedding_rate}")
    #logging.info(f"FSI_sizes:{FSI_sizes}")
    #logging.info(f"ORI_sizes:{ORI_sizes}")
    #logging.info(f"SGI_times:{SGI_times}")
    #logging.info(f"ORI_times:{ORI_times}")

    legends = ["CSM-Index", "CSM-FSI"]

    # plt index size by embedding_rate
    sizes_by_embedding = []
    sizes_by_embedding.append(ORI_sizes)
    sizes_by_embedding.append(list(np.array(FSI_sizes) + np.array(SGI_sizes) + np.array(ORI_sizes)))
    plt_util.plt_multi_line(embedding_rate, sizes_by_embedding, legends, "embeddings coverage ratio (%)", "index size (bytes)",
                            [0], "", "lower right", (6,5))

    # plt build time by embedding_rate
    times_by_embedding = []
    times_by_embedding.append(ORI_times)
    times_by_embedding.append(list(np.array(FSI_times) + np.array(SGI_times) + np.array(ORI_times)))
    plt_util.plt_multi_line(embedding_rate, times_by_embedding, legends, "embeddings coverage ratio (%)", "index building time (s)",
                            [0], "", "lower right", (6,5))


def plt_by_depth():
    max_path = 6
    dir_in = str(sys.argv[1]) + "/"
    y, embedding_rate, FSI_sizes, ORI_sizes, FSI_times, SGI_times, ORI_times = cu.read_offline(dir_in, "offline")

    legends = ["CSM-Index", "JSM-FSI"]
    path_len = range(2, max_path+1)

    # plt index size by depth
    sizes = []
    sizes.append(ORI_sizes[5])
    sizes.append(FSI_sizes[5])
    plt_util.plt_multi_line(path_len, sizes, legends, "traversal depth in building index", "index size (bytes)", [],
                            "", "upper left", (6,5))

    # plt build time by depth
    times = []
    times.append(ORI_times[5])
    times.append(list(np.array(SGI_times[5]) + FSI_times[5]))
    plt_util.plt_multi_line(path_len, times, legends, "traversal depth in building index", "index building time (s)", [],
                            "", "upper left", (6,5))

def plt_by_embedding():
    max_path = 6
    dir_in = str(sys.argv[1]) + "/"
    y, embedding_rate, FSI_sizes, ORI_sizes, FSI_times, SGI_times, ORI_times = cu.read_offline(dir_in, "offline")
    #logging.info(f"y:{y}")
    #logging.info(f"embedding_rate:{embedding_rate}")
    #logging.info(f"FSI_sizes:{FSI_sizes}")
    #logging.info(f"ORI_sizes:{ORI_sizes}")
    #logging.info(f"SGI_times:{SGI_times}")
    #logging.info(f"ORI_times:{ORI_times}")

    legends = ["CSM-Index", "JSM-FSI"]

    # plt index size by embedding_rate
    sizes_by_embedding = []
    ORI_sizes_by_embedding = []
    FSI_sizes_by_embedding = []
    for ORI_size in ORI_sizes:
        ORI_sizes_by_embedding.append(ORI_size[3])
    for FSI_size in FSI_sizes:
        FSI_sizes_by_embedding.append(FSI_size[3])
    sizes_by_embedding.append(ORI_sizes_by_embedding)
    sizes_by_embedding.append(FSI_sizes_by_embedding)
    plt_util.plt_multi_line(embedding_rate, sizes_by_embedding, legends, "embeddings coverage ratio (%)", "index size (bytes)",
                            [], "", "lower right", (6,5))

    # plt build time by embedding_rate
    times_by_embedding = []
    ORI_times_by_embedding = []
    SGI_times_by_embedding = []
    for ORI_time in ORI_times:
        ORI_times_by_embedding.append(ORI_time[3])
    for SGI_time in SGI_times:
        SGI_times_by_embedding.append(SGI_time[3])
    times_by_embedding.append(ORI_times_by_embedding)
    times_by_embedding.append(list(np.array(SGI_times_by_embedding) + np.array(FSI_times)))
    plt_util.plt_multi_line(embedding_rate, times_by_embedding, legends, "embeddings coverage ratio (%)", "index building time (s)",
                            [], "", "lower right", (6,5))


if __name__ == '__main__':
    cu.custom_logging()
    # Corresponding to Fig 8
    plt_by_depth()
    # Corresponding to Fig 10.(a).(b). Analyse indexes for CSM-Index and JSM-FSI
    plt_by_embedding()
    # Corresponding to Fig 11.(a).(b). Analyse indexes for CSM-FSI
    plt_mix()
