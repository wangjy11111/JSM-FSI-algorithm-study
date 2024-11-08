import logging

import numpy as np
from tqdm import tqdm
import common_util as cu
import plt_util
import sys

def plt_single_by_mix(dir_in):

    max_query_time = 600
    y, embedding_rates, FSI_elapses, ORI_elapses, FSI_sizes, ORI_sizes = cu.read_online(dir_in,f"mix_{max_query_time}_online")

    match_rates = []
    match_elapses = []

    Mix_match_ratios = []
    ORI_match_ratios = []
    Mix_match_elapses = []
    ORI_match_elapses = []
    for i in range(len(embedding_rates)):
        FSI_size = FSI_sizes[i]
        ORI_size = ORI_sizes[i]
        Mix_size = list(np.array(FSI_size) + np.array(ORI_size))
        FSI_elapse = FSI_elapses[i]
        ORI_elapse = ORI_elapses[i]
        Mix_elapse = [fsi if fsi_size >= 100 else fsi + ori for fsi, fsi_size, ori in zip(FSI_elapse, FSI_size, ORI_elapse)]
        Mix_match_ratio = 100 * len([x for x in Mix_size if x > 0]) / len(Mix_size)
        ORI_match_ratio = 100 * len([x for x in ORI_size if x > 0]) / len(ORI_size)
        Mix_match_elapse = np.log(sum(Mix_elapse) / len(Mix_elapse) if Mix_elapse else 0)
        ORI_match_elapse = np.log(sum(ORI_elapse) / len(ORI_elapse) if ORI_elapse else 0)
        Mix_match_ratios.append(Mix_match_ratio)
        ORI_match_ratios.append(ORI_match_ratio)
        Mix_match_elapses.append(Mix_match_elapse)
        ORI_match_elapses.append(ORI_match_elapse)
    match_rates.append(ORI_match_ratios)
    match_rates.append(Mix_match_ratios)
    match_elapses.append(ORI_match_elapses)
    match_elapses.append(Mix_match_elapses)
    logging.info(f"ORI_match_elapses:{ORI_match_elapses}, Mix_match_elapses:{Mix_match_elapses}")
    legends = ["CSM-Index", "CSM-FSI"]
    plt_util.plt_multi_line(embedding_rates, match_rates, legends, "embeddings coverage ratio (%)", "match ratio (%)",
                            [0], "", "lower right", (6,5))
    plt_util.plt_multi_line(embedding_rates, match_elapses, legends, "embeddings coverage ratio (%)", "log of query elapse (s)",
                            [0], "", "lower right", (6,5))

def plt_single_by_embedding_rate(dir_in):

    max_query_time = 600
    y, embedding_rates, FSI_elapses, ORI_elapses, FSI_sizes, ORI_sizes = cu.read_online(dir_in,f"max_query_time_{max_query_time}_online")

    match_rates = []
    match_elapses = []

    FSI_match_ratios = []
    ORI_match_ratios = []
    FSI_match_elapses = []
    ORI_match_elapses = []
    for i in range(len(embedding_rates)):
        FSI_size = FSI_sizes[i]
        ORI_size = ORI_sizes[i]
        FSI_elapse = FSI_elapses[i]
        ORI_elapse = ORI_elapses[i]
        FSI_match_ratio = 100 * len([x for x in FSI_size if x > 0]) / len(FSI_size)
        ORI_match_ratio = 100 * len([x for x in ORI_size if x > 0]) / len(ORI_size)
        FSI_match_elapse = np.log(sum(FSI_elapse) / len(FSI_elapse) if FSI_elapse else 0)
        ORI_match_elapse = np.log(sum(ORI_elapse) / len(ORI_elapse) if ORI_elapse else 0)
        FSI_match_ratios.append(FSI_match_ratio)
        ORI_match_ratios.append(ORI_match_ratio)
        FSI_match_elapses.append(FSI_match_elapse)
        ORI_match_elapses.append(ORI_match_elapse)
    match_rates.append(ORI_match_ratios)
    match_rates.append(FSI_match_ratios)
    match_elapses.append(ORI_match_elapses)
    match_elapses.append(FSI_match_elapses)
    legends = ["CSM-Index", "JSM-FSI"]
    plt_util.plt_multi_line(embedding_rates, match_rates, legends, "embeddings coverage ratio (%)", "match ratio (%)",
                            [0], "", "lower right", (6,5))
    plt_util.plt_multi_line(embedding_rates, match_elapses, legends, "embeddings coverage ratio (%)", "log of query elapse (s)",
                            [-6], "", "lower right", (6,5))

def plt_single_by_query_size(dir_in):
    query_sizes = range(3, 8)

    match_rates = {}
    match_elapses = {}
    for query_size in query_sizes:
        logging.info(f"begin query_size:{query_size}")
        embedding_count, embedding_rate, FSI_elapses, ORI_elapses, FSI_sizes, ORI_sizes = cu.read_online(dir_in,f"query_size_{query_size}_online")
        logging.info(f"embedding_count:{embedding_count}, embedding_rate:{embedding_rate}")

        FSI_match_ratio = 100 * len([x for x in FSI_sizes if x > 0]) / len(FSI_sizes)
        ORI_match_ratio = 100 * len([x for x in ORI_sizes if x > 0]) / len(ORI_sizes)
        FSI_match_elapse = np.log(sum(FSI_elapses) / len(FSI_elapses) if FSI_elapses else 0)
        ORI_match_elapse = np.log(sum(ORI_elapses) / len(ORI_elapses) if ORI_elapses else 0)
        logging.info(f"FSI_match_rate:{FSI_match_ratio}, ORI_match_rate:{ORI_match_ratio}")
        match_rates[query_size] = tuple((ORI_match_ratio, FSI_match_ratio))
        match_elapses[query_size] = tuple((ORI_match_elapse, FSI_match_elapse))
    logging.info(f"match_rates:{match_rates}")
    plt_util.plt_double_bars(match_rates, ("CSM-Index", "JSM-FSI"), "query graph size (node counts)",
                             "matched ratio (%)", f"embedding_ratio={embedding_rate}%")
    plt_util.plt_double_bars(match_elapses, ("CSM-Index", "JSM-FSI"), "query graph size (node counts)",
                             "log of query elapse time (s)", f"embedding_ratio={embedding_rate}%")

def plt_multi(dir_in):
    max_query_times = [60, 300, 600, 1200]

    match_rates = {}
    match_elapses = {}
    for max_query_time in max_query_times:
        logging.info(f"begin max_query_time:{max_query_time}")
        y, embedding_rates, FSI_elapses, ORI_elapses, FSI_sizes, ORI_sizes = cu.read_online(dir_in,f"max_query_time_{max_query_time}_online")

        for i in range(len(embedding_rates)):
            embedding_rate = embedding_rates[i]
            FSI_size = FSI_sizes[i]
            ORI_size = ORI_sizes[i]
            FSI_elapse = FSI_elapses[i]
            ORI_elapse = ORI_elapses[i]
            FSI_match_ratio = 100 * len([x for x in FSI_size if x > 0]) / len(FSI_size)
            ORI_match_ratio = 100 * len([x for x in ORI_size if x > 0]) / len(ORI_size)
            FSI_match_elapse = np.log(sum(FSI_elapse) / len(FSI_elapse) if FSI_elapse else 0)
            ORI_match_elapse = np.log(sum(ORI_elapse) / len(ORI_elapse) if ORI_elapse else 0)
            logging.info(f"FSI_match_rate:{FSI_match_ratio}, ORI_match_rate:{ORI_match_ratio}")
            if embedding_rate not in match_rates:
                match_rates[embedding_rate] = {}
            if embedding_rate not in match_elapses:
                match_elapses[embedding_rate] = {}
            match_rates[embedding_rate][max_query_time] = tuple((ORI_match_ratio, FSI_match_ratio))
            match_elapses[embedding_rate][max_query_time] = tuple((ORI_match_elapse, FSI_match_elapse))
    logging.info(f"match_rates:{match_rates}")
    for embedding_rate, match_rate_tuples in match_rates.items():
        plt_util.plt_double_bars(match_rate_tuples, ("CSM-Index", "JSM-FSI"), "max query time (s)",
                                 "matched ratio (%)", f"embedding_ratio={embedding_rate}%")
    for embedding_rate, match_elapse_tuples in match_elapses.items():
        plt_util.plt_double_bars(match_elapse_tuples, ("CSM-Index", "JSM-FSI"), "max query time (s)",
                                 "log of query elapse time (s)", f"embedding_ratio={embedding_rate}%")

    exit()

    logging.info(f"y:{y}")
    logging.info(f"embedding_rate:{embedding_rate}")
    logging.info(f"FSI_elapses:{FSI_elapses}")
    logging.info(f"ORI_elapses:{ORI_elapses}")
    logging.info(f"FSI_sizes:{FSI_sizes}")
    logging.info(f"ORI_sizes:{ORI_sizes}")
    logging.info(f"FSI_sizes:{FSI_sizes}")
    legends = [f"embedding_rate={x}" for x in embedding_rate]
    x_label = "Log of JSM-FSI query time (s)"
    y_label = "Log of CSM-Index query time (s)"
    title = "Log-Transformed FSI vs ORI Elapse Time"
    plt_util.plt_viridis_by_log(FSI_elapses, ORI_elapses, legends, x_label, y_label, title)
    plt_util.plt_viridis(FSI_sizes, ORI_sizes, legends, "JSM-FSI answer size", "CSM-Index answer size", "FSI vs ORI answer size")

if __name__ == '__main__':
    cu.custom_logging()
    dir_in = str(sys.argv[1]) + "/"
    #plt_multi(dir_in)
    # Corresponding to Fig 10.(c).(d). Analyse query performance for CSM-Index and JSM-FSI
    plt_single_by_embedding_rate(dir_in)
    # Corresponding to Fig 9
    plt_single_by_query_size(dir_in)
    # Corresponding to Fig 11.(c).(d). Analyse query performance for CSM-FSI
    plt_single_by_mix(dir_in)
