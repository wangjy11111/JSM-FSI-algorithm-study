# JSM-FSI: Jumping Subgraph Matching Based on Frequent Subgraph Indexing

This project corresponds to the paper “JSM-FSI: Jumping Subgraph Matching Based on Frequent Subgraph Indexing.”

## Project File Descriptions

### common_util.py
A basic file providing common methods for reading and writing files.

### plt_util.py
A basic file used for plotting.

### gen_datagraph.py
The main file used to generate simulation data graphs, frequent graphs, and indexes for FSI, SGI, and ORI.

**Usage:**

```bash
python gen_datagraph.py {dir} {path_len}
```
dir: Directory for storing data

path_len: Depth for traversal

### offline_analyse.py
A script for analyzing the offline indexing process.

**Usage:**

```bash
python offline_analyse.py {dir}
```
dir: Directory for storing data

### online_analyse.py
A script for analyzing the online query process.

**Usage:**

```bash
python online_analyse.py {dir}
```
dir: Directory for storing data

### plt_offline.py
A script for plotting the performance curve of index building.

**Usage:**

```bash
python plt_offline.py {dir}
```
dir: Directory for storing data

### plt_online.py
A script for plotting the performance curve of online retrieval.

**Usage:**

```bash
python plt_online.py {dir}
```
dir: Directory for storing data
