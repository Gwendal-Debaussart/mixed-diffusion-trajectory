# 🔀 Mixed-Diffusion Trajectories

This file contains the instructions and information about the Mixed-Diffusion Trajectories project.

## 🤓 Instructions

``mixed_diffusion.ipynb`` is a Jupyter notebook that contains the code for generating mixed-diffusion trajectories and visualizing the results. To run the notebook, ensure you have the required dependencies installed and follow the steps outlined in the notebook. This notebook serves as a guide for understanding the implementation and results.

Results obtained in the paper are obtained by running the code in ``clustering_benchmark.py``, and ``sensitivity_analysis.py``. These scripts can be run from the command line. Please refer to the comments in the scripts for more details on how to use them.

### 📩 Installation
To install the required dependencies, you can use the following command:

```bash
pip install -r requirements.txt
```

### 📄 Paper Results

The scripts ``clustering_benchmark.py`` and ``sensitivity_analysis.py`` can be executed to reproduce the results presented in the paper. Each script contains comments that explain how to run them and what parameters can be adjusted. Obtained results are available in the ``tables/`` directory. ``clustering_benchmark_raw/`` contains the raw data obtained from running ``clustering_benchmark.py``. ``benchmark_results/`` contains the processed results used to generate the tables in the paper. ``sensitivity_analysis/`` contains the results from running ``sensitivity_analysis.py``, showing how time parameter in MDTs affects clustering performance, both in AMI and ARI.

### Credit

This project was developed by Gwendal Debaussart-Joniec, under the supervision of Prof. Argyris Kalogeratos at the Centre Borelli, ENS Paris-Saclay.
Datasets used in the experiments partially originate from [this github repository](https://github.com/ChuanbinZhang/Multi-view-datasets/).
Synthetic datasets were generated using information from Lindenbaum et al. (2020) and Kuchroo et al. (2022).

## ⚖️ License
This project is licensed under the [GNU General Public License v3.0](LICENSE).