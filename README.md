# 🔀 Mixed-Diffusion Trajectories

This file contains the instructions and information about the Mixed-Diffusion Trajectories project.

## 🤓 Instructions

``mixed_diffusion.ipynb`` is a Jupyter notebook that contains the code for generating mixed-diffusion trajectories and visualizing the results. To run the notebook, ensure you have the required dependencies installed and follow the steps outlined in the notebook. This notebook serves as a guide for understanding the implementation and results.

### 📩 Installation
To install the required dependencies, you can use the following command:

```bash
pip install -r requirements.txt
```

### 📄 Paper Results

The scripts ``clustering_benchmark.py`` and ``sensitivity_analysis.py`` can be executed to reproduce the results presented in the paper. Each script contains comments that explain how to run them and what parameters can be adjusted, please refer to those.

Results showcased in the paper are available in the ``tables/`` directory.
- ``clustering_benchmark_raw/`` contains the raw data obtained from running ``clustering_benchmark.py`` (i.e each run for each method, and each clustering score).
- ``benchmark_results/`` contains the processed results used to generate the tables in the paper.
- ``sensitivity_analysis/`` contains the results from running ``sensitivity_analysis.py``, showing how time parameter in MDTs affects clustering performance, both in AMI and ARI. Additional measures of time-wise correlation between CH and AMI are also provided.

### Credit

This project was developed by Gwendal Debaussart-Joniec, under the supervision of Argyris Kalogeratos at the Centre Borelli, ENS Paris-Saclay.
Datasets used in the experiments partially originate from [this github repository](https://github.com/ChuanbinZhang/Multi-view-datasets/). Synthetic datasets were generated using information from Lindenbaum et al. (2020) and Kuchroo et al. (2022).

### 📄 Citation

Please cite the following paper if you use this code in your research:

```bibtex
@misc{debaussartjoniec2025multiviewdiffusiongeometryusing,
      title={Multi-view diffusion geometry using intertwined diffusion trajectories},
      author={Gwendal Debaussart-Joniec and Argyris Kalogeratos},
      year={2025},
      eprint={2512.01484},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2512.01484},
}
```

You can also find the paper on [arXiv](https://arxiv.org/abs/2512.01484).

## ⚖️ License
This project is licensed under the [GNU General Public License v3.0](LICENSE).