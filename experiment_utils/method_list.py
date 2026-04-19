from competitors.alternating_diffusion import alternating_diffusion
from competitors.composite_diffusion import composite_diffusion_operator
from competitors.cross_diffusion import cross_diffusion_operator
from competitors.gcca import gcca_embedding
from competitors.mvsc import mvsc_embedding
from competitors.integrated_diffusion import integrated_diffusion
from competitors.multiview_diffusion import multiview_diffusion
from competitors.alternating_diffusion import powered_alternating_diffusion
from mdt.random_mdt import random_mdt_operator
from mdt.mdt_direct import mdt_direct
from mdt.mdt_contrastive import mdt_contrastive
from mdt.mdt_tree import mdt_beam

from .get_diffusion_time import get_diffusion_time
from benchmarks.load_dataset import get_num_clusters, get_true_labels

def method_list():
    return (
        {
            "name": "Alternating Diffusion",
            "func": alternating_diffusion,
            "input_type": "preprocessed",
            "decomp_method": "svd",
        },
        {
            "name": "Multi-view Diffusion Maps",
            "func": multiview_diffusion,
            "input_type": "kernels",
            "decomp_method": "eigen",
        },
        {
            "name": "Single-view Diffusion Maps",
            "func": lambda X: X[0],
            "input_type": "preprocessed",
            "single_view": True,
            "decomp_method": "eigen",
        },
        {
            "name": "Integrated Diffusion Maps",
            "func": integrated_diffusion,
            "input_type": "preprocessed",
            "decomp_method": "svd",
        },
        {
            "name": "Composite Diffusion Maps",
            "func": composite_diffusion_operator,
            "input_type": "preprocessed",
            "decomp_method": "svd",
            "n_views": 2,
        },
        {
            "name": "Cross Diffusion Maps",
            "func": cross_diffusion_operator,
            "input_type": "preprocessed",
            "decomp_method": "svd",
        },
        {
            "name": "Powered Alternating Diffusion",
            "func": powered_alternating_diffusion,
            "input_type": "preprocessed",
            "decomp_method": "svd",
        },
        {
            "name": "GCCA",
            "func": gcca_embedding,
            "params": lambda dn: {"n_components": get_num_clusters(dn)},
            "input_type": "preprocessed",
            "decomp_method": "precomputed",
        },
        {
          "name": "GCCA + MDT",
          "func": gcca_embedding,
          "params": lambda dn: {"n_components": get_num_clusters(dn),
                                "diffusion_time": get_diffusion_time(dn)},
          "input_type": "trajectories",
          "decomp_method": "precomputed",
          "stochastic": True,
        },
        {
            "name": "MVSC",
            "func": mvsc_embedding,
            "params": lambda dn: {"n_clusters": get_num_clusters(dn)},
            "input_type": "preprocessed",
            "stochastic": False,
            "decomp_method": "precomputed",
        },
        {
          "name": "MVSC + MDT",
          "func": mvsc_embedding,
          "params": lambda dn: {"n_clusters": get_num_clusters(dn),
                                "diffusion_time": get_diffusion_time(dn)},
          "input_type": "trajectories",
          "decomp_method": "precomputed",
          "stochastic": True,
        },
        {
            "name": "Random Convex MDT",
            "func": random_mdt_operator,
            "input_type": "preprocessed",
            "decomp_method": "svd",
            "stochastic": True,
            "params": lambda dn: {
                "t": get_diffusion_time(dn),
                "convex": True,
            },
        },
        {
            "name": "Random MDT",
            "func": random_mdt_operator,
            "input_type": "preprocessed",
            "decomp_method": "svd",
            "stochastic": True,
            "params": lambda dn: {
                "t": get_diffusion_time(dn),
                "convex": False,
            },
        },
        {
            "name": "Direct MDT",
            "func": mdt_direct,
            "input_type": "preprocessed",
            "decomp_method": "svd",
            "params": lambda dn: {
                "t": get_diffusion_time(dn),
                "k": get_num_clusters(dn),
            },
            "task" : "clustering"
        },
        # {
        #     "name": "Contrastive MDT",
        #     "func": mdt_contrastive,
        #     "input_type": "preprocessed",
        #     "task": "manifold_learning",
        #     "decomp_method": "svd",
        #     "params": lambda dn: {
        #         "t": get_diffusion_time(dn),
        #     },
        # },
        {
            "name": "Beam-Search MDT",
            "func": mdt_beam,
            "input_type": "preprocessed",
            "decomp_method": "svd",
            "params": lambda dn: {
                "n_cluster": get_num_clusters(dn),
                "max_depth": 2*get_diffusion_time(dn),
            },
            "task" : "clustering"
        },
        # ----- Direct MDT using other optimization criteria
        # {
        #     "name": "Direct MDT (DBS)",
        #     "func": mdt_direct,
        #     "input_type": "preprocessed",
        #     "decomp_method": "svd",
        #     "params": lambda dn: {
        #         "t": get_diffusion_time(dn),
        #         "k": get_num_clusters(dn),
        #         "metric": "dbs"
        #     },
        #     "task" : "clustering"
        # },
        # {
        #     "name": "Direct MDT (SIL)",
        #     "func": mdt_direct,
        #     "input_type": "preprocessed",
        #     "decomp_method": "svd",
        #     "params": lambda dn: {
        #         "t": get_diffusion_time(dn),
        #         "k": get_num_clusters(dn),
        #         "metric": "sil"
        #     },
        #     "task" : "clustering"
        # },
        # {
        #     "name": "Direct MDT (AMI)",
        #     "func": mdt_direct,
        #     "input_type": "preprocessed",
        #     "decomp_method": "svd",
        #     "params": lambda dn: {
        #         "t": get_diffusion_time(dn),
        #         "k": get_num_clusters(dn),
        #         "metric": "ami",
        #         "true_labels": get_true_labels(dn)
        #     },
        #     "task" : "clustering"
        # },
    )
