from competitors import *
from mdt import *
from .get_diffusion_time import get_diffusion_time
from benchmarks.load_dataset import get_num_clusters

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
        {
            "name": "Contrastive MDT",
            "func": mdt_contrastive,
            "input_type": "preprocessed",
            "task": "manifold_learning",
            "decomp_method": "svd",
            "params": lambda dn: {
                "t": get_diffusion_time(dn),
            },
        },
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
    )
