from ._mdt_tree_utils import *

def mdt_beam(Xp, n_cluster, max_depth=3):
    """
    Perform MDT using Beam Search strategy.
    """
    metric = "chs"
    root = Node(operator=np.eye(Xp[0].shape[0])) # Identity operator as root
    def expand_fn(node):
        return node.expand(Xp)

    score_fn = make_score_fn(Xp, None, metric, n_cluster, evaluate_operator)
    search = BeamSearch(root, expand_fn, score_fn, max_depth=max_depth, beam_width=2)
    best_node = search.search()

    return best_node.path_operator