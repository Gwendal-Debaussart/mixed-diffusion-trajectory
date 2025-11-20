from utilities.evaluate import evaluate_operator
from .mdt_utils import *
import pandas as pd
import os
import heapq


class Node:
    def __init__(self, operator, parent=None):
        self.operator = operator
        self.parent = parent
        self.children = []
        self.path_operator = self._compute_path_operator()
        self.score = None

    def _compute_path_operator(self):
        """Compute the product of operators from root to this node."""
        if self.parent is None:
            return self.operator
        return self.parent.path_operator @ self.operator

    def expand(self, operators):
        """Generate child nodes with given possible operator values."""
        self.children = [Node(op, parent=self) for op in operators]
        return self.children

    def path(self):
        """Return the sequence of operators from root to this node."""
        node, seq = self, []
        while node:
            seq.append(node.operator)
            node = node.parent
        return list(reversed(seq))


class TreeSearch:
    def __init__(self, root, expand_fn, score_fn, max_depth):
        self.root = root
        self.expand_fn = expand_fn
        self.score_fn = score_fn
        self.max_depth = max_depth

    def search(self):
        raise NotImplementedError("This method should be implemented by subclasses.")

# Implementation of different tree search strategies

class BeamSearch(TreeSearch):
    def __init__(self, root, expand_fn, score_fn, max_depth, beam_width=3):
        super().__init__(root, expand_fn, score_fn, max_depth)
        self.beam_width = beam_width

    def search(self):
        beam = [(self.score_fn(self.root), self.root)]
        for _ in range(self.max_depth):
            candidates = []
            for _, node in beam:
                children = self.expand_fn(node)
                for child in children:
                    score = self.score_fn(child)
                    child.score = score
                    candidates.append((score, child))
            if not candidates:
                break
            beam = heapq.nlargest(self.beam_width, candidates, key=lambda x: x[0])
        return max(beam, key=lambda x: x[0])[1]

def make_score_fn(Xv, true_labels, metric, n_clusters, evaluate_operator):
    def score_fn(node):
        return evaluate_operator(
            node.path_operator,
            Xv,
            true_labels,
            metric,
            n_clusters
        )
    return score_fn

