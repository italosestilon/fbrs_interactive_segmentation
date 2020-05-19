#get from https://github.com/albertomontesg/davis-interactive/blob/master/davisinteractive/robot/interactive_robot.py
from __future__ import absolute_import, division

import time

import networkx as nx
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion
from scipy.special import comb
from skimage.filters import rank
from skimage.morphology import dilation, disk, erosion, medial_axis
from sklearn.neighbors import radius_neighbors_graph

from isegm.inference.operations import bezier_curve

class InteractiveScribblesRobot(object):
    """ Robot that generates realistic scribbles simulating human interaction.
    # Attributes
        kernel_size: Float. Fraction of the square root of the area used
            to compute the dilation and erosion before computing the
            skeleton of the error masks.
        max_kernel_radius: Float. Maximum kernel radius when applying
            dilation and erosion. Default 16 pixels.
        min_nb_nodes: Integer. Number of nodes necessary to keep a connected
            graph and convert it into a scribble.
        nb_points: Integer. Number of points to sample the bezier curve
            when converting the final paths into curves.
    """

    def __init__(self,
                 kernel_size=.15,
                 max_kernel_radius=16,
                 min_nb_nodes=4,
                 nb_points=1000):
        """ Robot constructor
        """
        if kernel_size >= 1. or kernel_size < 0:
            raise ValueError('kernel_size must be a value between [0, 1).')

        self.kernel_size = kernel_size
        self.max_kernel_radius = max_kernel_radius
        self.min_nb_nodes = min_nb_nodes
        self.nb_points = nb_points

    def _generate_scribble_mask(self, mask):
        """ Generate the skeleton from a mask
        Given an error mask, the medial axis is computed to obtain the
        skeleton of the objects. In order to obtain smoother skeleton and
        remove small objects, an erosion and dilation operations are performed.
        The kernel size used is proportional the squared of the area.
        # Arguments
            mask: Numpy Array. Error mask
        Returns:
            skel: Numpy Array. Skeleton mask
        """
        mask = np.asarray(mask, dtype=np.uint8)
        side = np.sqrt(np.sum(mask > 0))

        mask_ = mask
        # kernel_size = int(self.kernel_size * side)
        kernel_radius = self.kernel_size * side * .5
        kernel_radius = min(kernel_radius, self.max_kernel_radius)

        compute = True
        while kernel_radius > 1. and compute:
            kernel = disk(kernel_radius)
            mask_ = rank.minimum(mask.copy(), kernel)
            mask_ = rank.maximum(mask_, kernel)
            compute = False
            if mask_.astype(np.bool).sum() == 0:
                compute = True
                prev_kernel_radius = kernel_radius
                kernel_radius *= .9

        mask_ = np.pad(
            mask_, ((1, 1), (1, 1)), mode='constant', constant_values=False)
        skel = medial_axis(mask_.astype(np.bool))
        skel = skel[1:-1, 1:-1]
        return skel

    def _mask2graph(self, skeleton_mask):
        """ Transforms a skeleton mask into a graph
        Args:
            skeleton_mask (ndarray): Skeleton mask
        Returns:
            tuple(nx.Graph, ndarray): Returns a tuple where the first element
                is a Graph and the second element is an array of xy coordinates
                indicating the coordinates for each Graph node.
                If an empty mask is given, None is returned.
        """
        mask = np.asarray(skeleton_mask, dtype=np.bool)
        if np.sum(mask) == 0:
            return None

        h, w = mask.shape
        x, y = np.arange(w), np.arange(h)
        X, Y = np.meshgrid(x, y)

        X, Y = X.ravel(), Y.ravel()
        M = mask.ravel()

        X, Y = X[M], Y[M]
        points = np.c_[X, Y]
        G = radius_neighbors_graph(points, np.sqrt(2), mode='distance')
        T = nx.from_scipy_sparse_matrix(G)

        return T, points

    def _acyclics_subgraphs(self, G):
        """ Divide a graph into connected components subgraphs
        Divide a graph into connected components subgraphs and remove its
        cycles removing the edge with higher weight inside the cycle. Also
        prune the graphs by number of nodes in case the graph has not enought
        nodes.
        Args:
            G (nx.Graph): Graph
        Returns:
            list(nx.Graph): Returns a list of graphs which are subgraphs of G
                with cycles removed.
        """
        if not isinstance(G, nx.Graph):
            raise TypeError('G must be a nx.Graph instance')
        S = []  # List of subgraphs of G

        for c in nx.connected_components(G):
            g = G.subgraph(c).copy()

            # Remove all cycles that we may find
            has_cycles = True
            while has_cycles:
                try:
                    cycle = nx.find_cycle(g)
                    weights = np.asarray([G[u][v]['weight'] for u, v in cycle])
                    idx = weights.argmax()
                    # Remove the edge with highest weight at cycle
                    g.remove_edge(*cycle[idx])
                except nx.NetworkXNoCycle:
                    has_cycles = False

            if len(g) < self.min_nb_nodes:
                # Prune small subgraphs
                continue

            S.append(g)

        return S

    def _longest_path_in_tree(self, G):
        """ Given a tree graph, compute the longest path and return it
        Given an undirected tree graph, compute the longest path and return it.
        The approach use two shortest path transversals (shortest path in a
        tree is the same as longest path). This could be improve but would
        require implement it:
        https://cs.stackexchange.com/questions/11263/longest-path-in-an-undirected-tree-with-only-one-traversal
        Args:
            G (nx.Graph): Graph which should be an undirected tree graph
        Returns:
            list(int): Returns a list of indexes of the nodes belonging to the
                longest path.
        """
        if not isinstance(G, nx.Graph):
            raise TypeError('G must be a nx.Graph instance')
        if not nx.is_tree(G):
            raise ValueError('Graph G must be a tree (graph without cycles)')

        # Compute the furthest node to the random node v
        v = list(G.nodes())[0]
        distance = nx.single_source_shortest_path_length(G, v)
        vp = max(distance.items(), key=lambda x: x[1])[0]
        # From this furthest point v' find again the longest path from it
        distance = nx.single_source_shortest_path(G, vp)
        longest_path = max(distance.values(), key=len)
        # Return the longest path

        return list(longest_path)

    def interact(self,
                 pred_masks,
                 gt_masks,
                 nb_objects=1,
                 ignore=-1):
        """ Interaction of the Scribble robot given a prediction.
        Given the sequence and a mask prediction, the robot will return a
        scribble in the region that fails the most.
        # Arguments
            pred_masks: Numpy Array. Array with the prediction masks. It must
				be an integer array with shape (B x H x W), with B being the number
				of frames of the sequence.
            gt_masks: Numpy Array. Array with the ground truth of the sequence.
				It must have the same data type and shape as `pred_masks`.
            nb_objects: Integer. Number of objects in the ground truth mask. If
                `None` the value will be infered from `y_true`. Setting this
                value will speed up the computation.
        # Returns
            dict: Return a scribble (default representation).
        """
        robot_start = time.time()

        prediction = np.asarray(pred_masks, dtype=np.int)
        annotation = np.asarray(gt_masks, dtype=np.int)

        obj_ids = [i for i in range(nb_objects + 1)]
        # Infer height and width of the sequence
        h, w = annotation.shape[0:2]
        img_shape = np.asarray([w, h], dtype=np.float)

        pred, gt = prediction, annotation

        scribbles = [[]]

        worst_frame = 0
        ignore_mask = gt == ignore

        for obj_id in obj_ids:
            start_time = time.time()
            
            gt[ignore_mask] = obj_id
            pred[ignore_mask] = obj_id
            
            error_mask = (gt == obj_id) & (pred != obj_id)
            if error_mask.sum() == 0:
                continue

            # Generate scribbles
            skel_mask = self._generate_scribble_mask(error_mask)
            skel_time = time.time() - start_time
            
            if skel_mask.sum() == 0:
                continue

            G, P = self._mask2graph(skel_mask)

            t_start = time.time()
            S = self._acyclics_subgraphs(G)

            t_start = time.time()
            longest_paths_idx = [self._longest_path_in_tree(s) for s in S]
            longest_paths = [P[idx] for idx in longest_paths_idx]
            t = (time.time() - t_start) * 1000

            t_start = time.time()
            scribbles_paths = [
                bezier_curve(p, self.nb_points) for p in longest_paths
            ]
            t = (time.time() - t_start) * 1000
            
            end_time = time.time()
            
            # Generate scribbles data file
            for p in scribbles_paths[0:1]:
                p /= img_shape
                path_data = {
                    'path': p.tolist(),
                    'object_id': int(obj_id)
                }
                scribbles[worst_frame].append(path_data)

        scribbles_data = {'scribbles': scribbles, 'sequence': 0}

        t = time.time() - robot_start

        return scribbles_data