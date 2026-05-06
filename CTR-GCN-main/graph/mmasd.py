import numpy as np


class Graph:

    def __init__(self, labeling_mode='spatial'):

        self.num_node = 24

        self.self_link = [(i, i) for i in range(self.num_node)]

        self.inward = [

            (1, 0),
            (2, 1),
            (3, 2),
            (4, 3),

            (5, 0),
            (6, 5),
            (7, 6),
            (8, 7),

            (9, 0),
            (10, 9),
            (11, 10),

            (12, 0),
            (13, 12),
            (14, 13),

            (15, 0),
            (16, 15),
            (17, 16),

            (18, 0),
            (19, 18),
            (20, 19),

            (21, 0),
            (22, 21),
            (23, 22),
        ]

        self.outward = [(j, i) for (i, j) in self.inward]

        self.neighbor = self.inward + self.outward

        self.A = self.get_adjacency_matrix()

    def get_adjacency_matrix(self):

        A = np.zeros((self.num_node, self.num_node))

        for i, j in self.self_link:
            A[i, j] = 1

        for i, j in self.neighbor:
            A[i, j] = 1

        return A