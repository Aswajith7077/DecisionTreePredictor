
class Node:

    def __init__(self,data_points,a_class:int = -1,threshold:int = -1,weighted_gini:int = -1):
        self.threshold = threshold
        self.weighted_gini = weighted_gini
        self.data_points = data_points
        self.feature = ''
        self.associated_class = a_class

        self.left = None
        self.right = None
