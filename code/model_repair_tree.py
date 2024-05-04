import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor 
from sklearn.preprocessing import MultiLabelBinarizer
import graphviz
from enum import Enum

class TreeType(Enum):
    """Enum for which types of trees can be created"""
    REGRESSION = 'regression'
    DECISION = 'decision'

class TreeHandler:
    def __init__(self, alignment_values:list(), typ:TreeType):
        self.alignment_values=alignment_values
        self.typ=typ
        self.one_hot = None
        self.clf = None

    def alignments_to_one_hot_df(self, alignment_values):
        """
        List of Lists to pandas Series to one-hot-encoding (pandas df)
        """
        alignment_values_series = pd.Series(alignment_values)
        mlb = MultiLabelBinarizer()
        return pd.DataFrame(mlb.fit_transform(alignment_values_series),
                           columns=mlb.classes_,
                           index=alignment_values_series.index)
    
    def prepare_one_hot(self, case_ids:list()):
        one_hot = self.alignments_to_one_hot_df(self.alignment_values)
        one_hot['case'] = case_ids
        one_hot = one_hot.set_index('case')
        return one_hot

    def create_tree(self, target_values:list(), case_ids:list()):
        if self.typ==TreeType.DECISION:
            clf = tree.DecisionTreeClassifier()
        elif self.typ==TreeType.REGRESSION:
            clf = DecisionTreeRegressor(random_state = 0)
        x = self.one_hot = self.prepare_one_hot(case_ids)
        y = target_values
        clf = clf.fit(x, y)
        self.clf = clf
        return clf
    
    def tree_to_graph(self):
        dot_data = tree.export_graphviz(self.clf, out_file=None, feature_names=self.one_hot.columns[0:len(self.one_hot.columns)])
        graph = graphviz.Source(dot_data)
        return graph