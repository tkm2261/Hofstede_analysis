# -*- coding: utf-8 -*-

import numpy
import scipy
import pandas
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.cluster import hierarchy

import numpy.linalg

CANVAS_SIZE = (20, 20)
plt.figure(figsize=CANVAS_SIZE)

class HofstedeHierarchy(object):

    LIST_DATA_COLUMN = ["PDI", "IDV", "MAS", "UAI"]#, "LTO", "IVR"]
    FILE_NAME_COLUMN= "file"
    MARGIN_RATE = 0.1
    TEXT_MOVE_RATE = 1.
    JAPAN_NAME = u"日本"

    def __init__(self,
                 data_path,
                 data_name_column_idx=2
                ):
        self.data_path = data_path
        self.all_data = pandas.read_csv(data_path)
        self.hofstede_data = numpy.array(self.all_data[self.LIST_DATA_COLUMN],
                                        dtype=numpy.float64)
        ss = StandardScaler()
        self.hofstede_data = ss.fit_transform(self.hofstede_data)
        self.data_name = self.all_data.ix[:, data_name_column_idx]
        self.data_name = self.data_name.apply(lambda x:x.decode("utf-8"))
        self.cluster = None

    def show(self,
             distance_metric='euclidean',
             linkage_method='ward'):

        cluster = hierarchy.linkage(self.hofstede_data,
                                    method=linkage_method,
                                    metric=distance_metric)
        hierarchy.dendrogram(cluster,
                             orientation='left',
                             color_threshold=150,
                             labels=numpy.array(self.data_name),
                             leaf_font_size=18)
        
        ax = plt.gca()
        xlbls = ax.get_ymajorticklabels()
        for lbl in xlbls:
            if lbl.get_text() == self.JAPAN_NAME:
                lbl.set_color("r")

        self.cluster = cluster
        plt.show()

if __name__ == '__main__':

    hh = HofstedeHierarchy('country_data.csv')
    hh.show()