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

class HofstedeBiplot(object):
    '''ホフステッド指数をバイプロットするクラス

    :param str data_path: ホフステッド指数が格納されたCSVファイルパス
    :param int data_name_column_idx: 国名が入ったカラムインデックス
    '''
    LIST_DATA_COLUMN = ["PDI", "IDV", "MAS", "UAI", "LTO", "IVR"]
    FILE_NAME_COLUMN= "file"
    MARGIN_RATE = 0.1
    TEXT_MOVE_RATE = 1.1
    JAPAN_NAME = u"日本"
    
    def __init__(self,
                 data_path,
                 data_name_column_idx=2
                ):
        self.data_path = data_path
        self.all_data = pandas.read_csv(data_path)
        self.pca_data = numpy.array(self.all_data[self.LIST_DATA_COLUMN],
                                    dtype=numpy.float64)
        self.data_name = self.all_data.ix[:, data_name_column_idx]


        self.biblot_data, self.pca_score, self.pca_comp = self._calc_biplot()

    def show(self):
        '''バイプロット表示関数
        '''

        #: 各国のプロット
        for i in xrange(len(self.biblot_data)):
            x, y = self.biblot_data[i]
            data_name = self.data_name[i].decode("utf-8")
            if data_name != self.JAPAN_NAME:
                plt.text(x, y, data_name, ha='center', va='center', fontsize=18)
            else:
                plt.text(x, y, data_name, ha='center', va='center', fontsize=18, color="r")

        #: 因子負荷量のプロット
        for i in xrange(len(self.LIST_DATA_COLUMN)):
            x = self.pca_comp[0, i] * self.pca_score[0]
            y = self.pca_comp[1, i] * self.pca_score[1]
            plt.arrow(0, 0, x, y, color='r',
                      width=0.002, head_width=0.04)
            plt.text(x * self.TEXT_MOVE_RATE,
                     y * self.TEXT_MOVE_RATE,
                     self.LIST_DATA_COLUMN[i],
                     color='b', ha='center', va='center', fontsize=18)

        #: 画像表示系の設定
        self._set_canvas()
        plt.show()

    def _calc_biplot(self):
        '''バイプロット用データ計算関数
        '''

        #: 標準化
        ss = StandardScaler()
        pca_data = ss.fit_transform(self.pca_data)

        #: PCA
        pca = PCA(n_components=2)
        biplot_data = pca.fit_transform(pca_data)
 
        #: 固有値
        pca_score = pca.explained_variance_#ratio_

        #: 因子負荷量
        pca_comp = pca.components_
        #biplot_data = biplot_data * pca_score

        return biplot_data, pca_score, pca_comp

    def _set_canvas(self):
        '''画像表示系の設定関数
        '''

        xmin = self.biblot_data[:, 0].min()
        xmax = self.biblot_data[:, 0].max()

        ymin = self.biblot_data[:, 1].min()
        ymax = self.biblot_data[:, 1].max()

        xpad = (xmax - xmin) * self.MARGIN_RATE
        ypad = (ymax - ymin) * self.MARGIN_RATE
        plt.xlim(xmin - xpad, xmax + xpad)
        plt.ylim(ymin - ypad, ymax + ypad)

        plt.xlabel('PC1')
        plt.ylabel('PC2')

if __name__ == '__main__':
 
    hb = HofstedeBiplot('country_data.csv')
    hb.show()
