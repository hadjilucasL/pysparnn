# Copyright (c) 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Defines a distance search structure"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import abc as _abc
import numpy as _np
import scipy.sparse as _sparse
import scipy.spatial.distance as _spatial_distance


class MatrixMetricSearch(object):
    """A matrix representation out of features."""
    __metaclass__ = _abc.ABCMeta

    def __init__(self, features, records_data):
        """
        Args:
            features: A matrix with rows that represent records
                (corresponding to the elements in records_data) and columns
                that describe a point in space for each row.
            records_data: Data to return when a doc is matched. Index of
                corresponds to features.
        """
        self.matrix = features
        self.records_data = _np.array(records_data)

    def get_feature_matrix(self):
        return self.matrix

    def get_records(self):
        return self.records_data

    @staticmethod
    @_abc.abstractmethod
    def features_to_matrix(features):
        """
        Args:
            val: A list of features to be formatted.
        Returns:
            The transformed matrix.
        """
        return

    @staticmethod
    @_abc.abstractmethod
    def vstack(matrix_list):
        """
        Args:
            val: A list of features to be formatted.
        Returns:
            The transformed matrix.
        """
        return

    @_abc.abstractmethod
    def _transform_value(self, val):
        """
        Args:
            val: A numeric value to be (potentially transformed).
        Returns:
            The transformed numeric value.
        """
        return

    @_abc.abstractmethod
    def _distance(self, a_matrix):
        """
        Args:
            a_matrix: A matrix with rows that represent records
                to search against.
            records_data: Data to return when a doc is matched. Index of
                corresponds to features.
        Returns:
            A dense array representing distance.
        """
        return

    def nearest_search(self, features):
        """Find the closest item(s) for each set of features in features_list.

        Args:
            features: A matrix with rows that represent records
                (corresponding to the elements in records_data) and columns
                that describe a point in space for each row.

        Returns:
            For each element in features_list, return the k-nearest items
            and their distance scores
            [[(score1_1, item1_1), ..., (score1_k, item1_k)],
             [(score2_1, item2_1), ..., (score2_k, item2_k)], ...]
        """

        dist_matrix = self._distance(features)

        ret = []
        for i in range(dist_matrix.shape[0]):
            # replacing the for loop by matrix ops could speed things up

            scores = dist_matrix[i]
            records = self.records_data

            arg_index = _np.argsort(scores)

            curr_ret = list(zip(scores[arg_index], records[arg_index]))

            ret.append(curr_ret)

        return ret

    def nearest_search_vec(self, features):
        """Find the closest item(s) for each set of features in features_list.

        Args:
            features: A matrix with rows that represent records
                (corresponding to the elements in records_data) and columns
                that describe a point in space for each row.

        Returns:
            scores_full (numpy.array): 2D array where each record has its columns,
                                        distance scoress, sorted in ascending order
            records_full (numpy.array): 2D array where each record has its items
                                        sorted in ascending order of distance
        """

        # get distances
        dist_matrix = self._distance(features)

        # sort each row in ascending order of distances
        arg_index = _np.argsort(dist_matrix)

        # detect and randomise rows for which there is no clear order of distances
        scores_zero_distance = dist_matrix.sum(axis=1) < 0.00001
        arg_index[scores_zero_distance] = self.scramble(arg_index[scores_zero_distance])

        # sort scores row-wise depending on arg_index
        scores_full = dist_matrix[_np.arange(_np.shape(dist_matrix)[0])[:, _np.newaxis], arg_index]

        # sort records row-wise depending on arg_index
        records_full = self.records_data[_np.newaxis, :]
        records_full = _np.repeat(records_full, dist_matrix.shape[0], axis=0)
        records_full = records_full[_np.arange(_np.shape(dist_matrix)[0])[:, _np.newaxis], arg_index]

        return scores_full, records_full

    def scramble(self, a, axis=-1):
        """
        Return an array with the values of `a` independently shuffled along the
        given axis
        """
        b =_np.random.random(a.shape)
        idx = _np.argsort(b, axis=axis)
        shuffled = a[_np.arange(a.shape[0])[:, None], idx]
        return shuffled

    def remove_near_duplicates(self):
        """If there are 2 or more records with 0 distance from eachother - 
        keep only one. 
        """

        dist_matrix = self._distance(self.matrix)

        keeps = []
        dupes = set()
        for row_index in range(dist_matrix.shape[0]):
            max_dist = dist_matrix[row_index].max()
            for col_index in range(dist_matrix.shape[0]):
                if row_index < col_index:
                    if dist_matrix[row_index, col_index] / max_dist <= 0.001:
                        dupes.add(col_index)
            if not row_index in dupes:
                keeps.append(row_index)

        self.matrix = self.matrix[keeps]
        self.records = self.records_data[keeps]


class CosineDistance(MatrixMetricSearch):
    """A matrix that implements cosine distance search against it.

    cosine_distance = 1 - cosine_similarity

    Note: We want items that are more similar to be closer to zero so we are
    going to instead return 1 - cosine_similarity. We do this so similarity
    and distance metrics can be treated the same way.
    """

    def __init__(self, features, records_data):
        super(CosineDistance, self).__init__(features, records_data)

        m_c = self.matrix.copy()
        m_c.data **= 2
        self.matrix_root_sum_square = \
                _np.sqrt(_np.asarray(m_c.sum(axis=1)).reshape(-1))

    @staticmethod
    def features_to_matrix(features):
        """
        Args:
            val: A list of features to be formatted.
        Returns:
            The transformed matrix.
        """
        return _sparse.csr_matrix(features)

    @staticmethod
    def vstack(matrix_list):
        """
        Args:
            val: A list of features to be formatted.
        Returns:
            The transformed matrix.
        """
        return _sparse.vstack(matrix_list)

    def _transform_value(self, v):
        return v

    def _distance(self, a_matrix):
        """Vectorised cosine distance"""
        # what is the implmentation of transpose? can i change the order?
        dprod = self.matrix.dot(a_matrix.transpose()).transpose() * 1.0

        a_c = a_matrix.copy()
        a_c.data **= 2
        a_root_sum_square = _np.asarray(a_c.sum(axis=1)).reshape(-1)
        a_root_sum_square = \
                a_root_sum_square.reshape(len(a_root_sum_square), 1)
        a_root_sum_square = _np.sqrt(a_root_sum_square)

        magnitude = 1.0 / (a_root_sum_square * self.matrix_root_sum_square)

        return 1 - dprod.multiply(magnitude).toarray()

class UnitCosineDistance(MatrixMetricSearch):
    """A matrix that implements cosine distance search against it.

    cosine_distance = 1 - cosine_similarity

    Note: We want items that are more similar to be closer to zero so we are
    going to instead return 1 - cosine_similarity. We do this so similarity
    and distance metrics can be treated the same way.

    Assumes unit-vectors and takes some shortucts:
      * Uses integers instead of floats
      * 1**2 == 1 so that operation can be skipped
    """

    def __init__(self, features, records_data):
        super(UnitCosineDistance, self).__init__(features, records_data)
        self.matrix_root_sum_square = \
                _np.sqrt(_np.asarray(self.matrix.sum(axis=1)).reshape(-1))

    @staticmethod
    def features_to_matrix(features):
        """
        Args:
            val: A list of features to be formatted.
        Returns:
            The transformed matrix.
        """
        return _sparse.csr_matrix(features)

    @staticmethod
    def vstack(matrix_list):
        """
        Args:
            val: A list of features to be formatted.
        Returns:
            The transformed matrix.
        """
        return _sparse.vstack(matrix_list)

    def _transform_value(self, v):
        return 1

    def _distance(self, a_matrix):
        """Vectorised cosine distance"""
        # what is the implmentation of transpose? can i change the order?
        dprod = self.matrix.dot(a_matrix.transpose()).transpose() * 1.0

        a_root_sum_square = _np.asarray(a_matrix.sum(axis=1)).reshape(-1)
        a_root_sum_square = \
                a_root_sum_square.reshape(len(a_root_sum_square), 1)
        a_root_sum_square = _np.sqrt(a_root_sum_square)

        magnitude = 1.0 / (a_root_sum_square * self.matrix_root_sum_square)

        return 1 - dprod.multiply(magnitude).toarray()

class SlowEuclideanDistance(MatrixMetricSearch):
    """A matrix that implements euclidean distance search against it.
    WARNING: This is not optimized.
    """

    def __init__(self, features, records_data):
        super(SlowEuclideanDistance, self).__init__(features, records_data)
        self.matrix = self.matrix

    @staticmethod
    def features_to_matrix(features):
        """
        Args:
            val: A list of features to be formatted.
        Returns:
            The transformed matrix.
        """
        return _np.array(features, ndmin=2)

    @staticmethod
    def vstack(matrix_list):
        """
        Args:
            val: A list of features to be formatted.
        Returns:
            The transformed matrix.
        """
        return _np.vstack(matrix_list)

    def _transform_value(self, v):
        return v

    def _distance(self, a_matrix):
        """Euclidean distance"""

        return _spatial_distance.cdist(a_matrix, self.matrix, 'euclidean')

class DenseCosineDistance(MatrixMetricSearch):
    """A matrix that implements cosine distance search against it.

    cosine_distance = 1 - cosine_similarity

    Note: We want items that are more similar to be closer to zero so we are
    going to instead return 1 - cosine_similarity. We do this so similarity
    and distance metrics can be treated the same way.
    """

    def __init__(self, features, records_data):
        super(DenseCosineDistance, self).__init__(features, records_data)

        self.matrix_root_sum_square = \
                _np.sqrt((self.matrix**2).sum(axis=1).reshape(-1))

    @staticmethod
    def features_to_matrix(features):
        """
        Args:
            val: A list of features to be formatted.
        Returns:
            The transformed matrix.
        """
        return _np.array(features, ndmin=2)

    @staticmethod
    def vstack(matrix_list):
        """
        Args:
            val: A list of features to be formatted.
        Returns:
            The transformed matrix.
        """
        return _np.vstack(matrix_list)

    def _transform_value(self, v):
        return v

    def _distance(self, a_matrix):
        """Vectorised cosine distance"""
        # what is the implmentation of transpose? can i change the order?
        dprod = self.matrix.dot(a_matrix.transpose()).transpose() * 1.0

        a_root_sum_square = (a_matrix**2).sum(axis=1).reshape(-1)
        a_root_sum_square = a_root_sum_square.reshape(len(a_root_sum_square), 1)
        a_root_sum_square = _np.sqrt(a_root_sum_square)

        magnitude = 1.0 / (a_root_sum_square * self.matrix_root_sum_square)

        return 1 - (dprod * magnitude)
