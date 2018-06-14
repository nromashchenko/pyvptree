#cython: language_level=3, boundscheck=False, infer_types=True, c_string_type=unicode, c_string_encoding=utf8, profile=True
from typing import TypeVar, List, Callable, Tuple

#from Cython.Compiler.Options import directive_defaults
#directive_defaults['linetrace'] = True
#directive_defaults['binding'] = True

import numpy as np
cimport numpy as np
from libcpp.vector cimport vector

import queue
import random

cdef:
    int SIZE_IDX = 0
    int MEDIAN_IDX = 1
    int VP_POINT_IDX = 2
    int LEFT_CHILD_IDX = 3
    int RIGHT_CHILD_IDX = 4

    int TREE_DATA_WIDTH = 5

A = TypeVar('A')

ctypedef double DATA_TYPE_t
ctypedef np.ndarray SAMPLE_t
ctypedef np.float64_t (*DISTANCE_FUNC_t)(np.ndarray[DATA_TYPE_t, ndim=1],
                                         np.ndarray[DATA_TYPE_t, ndim=1])


cdef class Counter:
    cdef:
        int value

    def __cinit__(self, start: int):
        self.value = start

    cdef int next(self):
        cdef int old_value = self.value
        self.value += 1
        return old_value

class BaseVpTree:
    def __init__(self, objects: List[A], distance_func: Callable):
        self.objects = objects
        self.tree_data = np.zeros(shape=(len(objects), TREE_DATA_WIDTH), dtype=np.float)
        self.distance_func = distance_func
        self._build_impl()

    def _build_impl(self):
        cdef:
            int size = 0, left_child_id = -1, right_child_id = -1
            int diff = 0, additional_row_size = 0, data_size = 0
            float median = 0.0

            vector[double] distances
            vector[int] node_point_idx, left_side, right_side
            vector[vector[int]] queue
            vector[int] queued_task

            int vp_id = -1, vp_id_index = -1
            int point_id = -1, node_index = 0

            Counter node_id_counter = Counter(0)

        node_index = node_id_counter.next()
        for i in range(len(self.objects)):
            node_point_idx.push_back(i)

        node_point_idx.push_back(node_index)
        queue.push_back(node_point_idx)

        distances = vector[double](len(self.objects))

        while queue.size() > 0:
            node_point_idx = queue.back()
            node_index = node_point_idx[node_point_idx.size() - 1]
            queue.pop_back()

            size = node_point_idx.size() - 1

            if size > 0:
                if size == 1:
                    vp_id = node_point_idx[0]
                    median = 0
                    left_child_id = -1
                    right_child_id = -1
                else:
                    vp_id_index = random.randrange(size)
                    vp_id = node_point_idx[vp_id_index]
                    node_point_idx.erase(node_point_idx.begin() + vp_id_index)

                    for i in range(size - 1):
                        point_id = node_point_idx[i]
                        distances[i] = self.distance_func(self.objects[point_id], self.objects[vp_id])

                    median = np.median(distances)

                    left_side.clear()
                    right_side.clear()

                    for i in range(size - 1):
                        if distances[i] <= median:
                            left_side.push_back(node_point_idx[i])
                        else:
                            right_side.push_back(node_point_idx[i])

                    left_child_id = -1
                    if left_side.size() > 0:
                        left_child_id = node_id_counter.next()
                        left_side.push_back(left_child_id)
                        queue.push_back(left_side)

                    right_child_id = -1
                    if right_side.size() > 0:
                        right_child_id = node_id_counter.next()
                        right_side.push_back(right_child_id)
                        queue.push_back(right_side)

                self.tree_data[node_index, SIZE_IDX] = size
                self.tree_data[node_index, MEDIAN_IDX] = median
                self.tree_data[node_index, VP_POINT_IDX] = vp_id
                self.tree_data[node_index, LEFT_CHILD_IDX] = left_child_id
                self.tree_data[node_index, RIGHT_CHILD_IDX] = right_child_id


    def search(self, point: A, count: int) -> Tuple[np.array, np.array]:
        result = self._search_impl(point, count)
        result = sorted([(-value, point) for value, point in result])
        values, points = zip(*result)
        return np.array(values), np.array(points)

    def _search_impl(self, point: A, count: int) -> list:
        cdef:
            int node_index = -1
            float tau = np.inf, distance, median
            vector[int] node_queue
            np.ndarray[double, ndim=1] vantage_point, node_row

        node_queue.push_back(0)
        neighbors = queue.PriorityQueue()

        while len(node_queue) > 0:
            node_index = node_queue.back()
            node_queue.pop_back()
            if node_index == -1:
                continue

            node_row = self.tree_data[node_index]
            vantage_point = self.objects[int(node_row[VP_POINT_IDX])]
            distance = self.distance_func(point, vantage_point)

            if len(neighbors.queue) < count:
                neighbors.put((-distance, vantage_point))
            elif distance < tau:
                neighbors.put((-distance, vantage_point))
                if len(neighbors.queue) > count:
                    neighbors.get()

                tau, _ = neighbors.queue[0]
                tau *= -1

            median = node_row[MEDIAN_IDX]
            if not median:
                continue

            if distance < median:
                if distance < median + tau:
                    node_queue.push_back(int(node_row[LEFT_CHILD_IDX]))
                if distance >= median - tau:
                    node_queue.push_back(int(node_row[RIGHT_CHILD_IDX]))
            else:
                if distance < median + tau:
                    node_queue.push_back(int(node_row[LEFT_CHILD_IDX]))
                if distance >= median - tau:
                    node_queue.push_back(int(node_row[RIGHT_CHILD_IDX]))

        return neighbors.queue