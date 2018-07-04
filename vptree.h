#pragma once
#include <vector>
#include <functional>
#include <numeric>
#include <random>
#include <iterator>
#include <algorithm>
#include <deque>

#include <iostream>

namespace _utils
{
    template <typename Iter, 
              typename RandomGenerator>
    Iter random_choice(Iter start, Iter end, RandomGenerator& g)
    {
        std::uniform_int_distribution<> dis(0, std::distance(start, end) - 1);
        std::advance(start, dis(g));
        return start;
    }
 
    template <typename Iter>
    Iter random_choice(Iter start, Iter end)
    {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        return random_choice(start, end, gen);
    }

    template <typename T>
    T median(std::vector<T> v)
    {
        std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
        return v[v.size() / 2];
    }
}

template <class T, 
          class Func,
          class FuncReturnType = double> 
class vp_tree
{
public:
    vp_tree(const std::vector<T>& points, const Func& distance)
        : _points(points)
        , _distance(distance)
    {
        _build();
    }

    std::vector<T> search(const T& point, size_t count) const
    {
        return _search_impl(point, count);
    }

private:
    void _build()
    {
        tree_data = std::vector<std::vector<FuncReturnType>>(
            _points.size(),
            std::vector<FuncReturnType>(_impl::TREE_DATA_WIDTH, FuncReturnType(-1))            
        );
        size_t counter = 0;
        int node_index = -1;
        std::vector<std::vector<size_t>> queue;
        queue.reserve(_points.size());
        
        std::vector<size_t> node(_points.size());
        std::iota(node.begin(), node.end(), 0);
        node.push_back(counter++);
        queue.push_back(node);

        while (queue.size() > 0)
        {
            std::vector<size_t> subtree_point_indexes = queue.back();
            queue.pop_back();
            node_index = subtree_point_indexes.back();
            subtree_point_indexes.pop_back();

            size_t subtree_size = subtree_point_indexes.size();
            int vp_id = -1;
            int left_child_id = -1;
            int right_child_id = -1;
            FuncReturnType median = -1;

            std::vector<size_t> left_side, right_side;

            if (subtree_size > 0)
            {
                if (subtree_size == 1)
                {
                    vp_id = subtree_point_indexes[0];
                }
                else
                {   
                    auto vp_id_index = _pick_vantage_point(subtree_point_indexes);
                    vp_id = *vp_id_index;
                    subtree_point_indexes.erase(vp_id_index);
                    
                    std::vector<FuncReturnType> distances = get_distances(
                        subtree_point_indexes, vp_id);
                        
                    median = _utils::median(distances);

                    left_side.clear();
                    right_side.clear();

                    for (size_t i = 0; i < subtree_size - 1; ++i)
                    {
                        if (distances[i] <= median)
                        {
                            left_side.push_back(subtree_point_indexes[i]);
                        }
                        else
                        {
                            right_side.push_back(subtree_point_indexes[i]);
                        }                            
                    }

                    left_child_id = -1;
                    if (left_side.size() > 0)
                    {
                        left_child_id = counter++;
                        left_side.push_back(left_child_id);
                        queue.push_back(left_side);
                    }
                        
                    right_child_id = -1;
                    if (right_side.size() > 0)
                    {
                        right_child_id = counter++;
                        right_side.push_back(right_child_id);
                        queue.push_back(right_side);
                    }                        
                }

                tree_data[node_index][_impl::SIZE_IDX] = (FuncReturnType)subtree_size;
                tree_data[node_index][_impl::MEDIAN_IDX] = median;
                tree_data[node_index][_impl::VP_POINT_IDX] = (FuncReturnType)vp_id;
                tree_data[node_index][_impl::LEFT_CHILD_IDX] = (FuncReturnType)left_child_id;
                tree_data[node_index][_impl::RIGHT_CHILD_IDX] = (FuncReturnType)right_child_id;
            }
        }
    }

    std::vector<size_t>::const_iterator _pick_vantage_point(const std::vector<size_t>& points) const
    {
        return _utils::random_choice(points.begin(), points.end());
    }

    std::vector<FuncReturnType> get_distances(
        const std::vector<size_t>& point_ids, const size_t vp_id) const
    {
        std::vector<FuncReturnType> result(point_ids.size(), -1);

        for (size_t i = 0; i < point_ids.size(); ++i)
        {
            result[i] = _distance(_points[point_ids[i]], _points[vp_id]);
        }
        return result;
    }

    std::vector<T> _search_impl(const T& point, size_t count) const
    {
        FuncReturnType tau = std::numeric_limits<FuncReturnType>::max();
        std::vector<size_t> node_queue;
        std::deque<std::vector<FuncReturnType>> neighbors;

        node_queue.push_back(0);

        while (node_queue.size() > 0)
        {
            int node_index = node_queue.back();

            node_queue.pop_back();
            if (node_index == -1)
            {
                continue;
            }

            std::vector<FuncReturnType> node_row = tree_data[node_index];

            int vp_index = int(node_row[_impl::VP_POINT_IDX]);
            FuncReturnType distance = _distance(point, _points[vp_index]);
            if (neighbors.size() < count)
            {
                neighbors.push_back({-distance, (FuncReturnType)vp_index});

                std::sort(neighbors.begin(), neighbors.end());
            }
            else if (distance < tau)
            {
                neighbors.push_back({-distance, (FuncReturnType)vp_index});
                std::sort(neighbors.begin(), neighbors.end());

                if (neighbors.size() > count)
                {
                    neighbors.pop_front();
                }

                tau = neighbors[0][0] * -1;
            }

            FuncReturnType median = node_row[_impl::MEDIAN_IDX];
            if (median == -1)
            {
                continue;
            }                

            if (distance < median)
            {
                if (distance < median + tau)
                {
                    node_queue.push_back(int(node_row[_impl::LEFT_CHILD_IDX]));
                }
                    
                if (distance >= median - tau)
                {
                    node_queue.push_back(int(node_row[_impl::RIGHT_CHILD_IDX]));
                }                    
            }
            else
            {
                if (distance < median + tau)
                {
                    node_queue.push_back(int(node_row[_impl::LEFT_CHILD_IDX]));
                }
                    
                if (distance >= median - tau)
                {
                    node_queue.push_back(int(node_row[_impl::RIGHT_CHILD_IDX]));
                }                    
            }            
        }
         
        std::sort(neighbors.rbegin(), neighbors.rend());   
                
        std::vector<T> result;
        for (auto row : neighbors)
        {
            result.push_back(_points[row[1]]);
        }
        return result;
    }

protected:
    const std::vector<T> _points;
    const Func _distance;
    std::vector<std::vector<FuncReturnType>> tree_data;

    enum _impl
    {
        SIZE_IDX = 0,
        MEDIAN_IDX = 1,
        VP_POINT_IDX = 2,
        LEFT_CHILD_IDX = 3,
        RIGHT_CHILD_IDX = 4,
        TREE_DATA_WIDTH = 5
    } vp_tree_indexes;
};