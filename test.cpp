#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "vptree.h"


namespace py = pybind11;

template <class T, class PyFunc = py::function>
class py_vp_tree : public vp_tree<T, std::function<double(T, T)>, double>
{
public:
    py_vp_tree(const std::vector<T>& points, const PyFunc& distance)
        : vp_tree<T, std::function<double(T, T)>, double>(points, make_std_function(distance))
    {}

    std::function<double(T, T)> make_std_function(const PyFunc& distance)
    {        
        _py_distance = distance;
        return std::function<double(py::object, py::object)>(
            [&](py::object x, py::object y)
            {
                return py::cast<double>(distance(x, y));
            }
        );
    };

private:
    PyFunc _py_distance;
};


PYBIND11_MODULE(vptree, m) {
    m.doc() = "vantage point tree";

    py::class_<py_vp_tree<py::object>>(m, "Vptree")
        .def(py::init<const std::vector<py::object>&, py::function>(), py::keep_alive<1, 2>())
        .def("search", &py_vp_tree<py::object, py::function>::search), py::keep_alive<1, 2>();
}
