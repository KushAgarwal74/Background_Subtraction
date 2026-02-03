#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "gmm_model.hpp"

namespace py = pybind11;

PYBIND11_MODULE(gmm_cpp, m) {
    m.doc() = "GMM Background Subtraction C++ backend (pybind11)";

    py::class_<GMMModel>(m, "GMMModel")
        .def(py::init([](py::array_t<float, py::array::c_style | py::array::forcecast> first_frame,
                         int k, float alpha, float threshold, float bg_threshold) {
            auto buf = first_frame.request();
            if (buf.ndim != 2) {
                throw std::runtime_error("first_frame must be a 2D float32 array (H,W)");
            }
            int H = (int)buf.shape[0];
            int W = (int)buf.shape[1];
            const float* ptr = (const float*)buf.ptr;

            return new GMMModel(H, W, k, alpha, threshold, bg_threshold, ptr);
        }),
        py::arg("first_frame"),
        py::arg("k") = 3,
        py::arg("alpha") = 0.01f,
        py::arg("threshold") = 2.5f,
        py::arg("bg_threshold") = 0.7f)

        .def("apply", [](GMMModel& self,
                         py::array_t<float, py::array::c_style | py::array::forcecast> frame) {
            auto buf = frame.request();
            if (buf.ndim != 2) {
                throw std::runtime_error("frame must be a 2D float32 array (H,W)");
            }

            int H = (int)buf.shape[0];
            int W = (int)buf.shape[1];

            // output mask (uint8)
            py::array_t<uint8_t> out({H, W});
            auto out_buf = out.request();

            self.apply((const float*)buf.ptr, (uint8_t*)out_buf.ptr);
            return out;
        });
}
