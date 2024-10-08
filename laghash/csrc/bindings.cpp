/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2023, Towaki Takikawa.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#include <torch/extension.h>
#include "./ops/laghash_interpolate.h"

namespace laghash {

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::module ops = m.def_submodule("ops");
    ops.def("laghash_interpolate_cuda", &laghash_interpolate_cuda);
    ops.def("laghash_interpolate_backward_cuda", &laghash_interpolate_backward_cuda);

    // ops.def("hash_interpolate_cuda", &hash_interpolate_cuda);
    // ops.def("hash_interpolate_backward_cuda", &hash_interpolate_backward_cuda);
}

}
