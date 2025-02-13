User defined Kernels
---------------------

.. admonition:: Overview
   :class: Overview

    * **Tutorial:** 30 min

        **Objectives:**
            - Learn about Elementwise Kernels.
            - Learn about Reduction Kernels.
            - Learn about Raw Kernels.


Elementwise Kernel 
~~~~~~~~~~~~~~~~~~~~~~~~~~~

CuPy's `ElementwiseKernel` allows you to define custom element-wise operations that are executed on the GPU. This is useful for operations that are not covered by CuPy's built-in functions.

**Explanation of `ElementwiseKernel`**:

1. **Input and Output Types**:
   - The first argument specifies the input types. In this case, `'float32 x, float32 y'` means that the kernel takes two input arrays of type `float32`.
   - The second argument specifies the output type. Here, `'float32 z'` means that the output array will be of type `float32`.

2. **Kernel Code**:
   - The third argument is the kernel code itself. This is a string containing the operation to be performed element-wise. In this example, `'z = (x - y)'` means that for each element, the value of `z` will be the difference between the corresponding elements of `x` and `y`.

3. **Kernel Name**:
   - The fourth argument is the name of the kernel. This is useful for debugging and profiling. In this case, the kernel is named `'element_diff'`.

**Sample Code**:

.. code-block:: python

    element_diff = cp.ElementwiseKernel('float32 x, float32 y', 
                                        'float32 z', 
                                        'z = (x - y)', 
                                        'element_diff')


Reduction Kernel 
~~~~~~~~~~~~~~~~~~~~~~~~~~~

A reduction kernel in CuPy is used to perform reduction operations on arrays. Reduction operations are those that reduce an array to a single value, such as summing all elements or finding the maximum value. The `cp.ReductionKernel` function allows you to define custom reduction operations.

Here's a breakdown of the parameters used in the example:

- `'T x'`: The input parameter type and name.
- `'T y'`: The output parameter type and name.
- `'x'`: The map expression, which is applied to each element of the input array.
- `'a + b'`: The reduction expression, which combines two elements.
- `'y = a'`: The post-reduction map expression, which is applied to the result of the reduction.
- `'0'`: The identity value for the reduction operation.
- `'reduction_kernel'`: The name of the kernel.


.. code-block:: python

    import cupy as cp

    reduction_kernel = cp.ReductionKernel(
        'T x',  # input param
        'T y',  # output param
        'x',  # map
        'a + b',  # reduce
        'y = a',  # post-reduction map
        '0',  # identity value
        'reduction_kernel'  # kernel name
    )

    # Example usage
    x = cp.array([1, 2, 3, 4, 5], dtype=cp.float32)
    result = reduction_kernel(x)
    print(result)  # Output: 15.0

Raw Kernel 
^^^^^^^^^^^^^^^

CuPy's `RawKernel` allows you to define custom CUDA kernels using CUDA C/C++ code. This is useful for operations that are not covered by CuPy's built-in functions or when you need more control over the GPU execution.

**Explanation of `RawKernel`**:

1. **RawKernel Definition**:
    ```python
    add_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void my_add(const float* x1, const float* x2, float* y) 
    {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        y[tid] = x1[tid] + x2[tid];
    }''', 'my_add')
    ```
    - `cp.RawKernel`: This is a CuPy class that allows you to define a raw CUDA kernel.
    - `r''' ... '''`: This is a raw string literal in Python, which allows you to include the CUDA C code without escaping backslashes.
    - `'my_add'`: This is the name of the kernel function defined in the CUDA C code.

2. **CUDA C Code**:
    ```c
    extern "C" __global__
    void my_add(const float* x1, const float* x2, float* y) 
    {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        y[tid] = x1[tid] + x2[tid];
    }
    ```
    - `extern "C"`: This specifies that the function should use C linkage, which is necessary for interoperability between C++ and C.
    - `__global__`: This is a CUDA keyword that indicates the function is a kernel that runs on the GPU.
    - `void my_add(const float* x1, const float* x2, float* y)`: This is the kernel function definition. It takes three pointers to float arrays as arguments.
    - `int tid = blockDim.x * blockIdx.x + threadIdx.x;`: This calculates the thread ID (`tid`) based on the block and thread indices. Each thread will have a unique `tid`.
    - `y[tid] = x1[tid] + x2[tid];`: This performs the element-wise addition for the element corresponding to the thread ID.

**Usage Example**:

.. code-block:: python
    :linenos:

    import cupy as cp

    # Define the kernel
    add_kernel = cp.RawKernel(r'''
        extern "C" __global__
        void my_add(const float* x1, const float* x2, float* y) 
        {
            int tid = blockDim.x * blockIdx.x + threadIdx.x;
            y[tid] = x1[tid] + x2[tid];
        }''', 'my_add')

    # Create input arrays
    x1 = cp.array([1, 2, 3, 4, 5], dtype=cp.float32)
    x2 = cp.array([10, 20, 30, 40, 50], dtype=cp.float32)
    y = cp.empty_like(x1)

    # Launch the kernel
    threads_per_block = 32
    blocks_per_grid = (x1.size + threads_per_block - 1) // threads_per_block
    add_kernel((blocks_per_grid,), (threads_per_block,), (x1, x2, y))

    # Print the result
    print(y)  # Output: [11. 22. 33. 44. 55.]


.. admonition:: Key Points
   :class: hint

    - CuPy's `ElementwiseKernel` allows custom element-wise GPU operations.
    - `ReductionKernel` is used for custom reduction operations on arrays.
    - `RawKernel` enables defining custom CUDA kernels using CUDA C/C++.
    - Elementwise and reduction kernels simplify GPU programming for specific tasks.

