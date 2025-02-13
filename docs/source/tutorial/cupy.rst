CuPy Function Overview: Key Operations and Utilities
----------------------------------------------------

.. admonition:: Overview
   :class: Overview

    * **Tutorial:** 60 min

        **Objectives:**
            - Learn the key operations and utilities provided by CuPy.
            - Understand how to work with GPU devices, memory management, and data movement.


CuPy is a GPU-accelerated array library for Python, designed as a drop-in replacement for NumPy. It enables fast numerical computations by leveraging
NVIDIA CUDA, allowing seamless execution of array operations on the GPU. CuPy provides a NumPy-compatible API, making it easy to port existing 
NumPy code to run on GPUs with minimal modifications. 

CuPy offers several advantages, especially for GPU-accelerated computing in Python:

1. **NumPy Compatibility** - CuPy provides a NumPy-like API, making it easy to convert existing NumPy code to run on the GPU with minimal changes.
2. **High Performance** - It leverages CUDA for parallel computation, significantly accelerating array operations compared to CPU-based NumPy.
3. **Memory Management** - CuPy efficiently manages GPU memory, reducing unnecessary data transfers and providing features like memory pools to minimize allocation overhead.
4. **Support for Advanced Features** - It includes support for sparse matrices, Fast Fourier Transforms (FFTs), random number generation, and linear algebra, optimized for GPU execution.
5. **Interoperability** - CuPy integrates well with deep learning frameworks like TensorFlow and PyTorch and supports interoperability with libraries like cuBLAS, cuDNN, and NCCL.
6. **Custom Kernel Support** - Users can define and execute custom CUDA kernels directly within Python, enabling fine-grained GPU optimization for specific applications.
7. **Multi-GPU and Distributed Computing** - CuPy supports multi-GPU processing and can be integrated into distributed computing frameworks for large-scale computations.


Despite its advantages, CuPy has some limitations:

1. **NVIDIA CUDA Dependency** - CuPy requires an NVIDIA GPU with CUDA support, making it unusable on systems with AMD GPUs or without a GPU.
2. **Limited CPU Fallback** - Unlike NumPy, which works universally on CPUs, CuPy does not automatically fall back to CPU execution if a GPU is unavailable. Manual handling is required.
3. **Memory Constraints** - GPU memory is limited compared to system RAM, which can lead to out-of-memory errors for large datasets if not managed carefully.
4. **Data Transfer Overhead** - Moving data between CPU (host) and GPU (device) can be slow, affecting performance when frequent transfers are needed. Efficient memory management is crucial.
5. **Incomplete NumPy Feature Parity** - While CuPy supports most NumPy functions, some rarely used or complex operations may not be available or may require workarounds.
6. **Debugging Complexity** - Debugging CUDA-based applications is more challenging compared to CPU-based NumPy code, requiring specialized tools like cuda-gdb or Nsight Systems.
7. **Limited Multi-GPU Support** - While CuPy supports multiple GPUs, its built-in multi-GPU handling is not as advanced as frameworks like PyTorch or TensorFlow, requiring manual coordination for complex workloads.
8. **Installation Complexity** - Installing CuPy with the correct CUDA and cuDNN versions can be tricky, especially in environments with multiple CUDA versions.

Device Count
~~~~~~~~~~~~

CuPy can be used to query the number of available GPU devices.

.. code-block:: python
    :linenos:

    import cupy as cp

    # Get the number of available GPU devices
    device_count = cp.cuda.runtime.getDeviceCount()
    print(f"Number of available GPU devices: {device_count}")

  
   

This function is important because it allows the application to determine the number of devices
that are currently available for use. Knowing the device count is crucial for resource allocation,
load balancing, and ensuring that the application can scale appropriately based on the number of
devices. It also helps in debugging and monitoring the system to ensure that all expected devices
are detected and functioning correctly.

Device Properties
~~~~~~~~~~~~~~~~~

CuPy can be used to query the properties of available GPU devices.

.. code-block:: python
    :linenos:

    import cupy as cp

    # Get the properties of the first GPU device
    device_properties = cp.cuda.runtime.getDeviceProperties(0)
    print(f"Device properties: {device_properties}")

This function is useful because it provides detailed information about the GPU device, such as its name, total memory, compute capability, and 
other hardware specifications. This information is crucial for optimizing performance, debugging, and ensuring compatibility with specific GPU features. 
By understanding the properties of the GPU, developers can make informed decisions about resource allocation and performance tuning.
  
Current Device
~~~~~~~~~~~~~~

CuPy allows you to query and set the current GPU device. This is useful when working with multiple GPUs, as it enables you to control which device is 
used for computations.

To get the current device, you can use the `cp.cuda.runtime.getDevice()` function.

.. code-block:: python
    :linenos:

    import cupy as cp

    # Get the current GPU device
    current_device = cp.cuda.runtime.getDevice()
    print(f"Current GPU device: {current_device}")

This function returns the ID of the currently active GPU device. Knowing the current device is important for managing resources and ensuring that 
computations are performed on the intended GPU.

To set the current device, you can use the `cp.cuda.Device` context manager.

.. code-block:: python
    :linenos:

    import cupy as cp

    # Set the current GPU device to device 1
    with cp.cuda.Device(1):
        # Perform operations on device 1
        a = cp.array([1, 2, 3])
        print(f"Array on device 1: {a}")

    # Back to the original device
    current_device = cp.cuda.runtime.getDevice()
    print(f"Current GPU device: {current_device}")

Using the `cp.cuda.Device` context manager ensures that the specified device is used for all operations within the context. This is particularly 
useful when you need to perform computations on different devices in a controlled manner.

CuPy ndarray
~~~~~~~~~~~~

The `ndarray` in CuPy is a core data structure that represents a multidimensional, homogeneous array of fixed-size items. It is similar 
to the `ndarray` in NumPy but is designed to leverage GPU acceleration for high-performance computations.

Comparison with NumPy ndarray
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. **Memory Location**:
    - **NumPy**: The `ndarray` in NumPy resides in the system's main memory (RAM).
    - **CuPy**: The `ndarray` in CuPy resides in the GPU memory, allowing for faster computations by utilizing the parallel processing capabilities of the GPU.

2. **Performance**:
    - **NumPy**: Operations on NumPy arrays are performed on the CPU, which may be slower for large-scale computations.
    - **CuPy**: Operations on CuPy arrays are performed on the GPU, which can significantly speed up computations, especially for large datasets.

3. **Interoperability**:
    - **NumPy**: NumPy arrays are not directly compatible with GPU operations.
    - **CuPy**: CuPy arrays can be easily converted to and from NumPy arrays, allowing for seamless integration between CPU and GPU computations.

Allocating ndarray on a Specific GPU
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

CuPy allows you to allocate an `ndarray` on a specific GPU device. This is useful when working with multiple GPUs to distribute the workload.

To allocate an `ndarray` on a particular GPU, you can use the `cp.cuda.Device` context manager.

.. code-block:: python
     :linenos:

     import cupy as cp

     # Set the current GPU device to device 1
     with cp.cuda.Device(1):
          # Allocate an ndarray on device 1
          a = cp.array([1, 2, 3])
          print(f"Array on device 1: {a}")

     # Back to the original device
     current_device = cp.cuda.runtime.getDevice()
     print(f"Current GPU device: {current_device}")

By using the `cp.cuda.Device` context manager, you can ensure that the `ndarray` is allocated on the specified GPU device. This is particularly useful 
for managing resources and optimizing performance in multi-GPU environments.

Finding the GPU where the ndarray is located
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

CuPy provides a way to determine which GPU device an `ndarray` is located on. This is useful for managing resources and optimizing performance, 
especially in multi-GPU environments.

To find the GPU device where an `ndarray` is located, you can use the `ndarray.device` attribute.

.. code-block:: python
    :linenos:

    import cupy as cp

    # Allocate an ndarray on device 1
    with cp.cuda.Device(1):
        a = cp.array([1, 2, 3])

    # Find the device where the ndarray is located
    device = a.device
    print(f"ndarray is located on device: {device}")

This functionality is important for several reasons:

1. **Resource Management**: Knowing the device where an `ndarray` is located helps in managing GPU resources effectively. It allows you to track memory usage and ensure that computations are performed on the intended device.

2. **Performance Optimization**: By understanding the device allocation of `ndarray` objects, you can optimize performance by minimizing data transfers between devices and ensuring that computations are performed on the most suitable GPU.

3. **Debugging**: When working with multiple GPUs, it is crucial to know the device allocation of `ndarray` objects to debug issues related to device-specific computations and memory management.

By leveraging the `ndarray.device` attribute, developers can gain better control over their multi-GPU applications and optimize their code for improved 
performance and resource utilization.


Data Movement in CuPy
~~~~~~~~~~~~~~~~~~~~~

CuPy allows you to move data between the host (CPU) and different GPU devices. This is useful for optimizing performance and managing resources in 
multi-GPU environments.

1. **Create a NumPy Array**: First, create a NumPy array on the host (CPU).

.. code-block:: python
    :linenos:

    import numpy as np

    # Create a NumPy array on the host
    data_cpu = np.array([1, 2, 3, 4, 5])
    print(f"Data on CPU: {data_cpu}")

2. **Move the NumPy Array to GPU 1**: Use CuPy to move the NumPy array from the host to GPU 1.

.. code-block:: python
    :linenos:

    import cupy as cp

    # Move the NumPy array to GPU 1
    with cp.cuda.Device(1):
        data_gpu_1 = cp.asarray(data_cpu)
        print(f"Data on GPU 1: {data_gpu_1}")

3. **Move the Data from GPU 1 to GPU 0**: Transfer the data from GPU 1 to GPU 0.

.. code-block:: python
    :linenos:

    # Move the data from GPU 1 to GPU 0
    with cp.cuda.Device(0):
        data_gpu_0 = cp.asarray(data_gpu_1)
        print(f"Data on GPU 0: {data_gpu_0}")

By following these steps, you can efficiently manage and transfer data between the host and multiple GPU devices using CuPy. This is particularly 
useful for optimizing performance and resource utilization in multi-GPU applications.


Transferring Data from GPU to Host
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When working with CuPy, you often need to transfer data from the GPU back to the host (CPU). CuPy provides two main methods for 
this purpose: ``cupy.ndarray.get()`` and ``cupy.asnumpy()``. Here's an explanation of each method and their differences:

1. **``cupy.ndarray.get()``**:
   - This method is called on a CuPy array object.
   - It returns a NumPy array containing the same data as the CuPy array.
   - Example:

.. code-block:: python
    :linenos:

    import cupy as cp

    # Create a CuPy array
    data_gpu = cp.array([1, 2, 3, 4, 5])

    # Transfer data from GPU to host
    data_cpu = data_gpu.get()
    print(data_cpu)  # Output: [1 2 3 4 5]

2. **``cupy.asnumpy()``**:
   - This is a standalone function that takes a CuPy array as an argument.
   - It returns a NumPy array containing the same data as the input CuPy array.
   - Example:

.. code-block:: python
    :linenos:

    import cupy as cp

    # Create a CuPy array
    data_gpu = cp.array([1, 2, 3, 4, 5])

    # Transfer data from GPU to host
    data_cpu = cp.asnumpy(data_gpu)
    print(data_cpu)  # Output: [1 2 3 4 5]

Differences:

- **Method of Invocation**:
  - ``cupy.ndarray.get()`` is called on a CuPy array object.
  - ``cupy.asnumpy()`` is a standalone function that takes a CuPy array as an argument.
- **Usage Context**:
  - Use ``cupy.ndarray.get()`` when you prefer method chaining or object-oriented style.
  - Use ``cupy.asnumpy()`` when you prefer a functional approach or need to convert multiple arrays in a consistent manner.

By understanding these methods, you can efficiently transfer data from the GPU to the host, enabling further processing or analysis 
using NumPy functions.

Building Device Agnostic Code
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When developing applications with CuPy, it is often useful to write code that can run seamlessly on both CPU and GPU. This is known as device agnostic 
code. CuPy provides utilities to facilitate this, allowing you to write functions that can operate on either NumPy arrays (CPU) or CuPy arrays (GPU) 
without modification.

To demonstrate this, consider the following example function that computes the logarithm of an array in a device agnostic manner:

.. code-block:: python
    :linenos:

    import cupy as cp
    import numpy as np

    def log_array(x):
        xp = cp.get_array_module(x)  # Returns cupy if x is a CuPy array, numpy if x is a NumPy array
        return xp.log1p(xp.exp(-abs(x)))

In this example, the `log_array` function uses `cp.get_array_module(x)` to determine whether the input array `x` is a CuPy array or a NumPy array. 
The function then uses the appropriate module (`cupy` or `numpy`) to perform the computation. This allows the same function to work 
with both CPU and GPU arrays.

By writing device agnostic code, you can ensure that your applications are flexible and can take advantage of GPU acceleration when available, 
while still being able to run on systems without a GPU.

Explicit data transferDifferences
~~~~~~~~~~~~~~~~~~~~~~

In CuPy, explicit data transfers between the host (CPU) and the device (GPU) can be performed using functions like ``cupy.asnumpy()`` and ``cupy.asarray()``.

- ``cupy.asnumpy(array)``: This function transfers a CuPy array from the GPU to a NumPy array on the CPU. It is useful when you need to perform operations that are not supported by CuPy or when you need to interface with libraries that only accept NumPy arrays.

  Example:

.. code-block:: python
    :linenos:

    import cupy as cp
    gpu_array = cp.array([1, 2, 3])
    cpu_array = cp.asnumpy(gpu_array)
    print(type(cpu_array))  # <class 'numpy.ndarray'>

- ``cupy.asarray(array)``: This function transfers a NumPy array from the CPU to a CuPy array on the GPU. It is useful when you want to leverage GPU acceleration for computations on data that initially resides on the CPU.


Example:

.. code-block:: python
    :linenos:

    import numpy as np
    import cupy as cp
    cpu_array = np.array([1, 2, 3])
    gpu_array = cp.asarray(cpu_array)
    print(type(gpu_array))  # <class 'cupy._core.core.ndarray'>

The difference between explicit and automatic data transfer:
- **Explicit data transfer**: As shown above, explicit data transfer requires the user to manually transfer data between the CPU and GPU using functions like ``cupy.asnumpy()`` and ``cupy.asarray()``. This gives the user full control over when and how data is transferred, which can be important for optimizing performance.


Automatic Data Transfer
~~~~~~~~~~~~~~~~~~~~~~~

CuPy can automatically transfer data between the host (CPU) and the device (GPU) when performing operations between CuPy and NumPy arrays. This can 
simplify code but may introduce performance overhead due to implicit data transfers.

Example:

.. code-block:: python
    :linenos:

    import numpy as np
    import cupy as cp

    # Create a NumPy array on the host
    data_cpu = np.array([1, 2, 3, 4, 5])

    # Create a CuPy array on the GPU
    data_gpu = cp.array([10, 20, 30, 40, 50])

    # Perform an operation between NumPy and CuPy arrays
    result = data_cpu + data_gpu

    # The result is a CuPy array
    print(result)  # Output: [11 22 33 44 55]
    print(type(result))  # <class 'cupy._core.core.ndarray'>

In this example, the addition operation between a NumPy array (`data_cpu`) and a CuPy array (`data_gpu`) triggers an automatic data transfer. The NumPy 
array is transferred to the GPU, and the result is a CuPy array.

By understanding and utilizing explicit data transfers, you can optimize your applications to efficiently use both CPU and GPU resources.


.. admonition:: Key Points
   :class: hint

        - CuPy is an open-source array library that leverages NVIDIA CUDA for GPU acceleration, designed to be a drop-in replacement for NumPy.
        - CuPy provides functions to query the number of available GPU devices, their properties, and to set or get the current device.
        - The `ndarray` in CuPy is similar to NumPy's but resides in GPU memory, offering significant performance improvements for large-scale computations.
        - CuPy allows efficient data transfer between the host (CPU) and GPU devices, supporting both explicit and automatic data transfers.
        - CuPy enables writing device agnostic code that can run on both CPU and GPU, facilitating seamless integration and performance optimization.