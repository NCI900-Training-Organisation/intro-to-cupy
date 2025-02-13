
CUDA Streams
-------------

.. admonition:: Overview
   :class: Overview

    * **Tutorial:** 20 min

        **Objectives:**
            - Learn how to use CUDA streams in CuPy.

.. admonition:: Key Points
   :class: hint

    - CUDA events are used to measure the time between different points in your CUDA code.


CUDA streams are sequences of operations that are executed on the GPU in the order they are issued. Streams allow for concurrent execution of operations, which can improve performance by overlapping computation and data transfers.

Here's an example of how to use CUDA streams in CuPy:

.. code-block:: python
    :linenos:

    import numpy as np
    import cupy as cp

    a_np = np.arange(10)
    s = cp.cuda.Stream()

    with s:
        a_cp = cp.asarray(a_np)  # H2D transfer on stream s
        b_cp = cp.sum(a_cp)      # kernel launched on stream s 

    # or we can use 'use()'
    # if we use 'use()' any subsequent CUDA operation will be completed
    # using the stream we specify, until we make a change 
    s.use()

    b_np = cp.asnumpy(b_cp)

    assert s == cp.cuda.get_current_stream()

    # go back to the default stream
    cp.cuda.Stream.null.use()

    assert s == cp.cuda.get_current_stream()  # run fails if assert condition is false
                                              # generates an error

**Explanation**:

1. **Create Arrays**:
    ```python
    a_np = np.arange(10)
    s = cp.cuda.Stream()
    ```
    A NumPy array `a_np` is created, and a CUDA stream `s` is initialized.

2. **Using the Stream**:
    ```python
    with s:
        a_cp = cp.asarray(a_np)  # H2D transfer on stream s
        b_cp = cp.sum(a_cp)      # kernel launched on stream s 
    ```
    Within the context of the stream `s`, the NumPy array `a_np` is transferred to the GPU as `a_cp`, and a sum operation is performed on `a_cp`.

3. **Using `use()` Method**:
    ```python
    s.use()
    b_np = cp.asnumpy(b_cp)
    ```
    The `use()` method sets the stream `s` as the current stream. Any subsequent CUDA operations will use this stream until changed. The result `b_cp` is transferred back to the host as `b_np`.

4. **Assertions**:
    ```python
    assert s == cp.cuda.get_current_stream()
    ```
    This assertion checks that the current stream is `s`.

5. **Reverting to Default Stream**:
    ```python
    cp.cuda.Stream.null.use()
    assert s == cp.cuda.get_current_stream()  # run fails if assert condition is false
                                              # generates an error
    ```
    The default stream is set as the current stream, and an assertion checks that the current stream is no longer `s`.

This example demonstrates how to use CUDA streams in CuPy to manage concurrent execution of operations on the GPU.

.. admonition:: Key Points
   :class: hint

    - CUDA streams allow for concurrent execution of operations on the GPU.
    - Streams can be used to overlap computation and data transfers for improved performance.
    - The `use()` method sets the current stream for subsequent CUDA operations.