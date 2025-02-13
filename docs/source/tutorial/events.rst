CUDA Events
------------------------

.. admonition:: Overview
   :class: Overview

    * **Tutorial:** 20 min

        **Objectives:**
            - Learn how to use cuda events in CuPy.



CUDA events are used to measure the time between different points in your CUDA code, synchronize streams, and manage dependencies between operations. Here's a step-by-step explanation of the provided sample program:

**Sample Program**:

.. code-block:: python
    :linenos:

    import cupy as cp

    a_cp = cp.arange(10)
    b_cp = cp.arange(10)
    e1 = cp.cuda.Event() # create an event
    e1.record() # Records an event on the stream.
    a_cp = b_cp * a_cp + 8
    e2 = cp.cuda.get_current_stream().record() # create and record the event
    s = cp.cuda.Stream.null
    # make sure the stream wait for the second event
    s.wait_event(e2)

    with s:
        # all actions in a stream happens sequentially
        # as the stream is waiting for the second event to complete
        # we can be assured that all the operations before it also has been complete.
        a_np = cp.asnumpy(a_cp)

    # Waits for the stream that track an event to complete that event
    e2.synchronize()
    t = cp.cuda.get_elapsed_time(e1, e2)

    print(t)

**Explanation**:

1. **Create Arrays**:
    ```python
    a_cp = cp.arange(10)
    b_cp = cp.arange(10)
    ```
    Two arrays `a_cp` and `b_cp` are created using CuPy.

2. **Create and Record First Event**:
    ```python
    e1 = cp.cuda.Event() # create an event
    e1.record() # Records an event on the stream.
    ```
    An event `e1` is created and recorded. This marks a point in the default stream.

3. **Perform Operations**:
    ```python
    a_cp = b_cp * a_cp + 8
    ```
    An element-wise operation is performed on the arrays.

4. **Create and Record Second Event**:
    ```python
    e2 = cp.cuda.get_current_stream().record() # create and record the event
    ```
    Another event `e2` is created and recorded on the current stream.

5. **Wait for Event in Null Stream**:
    ```python
    s = cp.cuda.Stream.null
    s.wait_event(e2)
    ```
    The null stream `s` is instructed to wait for the event `e2` to complete.

6. **Operations in Stream**:
    ```python
    with s:
        a_np = cp.asnumpy(a_cp)
    ```
    Within the context of the null stream, the array `a_cp` is converted to a NumPy array `a_np`. Since the stream waits for `e2`, all previous operations are guaranteed to be complete.

7. **Synchronize and Measure Time**:
    ```python
    e2.synchronize()
    t = cp.cuda.get_elapsed_time(e1, e2)
    print(t)
    ```
    The code waits for the event `e2` to complete and then measures the elapsed time between `e1` and `e2`.

This program demonstrates how to use CUDA events to measure the time taken for GPU operations and ensure proper synchronization between different streams.

.. admonition:: Key Points
:class: hint

 - CUDA events are used to measure the time between different points in your CUDA code.
 - CUDA events help synchronize streams and manage dependencies between operations.
 - The null stream can wait for events to ensure all previous operations are complete.
 - CuPy provides functions to create, record, and synchronize CUDA events.
 - You can measure the elapsed time between two events using `cp.cuda.get_elapsed_time`.