Introduction to CuPy
=====================

This repository provides a comprehensive guide to understanding and utilizing CuPy, a GPU-accelerated library for Python that is 
designed to scale large-scale numerical computations and data analysis workflows.

.. note::

   This project is under active development.

Contents
--------

.. list-table:: Topic Covered
   :widths: 20 20 
   :header-rows: 1   
 
   * - Topics
     - Duration
   * - ARE Setup
     - 30 Minutes
   * - GPU architecture
     - 30 Minutes
   * - CuPy Basics
     - 60 Minutes
   * - User Kernels
     - 30 Minutes
   * - CUDA Events
     - 15 Minutes
   * - CUDA Stream
     - 15 Minutes
  

.. toctree::
    :maxdepth: 1
    :caption: Setup and Prerequisites

    prerequisite.rst
    outcomes.rst  
    modules.rst
    packages.rst

.. toctree::
    :maxdepth: 4
    :caption: Tutorial
    :numbered:
    
    tutorial/gpu
    tutorial/cupy
    tutorial/kernel.rst
    tutorial/events.rst
    tutorial/streams.rst
    references.rst



