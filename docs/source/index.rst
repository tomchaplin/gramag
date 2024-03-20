.. gramag documentation master file, created by
   sphinx-quickstart on Mon Jun 19 15:02:05 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

gramag - Python API Docs
=====================================

.. automodule:: gramag
   :members:
   :undoc-members:

   .. literalinclude:: examples/simple.py
      :caption: A simple example
      :language: python

.. doctest::

   >>> from gramag import MagGraph, format_table
   >>> mg = MagGraph([(0, 1), (0, 2), (1, 4), (2, 3), (3, 4)])
   >>> mg.populate_paths(l_max=3)
   >>> print(format_table(mg.rank_homology()))
   ╭─────┬─────────╮
   │ k=  │ 0  1  2 │
   ├─────┼─────────┤
   │ l=0 │ 5  0  0 │
   │ l=1 │ 0  5  0 │
   │ l=2 │ 0  0  0 │
   │ l=3 │ 0  0  1 │
   ╰─────┴─────────╯
