<div align="center">

<h1>gramag</h1>

<b>Gra</b>ph <b>Mag</b>nitude Homology in Rust, with Python bindings

[![PyPi](https://img.shields.io/pypi/v/gramag)](https://pypi.org/project/gramag/)
[![Read The Docs](https://readthedocs.org/projects/gramag/badge/?version=latest)](https://gramag.readthedocs.io/en/latest/?badge=latest)

</div>

## Overview

`gramag` is a library for computing the magnitude homology of finite (directed) graphs in Python and Rust.
The library is capable of computing homology ranks and representatives, over ℤ₂.
For background on graph magnitude homology, see the original paper by Hepworth and Willerton [[1]](#1).

## Computational detail

In an attempt to compute magnitude homology for large graphs, we attempt to parallelise computation wherever possible; this may result in increased memory usage.
In particular, the initial basis for each of the magnitude chain groups is computed via a parallelised breadth-first search.
To limit the number of threads used, simply set the environment variable `RAYON_NUM_THREADS` appropriately.

Throughout the codebase we make extensive use of the fact that the magnitude chain complex splits over node pairs, i.e.
$$MC_{\bullet, \bullet} = \bigoplus_{(s, t)\in V\times V} MC_{\bullet, \bullet}^{(s, t)}$$
where $MC_{\bullet, \bullet}^{(s, t)}$ is the sub-complex of $MC_{\bullet, \bullet}$ generated by those paths starting at $s$ and ending at $t$.
All of the Python APIs admit a `node_pairs` argument to restrict the set of node pairs $(s, t)$ over which this direct sum is computed.
Unfortunately, the initial path search does not admit such an argument at the moment.

## Python

The easiest way to use `gramag` is to install the Python bindings.
Pre-compiled packages are available for most systems on [PyPi](https://pypi.org/project/gramag/), failing which the source distribution can be installed in the presence of a suitable `cargo` toolchain.
On most modern systems, `gramag` can be installed through `pip` via:

```bash
pip install gramag
```

### Usage

Full documentation is available on [Read The Docs](https://gramag.readthedocs.io) or can be built from source by calling
```bash
just setup_venv
just py_docs_open
```
A simple example script is provided in [`simple.py`](docs/source/examples/simple.py).
For more detailed usage, please refer to [`advanced.py`](docs/source/examples/advanced.py).

## Rust

The Rust library has not yet been finalised.

## References

<a id="1">[1]</a>
Hepworth, Richard, and Simon Willerton.
"Categorifying the magnitude of a graph."
arXiv preprint [arXiv:1505.04125](https://arxiv.org/abs/1505.04125) (2015).
