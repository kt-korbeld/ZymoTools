"""
Run a foundry CLI (``rfd3`` / ``rf3`` / ``mpnn``) with the residue-embedding
cache disabled.

Why this exists
---------------
The published RFdiffusion3 and RF3 checkpoints carry a training config whose
``paths.data.residue_cache_dir`` points at an IPD-internal directory under
``/net/tukwila/...``. Nobody outside the IPD has that directory, and upstream
that is harmless: ``Path.exists()`` returns False, the loader falls back to an
empty cache, and the run continues with one warning per residue type. That is
what RosettaCommons/foundry#180 reports -- noisy, but only a warning.

On Habrok's GPU nodes it is not only a warning. ``/net`` is an automount that
answers ``stat()`` on an unknown key with EPERM instead of ENOENT, and pathlib
only swallows a fixed set of errnos (ENOENT, ENOTDIR, EBADF, ELOOP) -- EPERM is
not among them. So ``Path.exists()`` *raises*, and ``rfd3 design`` dies while
building its transform pipeline, before it ever reaches the GPU::

    PermissionError: [Errno 1] Operation not permitted:
        '/net/tukwila/ncorley/datahub/MACE-OFF23_medium/global_stats.pt'

The same node answers ENOENT for the same path from a login node, which is why
this only shows up inside a job.

What the patch does
-------------------
``LoadCachedResidueLevelData`` is the one transform that stats those paths, and
it takes the directory as a constructor argument. We force that argument to
``None``, which makes its loader short-circuit to an empty cache. This is not a
behaviour change invented here: it is the value rfd3's own pipeline builder
defaults to for checkpoints that carry no cache path
(``rfd3/transforms/pipelines.py``), and the end state matches every upstream
user whose ``/net`` path simply does not resolve.

Set ``FOUNDRY_RESIDUE_CACHE_DIR`` to a real cache directory to use one instead.

This module imports nothing but the standard library, so it can be run directly by
foundry's own interpreter without this package being installed there::

    <foundry-venv>/bin/python .../binder_pipeline/foundry_shim.py rfd3 design inputs=...
"""

import importlib
import os
import sys

# console_scripts declared by rc-foundry, from its entry_points.txt
ENTRY_POINTS = {
    "rfd3": ("rfd3.cli", "app"),
    "rfd3na": ("rfd3na.cli", "app"),
    "rf3": ("rf3.cli", "app"),
    "mpnn": ("mpnn.inference", "main"),}

CACHE_ENV = "FOUNDRY_RESIDUE_CACHE_DIR"


def residue_cache_dir():
    """
    The cache directory to force, or None to disable the cache entirely.
    """
    value = os.environ.get(CACHE_ENV, "").strip()
    return value or None


def patch_residue_cache(cache_dir=None, verbose=True):
    """
    Make every ``LoadCachedResidueLevelData`` use ``cache_dir`` instead of the
    path baked into the checkpoint.

    foundry declares its inference APIs unstable, so a layout change here is
    reported and tolerated rather than raised: the run then fails exactly the way
    it would have without the shim, which is a far clearer signal than an
    ImportError from a wrapper script.
    Returns True if the patch was applied.
    """
    try:
        from atomworks.ml.transforms.cached_residue_data import LoadCachedResidueLevelData
    except Exception as exc:
        print("[shim] could not import the residue cache transform ({}); "
              "running unpatched".format(exc), flush=True)
        return False
    # idempotent: the shim may be re-entered within one interpreter
    if getattr(LoadCachedResidueLevelData, "_binder_pipeline_patched", False):
        return True
    original = LoadCachedResidueLevelData.__init__

    def __init__(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.dir = cache_dir

    LoadCachedResidueLevelData.__init__ = __init__
    LoadCachedResidueLevelData._binder_pipeline_patched = True
    if verbose:
        print("[shim] residue cache -> {}".format(
            cache_dir if cache_dir else "disabled"), flush=True)
    return True


def ensure_cuequivariance_usable(verbose=True):
    """
    Disable foundry's fused cuEquivariance kernels when they cannot actually run.

    foundry decides at import time whether to use them, and its test is
    ``import cuequivariance_torch``. That wrapper imports fine even when the
    compiled ops behind it do not -- as happens when torch has been reinstalled at
    a CUDA version ``cuequivariance-ops-torch`` was not built for, leaving
    ``libcue_ops.so`` unable to find ``libnvrtc.so.12``. The failure then surfaces
    as an ImportError deep inside RF3's pairformer forward pass, minutes into a
    GPU job.

    So probe the compiled ops directly, and if they are unusable set foundry's own
    ``DISABLE_CUEQUIVARIANCE`` switch, which routes the triangle multiplication
    through its vanilla PyTorch implementation instead. Slower, but it runs.
    Machines where the kernels do work are left alone, and an explicit setting by
    the caller is never overridden.

    Must be called *before* foundry is imported, since the flag is read once at
    import time.
    """
    if os.environ.get("DISABLE_CUEQUIVARIANCE"):
        return False
    try:
        ops = importlib.import_module("cuequivariance_ops_torch")
        if not hasattr(ops, "triangle_multiplicative_update"):
            raise ImportError("cuequivariance_ops_torch exposes no "
                              "triangle_multiplicative_update")
    except Exception as exc:
        os.environ["DISABLE_CUEQUIVARIANCE"] = "True"
        if verbose:
            print("[shim] cuEquivariance kernels unusable ({}); falling back to "
                  "vanilla attention".format(exc), flush=True)
        return True
    return False


def main(argv=None):
    """
    Dispatch to one of foundry's console scripts with the patch in place.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in ENTRY_POINTS:
        print("usage: {} {{{}}} [args...]".format(
            os.path.basename(__file__), "|".join(sorted(ENTRY_POINTS))),
            file=sys.stderr)
        return 2
    tool, rest = argv[0], argv[1:]
    # Run by file path, sys.path[0] is this package's directory, where a module
    # sharing a name with one of foundry's (util, config, ...) would shadow it.
    # Nothing here is imported from the package, so drop it before importing.
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path[:] = [p for p in sys.path if os.path.abspath(p) != here]
    module_name, attr = ENTRY_POINTS[tool]
    # before any foundry import: the flag it reads is evaluated at import time
    ensure_cuequivariance_usable()
    try:
        entry = getattr(importlib.import_module(module_name), attr)
    except Exception as exc:
        # foundry is not importable from this interpreter. Hand off to whatever
        # ``tool`` is on PATH so the shim degrades to a pass-through instead of
        # becoming a second thing that can fail; note that if foundry really is
        # broken, the exec below will fail too, which is the honest outcome.
        print("[shim] {} is not importable ({}); exec'ing {} from PATH "
              "without the residue-cache patch".format(module_name, exc, tool),
              flush=True)
        os.execvp(tool, [tool] + rest)
    patch_residue_cache(residue_cache_dir())
    # hydra (rfd3, rf3) and argparse (mpnn) both read sys.argv, so present the
    # command line as though the console script itself had been invoked
    sys.argv = [tool] + rest
    return entry()


if __name__ == "__main__":
    sys.exit(main())
