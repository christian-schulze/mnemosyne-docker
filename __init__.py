"""
Mnemosyne Plugin for Hermes Agent
Entry point at repo root for `hermes plugins install` compatibility.
"""

import sys
from pathlib import Path

# Ensure this directory is on path so `hermes_plugin` is discoverable
_repo_root = Path(__file__).resolve().parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# Ensure the hermes_memory_provider subdirectory is importable
_hmp_dir = str(_repo_root / "hermes_memory_provider")
if _hmp_dir not in sys.path:
    sys.path.insert(0, _hmp_dir)

# Re-export __version__ / __author__ from the inner mnemosyne subpackage so
# `from mnemosyne import __version__` works in either install layout:
#   - Hermes plugin tree: outer `mnemosyne/` is the resolved package, inner
#     `mnemosyne/mnemosyne/` is the subpackage `mnemosyne.mnemosyne`.
#   - pip / repo-direct install: inner `mnemosyne/` is the resolved package
#     directly and this stub is never loaded.
# Without this re-export, `hermes mnemosyne version` (and any other caller
# doing `from mnemosyne import __version__`) crashed with ImportError under
# the Hermes plugin layout. See issue #53.
try:
    from .mnemosyne import __version__, __author__
except ImportError:
    __version__ = "unknown"
    __author__ = "Abdias J"

# Expose the memory provider for Hermes memory provider discovery
# (plugins/memory/__init__.py scans for "register_memory_provider" or
# "MemoryProvider" in the source text, then loads and calls register().)
try:
    from hermes_memory_provider import MnemosyneMemoryProvider as _MnemosyneMemoryProvider

    def register_memory_provider(ctx):
        """Called by Hermes memory provider discovery system."""
        ctx.register_memory_provider(_MnemosyneMemoryProvider())

    # Also expose MemoryProvider reference for _is_memory_provider_dir heuristic
    MemoryProvider = _MnemosyneMemoryProvider
except ImportError as _import_err:
    _import_err  # silence unused-variable lint in graceful fallback

# Graceful fallback when Hermes framework is not present
# (e.g. pip-only / standalone installs without hermes_plugin)
try:
    from hermes_plugin import register
    __all__ = ["register", "register_memory_provider", "__version__", "__author__"]
except ImportError:
    __all__ = ["register_memory_provider", "__version__", "__author__"]
