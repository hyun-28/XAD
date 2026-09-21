"""Run-log header required by BRIEF §2.2: every script prints this first."""
import platform
import socket
import sys
from importlib.metadata import PackageNotFoundError, version


def _ver(pkg: str) -> str:
    try:
        return version(pkg)
    except PackageNotFoundError:
        return "not-installed"


def env_header() -> str:
    # TSB-AD is installed from the PyPI sdist, so there is no git commit hash
    # to record; the released version string is the reproducibility key
    # (DECISIONS.md D-A0-2).
    return (
        f"python={sys.version.split()[0]} numpy={_ver('numpy')} "
        f"sklearn={_ver('scikit-learn')} TSB-AD={_ver('TSB-AD')}(PyPI) "
        f"OS={platform.system()}-{platform.release()}-{platform.machine()} "
        f"hostname={socket.gethostname()}"
    )
