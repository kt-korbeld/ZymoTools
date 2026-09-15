"""
Various utility functions, mainly for managing the various environments and backends. 
"""

from pathlib import Path
import os
import shlex
import shutil

# Subdirectory holding an environment's executables. venv and conda agree on
# bin/ everywhere except Windows, where both use Scripts/.
BIN_DIRS = ("bin", "Scripts")
# Names venv gives its activation script, one per supported shell.
ACTIVATE_NAMES = ("activate", "activate.sh", "activate.csh", "activate.fish")


def env_bin(env_path):
    """
    The directory holding an environment's executables, or None if ``env_path``
    is not laid out like an environment.
    """
    if not env_path:
        return None
    env_path = Path(env_path).expanduser()
    for name in BIN_DIRS:
        bin_dir = env_path / name
        if bin_dir.is_dir():
            return bin_dir
    return None


def find_executable_in_env(env_path, command):
    """
    Path of ``command`` inside a given environment, or None if it is not there.
    Supports venv and conda layouts.
    """
    bin_dir = env_bin(env_path)
    if bin_dir is None:
        return None
    for candidate in (bin_dir / command, bin_dir / "{}.exe".format(command)):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    return None


def search_local_envs(command, max_depth=2):
    """
    Search the current directory for an environment providing ``command``.

    Directories whose name looks like an environment are tried first, then every
    other directory, then one level below each. A last resort for when nothing
    was passed and the command is not on PATH.
    """
    cwd = Path.cwd()
    # common names for an environment directory, matched as substrings so that
    # venv-foundry, .venv and miniconda3 all qualify
    common_names = ("venv", "env", "conda", "mamba")
    try:
        dirs = sorted(d for d in cwd.iterdir() if d.is_dir())
    except OSError:
        return None
    likely = [d for d in dirs if any(n in d.name.lower() for n in common_names)]
    for path in likely + [d for d in dirs if d not in likely]:
        cmd = find_executable_in_env(path, command)
        if cmd:
            return cmd
        # one level deeper if allowed: environments are often collected under a
        # single directory (environments/foundry-env), but do not recurse further
        if max_depth > 1:
            try:
                subs = sorted(s for s in path.iterdir() if s.is_dir())
            except OSError:
                continue
            for sub in subs:
                cmd = find_executable_in_env(sub, command)
                if cmd:
                    return cmd
    return None


def resolve_command(env_path, command):
    """
    Absolute path of ``command``:
    - inside the user-provided environment, if one was given
    - otherwise whatever is on PATH
    - otherwise an environment found beside the working directory

    A user-provided environment is authoritative: if the command is not in it,
    that is an error rather than a reason to fall back to some other copy.
    """
    # check the user-provided env, which may be named by an activation command
    if env_path:
        root = env_root(env_path) or Path(env_path).expanduser()
        cmd = find_executable_in_env(root, command)
        if cmd:
            return cmd
        raise ValueError("command '{}' not found in env: {}".format(command, env_path))
    # if None, check if the command is on PATH
    on_path = shutil.which(command)
    if on_path:
        return Path(on_path)
    # last resort: an environment sitting in the current directory
    cmd = search_local_envs(command)
    if cmd:
        return cmd
    raise RuntimeError("could not locate '{}' in PATH or local environments".format(command))


# --------------------------------------------------------------------------
# turning an environment flag into job-script lines
# --------------------------------------------------------------------------

def get_words(spec):
    """
    Split a flag value the way a shell would, falling back to whitespace when it
    does not tokenise (an unbalanced quote, say).
    """
    try:
        return shlex.split(str(spec).strip())
    except ValueError:
        return str(spec).strip().split()


def is_env_path(spec):
    """
    Check whether the flag is just env path: 
    must be viable path, no other flags, commands, etc
    """
    if not spec:
        return False
    spec = str(spec).strip()
    # check if provided str is a directory
    if Path(spec).expanduser().is_dir():
        return True
    # if it can be
    words = get_words(spec)
    if len(words) != 1:
        return False
    path = Path(words[0]).expanduser()
    # check if path is or could be a dir or activate script
    isdir = path.is_dir()
    couldbedir = words[0].startswith(("/", "./", "../", "~"))
    activate = path.name in ACTIVATE_NAMES
    return (isdir or couldbedir or activate)


def env_root(spec):
    """
    The environment directory named by an environment flag, or None.

    Accepts the directory itself, the path of its activate script, or a command
    that sources that script -- all three name the same directory. A command is
    read only to *locate* the environment (so its interpreter can be named); it
    is never rewritten from what is found here, since a command may well do more
    than activate.

    Returns None when nothing in the value names a directory, as in ``module load
    foundry`` or ``conda activate myenv``, which names an environment by name.
    """
    # check if empty string or empty line
    if not spec:
        return None
    spec = str(spec).strip()
    if not spec:
        return None
    # a bare path should be the environment or activate script
    if is_env_path(spec):
        path = Path(spec).expanduser()
        if not path.is_dir():
            path = Path(get_words(spec)[0]).expanduser()
        if path.name in ACTIVATE_NAMES and path.parent.name in BIN_DIRS:
            return path.parent.parent
        return path
    # otherwise pick the activate script out of a command that sources one:
    # `source <env>/bin/activate` puts the environment two levels above it
    for word in get_words(spec):
        path = Path(word).expanduser()
        if path.name in ACTIVATE_NAMES and path.parent.name in BIN_DIRS:
            return path.parent.parent
    return None


def env_python(spec):
    """
    Find python executable within a given env path
    """
    # find root of env path
    root = env_root(spec)
    if root is None:
        return None
    # find python executable in root of env
    for name in ("python3", "python"):
        python = find_executable_in_env(root, name)
        if python:
            return python
    return None


def activation_command(spec):
    """
    Shell line that puts an environment on PATH for the rest of a job script.

    Only a bare path is turned into a line. A value that is already a command is
    handed to the shell exactly as written -- rebuilding it from the environment
    it happens to mention would quietly drop the rest of what it does, such as
    the ``module load`` half of ``module load Python/3.12 && source .../activate``.
    """
    # if None, return empty string
    if not spec:
        return ""
    # if not just an env path, assume its a whole command
    if not is_env_path(spec):
        return str(spec).strip()
    # if just an env path, get root of env
    root = env_root(spec)
    if root is None:
        return str(spec).strip()
    bin_dir = env_bin(root) or root / "bin"
    # a venv (and conda's base) ships an activate script; sourcing it is what the
    # environment's own documentation tells you to do
    for name in ACTIVATE_NAMES[:2]:
        if (bin_dir / name).is_file():
            return "source {}".format(shlex.quote(str(bin_dir / name)))
    # a conda env has no activate script of its own -- conda's shell function
    # takes the prefix directly
    if (root / "conda-meta").is_dir():
        return "conda activate {}".format(shlex.quote(str(root)))
    # nothing to source: PATH is all the console scripts actually need
    return 'export PATH={}:"$PATH"'.format(shlex.quote(str(bin_dir)))
