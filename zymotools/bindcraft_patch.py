"""
Applies the target-binder termini distance loss feature to a BindCraft
installation: a loss pulling the binder's and target's termini together,
so a fusion linker has less distance to bridge.

Supports both the original BindCraft (martinpacesa/BindCraft) and BindCraft2
(PacesaLab/BindCraft2). Their install layouts and loss systems differ enough
(settings_advanced/*.json + functions/colabdesign_utils.py vs. a single
campaign JSON + bindcraft/loss.py's @loss(...) registry) that they need
separate patches; patch_bindcraft() picks the right one, by directory layout
when possible or by an explicit ``backend``.
"""

import json
import subprocess
from pathlib import Path

PATCH_DIR = Path(__file__).resolve().parent / "patches"
PATCH_FILE_BC1 = PATCH_DIR / "bindcraft_tb_termini_order.patch"
PATCH_FILE_BC2 = PATCH_DIR / "bindcraft2_tb_termini.patch"

# new keys added to every settings_advanced/*.json that lacks it. BindCraft2 has
# no equivalent step: any weights_<name> is accepted automatically once a loss
# with that name is registered in bindcraft/loss.py.
NEW_JSON_KEYS = [
    ("use_termini_distance_loss-TB", False),
    ("weights_teminiTB_loss", 0),
    ("termini_loss_order-TB", "binder-target"),
]


def detect_backend(bindcraft_dir):
    """
    Guess whether ``bindcraft_dir`` holds an original BindCraft or a
    BindCraft2 install, from directory layout alone. Returns 'bindcraft',
    'bindcraft2', or None if that is ambiguous (both or neither layout
    marker present), in which case the caller must be told explicitly.
    """
    bindcraft_dir = Path(bindcraft_dir)
    is_bc1 = (bindcraft_dir / "settings_advanced").is_dir() and (bindcraft_dir / "functions").is_dir()
    is_bc2 = (bindcraft_dir / "settings" / "core").is_dir() and (bindcraft_dir / "bindcraft").is_dir()
    if is_bc1 and not is_bc2:
        return "bindcraft"
    if is_bc2 and not is_bc1:
        return "bindcraft2"
    return None


def _add_missing_keys(text):
    """
    Return the input text with any of NEW_JSON_KEYS it lacks appended before the final
    closing brace, or None if it already has all of them.
    """
    data = json.loads(text)
    missing = [(k, v) for k, v in NEW_JSON_KEYS if k not in data]
    if not missing:
        return None
    idx = text.rstrip().rfind("}")
    if idx == -1:
        raise ValueError("not a JSON object")
    before, after = text[:idx].rstrip(), text[idx:]
    if not before.endswith(","):
        before += ","
    new_lines = ",\n".join(
        "    {}: {}".format(json.dumps(k), json.dumps(v)) for k, v in missing)
    new_text = before + "\n" + new_lines + "\n" + after
    json.loads(new_text)  # fail loudly here, not on the caller's next read
    return new_text


def patch_json_defaults(bindcraft_dir, verbose=True):
    """
    Add any missing termini-loss-TB keys to every settings_advanced/*.json
    under bindcraft_dir. Returns the list of filenames actually changed.
    """
    settings_dir = Path(bindcraft_dir) / "settings_advanced"
    if not settings_dir.is_dir():
        raise FileNotFoundError("no settings_advanced/ under {}".format(bindcraft_dir))
    changed = []
    for fp in sorted(settings_dir.glob("*.json")):
        new_text = _add_missing_keys(fp.read_text())
        if new_text is not None:
            fp.write_text(new_text)
            changed.append(fp.name)
            if verbose:
                print("  updated {}".format(fp.name))
        elif verbose:
            print("  {} already has all keys, skipping".format(fp.name))
    return changed


def _already_generalized_bc1(text):
    """
    True if colabdesign_utils.py is already in the updated state.
    """
    def_marker = "def add_tb_termini_distance_loss"
    n_defs = text.count(def_marker)
    if n_defs != 1:
        return False
    def_head = text.split(def_marker, 1)[1].split("\n", 1)[0]
    if "order" not in def_head:
        return False

    call_marker = "add_tb_termini_distance_loss(af_model"
    n_calls = text.count(call_marker)
    if n_calls != 1:
        return False
    call_line = text.split(call_marker, 1)[1]
    call_stmt = call_line[:call_line.find(")") + 1]
    return "order=" in call_stmt


def patch_colabdesign_utils(bindcraft_dir, verbose=True):
    """
    Apply the stored git patch to functions/colabdesign_utils.py. Returns
    True if applied, False if it was already generalized (no-op). Raises
    RuntimeError if the patch cannot be applied cleanly
    """
    target = Path(bindcraft_dir) / "functions" / "colabdesign_utils.py"
    if not target.is_file():
        raise FileNotFoundError("no functions/colabdesign_utils.py under {}".format(bindcraft_dir))
    if _already_generalized_bc1(target.read_text()):
        if verbose:
            print("  colabdesign_utils.py: already generalized, skipping")
        return False

    # Deliberately plain patch, not git apply so it fails more explicity
    root = str(Path(bindcraft_dir).resolve())
    result = subprocess.run(
        ["patch", "-p1", "--batch", "--fuzz=0", "-i", str(PATCH_FILE_BC1)],
        cwd=root, capture_output=True, text=True)

    # Verify by re-reading the file
    orig_backup = target.with_suffix(target.suffix + ".orig")
    reject = target.with_suffix(target.suffix + ".rej")
    if not _already_generalized_bc1(target.read_text()):
        if orig_backup.is_file():
            # restore the untouched original
            orig_backup.replace(target)
        if reject.is_file():
            reject.unlink()
        raise RuntimeError("the stored patch did not end up generalizing")
    if orig_backup.is_file():
        orig_backup.unlink()
    if verbose:
        print("  colabdesign_utils.py: patched")
    return True


def _already_patched_bc2(text):
    """
    True if bindcraft/loss.py already registers the termini_distance_tb loss.
    """
    return "def termini_distance_tb_loss" in text and "@loss('termini_distance_tb'" in text


def patch_loss_py(bindcraft_dir, verbose=True):
    """
    Apply the stored patch to bindcraft/loss.py (BindCraft2), registering the
    termini_distance_tb loss. Returns True if applied, False if it was
    already there (no-op). Raises RuntimeError if the patch cannot be
    applied cleanly.
    """
    target = Path(bindcraft_dir) / "bindcraft" / "loss.py"
    if not target.is_file():
        raise FileNotFoundError("no bindcraft/loss.py under {}".format(bindcraft_dir))
    if _already_patched_bc2(target.read_text()):
        if verbose:
            print("  bindcraft/loss.py: already patched, skipping")
        return False

    root = str(Path(bindcraft_dir).resolve())
    result = subprocess.run(
        ["patch", "-p1", "--batch", "--fuzz=0", "-i", str(PATCH_FILE_BC2)],
        cwd=root, capture_output=True, text=True)

    orig_backup = target.with_suffix(target.suffix + ".orig")
    reject = target.with_suffix(target.suffix + ".rej")
    if not _already_patched_bc2(target.read_text()):
        if orig_backup.is_file():
            orig_backup.replace(target)
        if reject.is_file():
            reject.unlink()
        raise RuntimeError("the stored patch did not end up registering "
                           "termini_distance_tb: {}".format((result.stderr or result.stdout).strip()))
    if orig_backup.is_file():
        orig_backup.unlink()
    if verbose:
        print("  bindcraft/loss.py: patched")
    return True


def patch_bindcraft(bindcraft_dir, backend=None, verbose=True):
    """
    Apply the whole target-binder termini distance loss feature to a
    BindCraft or BindCraft2 install. Safe to run twice: already-applied
    pieces are skipped. ``backend`` selects which patch to apply
    ('bindcraft' or 'bindcraft2'); left unset, it is guessed from the
    install's directory layout, and a RuntimeError asks for it explicitly
    when that guess is ambiguous.
    """
    bindcraft_dir = Path(bindcraft_dir)
    if backend is None:
        backend = detect_backend(bindcraft_dir)
        if backend is None:
            raise RuntimeError(
                "cannot tell whether {} is a BindCraft or BindCraft2 install "
                "from its layout; pass --backend bindcraft or --backend "
                "bindcraft2 explicitly.".format(bindcraft_dir))
    if backend not in ("bindcraft", "bindcraft2"):
        raise ValueError("backend must be 'bindcraft' or 'bindcraft2', got {!r}".format(backend))
    if backend == "bindcraft":
        json_changed = patch_json_defaults(bindcraft_dir, verbose=verbose)
        py_patched = patch_colabdesign_utils(bindcraft_dir, verbose=verbose)
    else:
        json_changed = []
        py_patched = patch_loss_py(bindcraft_dir, verbose=verbose)
    return {"colabdesign_utils_patched": py_patched, "json_files_updated": json_changed, "backend": backend}
