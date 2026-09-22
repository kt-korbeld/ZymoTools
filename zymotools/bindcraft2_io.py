"""
Reading, writing and validating BindCraft2 campaign settings JSON files.
Used by hotspots.py (which writes one campaign JSON per patch) and
bindcraft2_slurm.py (which validates and resubmits).
BindCraft2 requires slightly different formatting of the input files. 
"""

import json
import os
from pathlib import Path


def default_settings(base_name, struc, design_root=".", chain="A"):
    """
    A minimal BindCraft2 campaign settings dict for a structure.
    """
    return {
        "campaign_name": base_name,
        "project_folder": os.path.join(design_root, base_name),
        "targets": [{
            "name": base_name,
            "target_path": str(struc),
            "chains": chain,
            "hotspots": "",
        }],
        "binder_lengths": [65, 100],
        "number_of_final_designs": 100,
    }


def load_settings(path):
    """
    Load and return a BindCraft2 campaign settings JSON as a dict.
    """
    return json.loads(Path(path).read_text())


def validate_settings(path, required_keys_in=None):
    """
    Load a campaign settings JSON and raise if any required key is missing,
    or if it has no target to attach hotspots to.
    """
    # Keys every BindCraft2 campaign JSON must carry for the screen to run it.
    required_keys = ["project_folder", "targets", "number_of_final_designs"]
    if required_keys_in is not None:
        required_keys = required_keys_in
    data = load_settings(path)
    missing = [k for k in required_keys if k not in data]
    if missing:
        raise ValueError("{}: missing required keys: {}".format(path, missing))
    if not data.get("targets"):
        raise ValueError("{}: 'targets' is empty".format(path))
    return data


def write_patch_settings(patch, out_json, base_name_out, struc, template=None,
                         design_root=".", chain="A"):
    """
    Write one BindCraft2 campaign settings JSON for a hotspot patch.
    If no template is specified, generate a default campaign JSON.
    patch:     a list of residue numbers
    out_json:  the path to the output .json
    base_name_out: the campaign/output-folder name for this patch
    struc:     the path to the input PDB file used in the default template
    template:  the template JSON file to copy
    design_root: the root used in the default template's project_folder
    """
    # if no template is provided, generate default settings
    if template is None:
        data = default_settings(base_name_out, struc, design_root=design_root,
                                chain=chain)
    # if template is provided, keep settings the same but give a unique project folder
    else:
        data = load_settings(template)
        if not data.get("targets"):
            raise ValueError("{}: 'targets' is empty; nothing to attach hotspots to".format(template))
        data["project_folder"] = os.path.join(data.get("project_folder", str(design_root)), base_name_out)
    # attach the patch to the first (and normally only) target entry
    data["targets"][0]["hotspots"] = ",".join(str(i) for i in patch)
    out_json = Path(out_json).with_suffix('.json')
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(data, indent=4))
    return str(out_json)
