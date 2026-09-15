"""
Reading, writing and validating BindCraft settings JSON files.
Used by hotspots.py (which writes one settings JSON per patch)
and bindcraft_slurm.py (which validates and resubmits).
"""

import json
import os
from pathlib import Path

def default_settings(base_name, struc, design_root=".", chain="A"):
    """
    A minimal BindCraft settings dict for a structure.
    """
    return {
        "design_path": os.path.join(design_root, base_name),
        "binder_name": base_name,
        "starting_pdb": str(struc),
        "chains": chain,
        "target_hotspot_residues": "",
        "lengths": [65, 100],
        "number_of_final_designs": 100,}


def load_settings(path):
    """
    Load and return a BindCraft settings JSON as a dict.
    """
    return json.loads(Path(path).read_text())


def validate_settings(path, required_keys_in=None):
    """
    Load a settings JSON and raise if any required key is missing.
    """
    # Keys every BindCraft settings JSON must carry for the screen to run it.
    required_keys = ["design_path", "binder_name", "lengths",
                     "starting_pdb", "chains",
                     "target_hotspot_residues", 
                     "number_of_final_designs",]
    # use default if not specified
    if required_keys_in != None:
        required_keys = required_keys_in
    # load in data and check required keys
    data = load_settings(path)
    missing = [k for k in required_keys if k not in data]
    if missing:
        raise ValueError("{}: missing required keys: {}".format(path, missing))
    return data


def write_patch_settings(patch, out_json, base_name_out, struc, template=None,
                         design_root=".", chain="A"):
    """
    Write one BindCraft settings JSON for a hotspot patch.
    If no template is specified, generate default .json
    patch:     a list of residue numbers
    out_json:  the path to the output .json
    base_name_out: the binder name and output directory
    struc:     the path to the input PDB file used in the default template
    template:  the template JSON file to copy
    design_root: the root used in the default template 
    """
    # if no template is provided, generate default settings
    if template is None:
        data = default_settings(base_name_out, struc, design_root=design_root,
                                chain=chain)
    # if template is provided, keep settings the same but give unique design path
    else:
        data = load_settings(template)
        data["design_path"] = os.path.join(data["design_path"], base_name_out)
    data["target_hotspot_residues"] = ",".join(str(i) for i in patch)
    out_json = Path(out_json).with_suffix('.json')
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(data, indent=4))
    return str(out_json)
