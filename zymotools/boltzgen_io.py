"""
Reading, writing and validating Boltzgen settings YAML files.
Used by hotspots.py (which writes one settings YAML per patch)
and boltzgen_slurm.py (which validates them and resubmits).
"""

import yaml
import os
import re
import textwrap
from pathlib import Path

def default_slurm(args):
    """
    Minimal template for running BoltzGen jobs
    """
    template = textwrap.dedent(f"""#!/bin/bash
    #SBATCH --nodes 1
    #SBATCH --ntasks 1
    #SBATCH --gpus-per-node=1
    #SBATCH --partition=gpu
    #SBATCH --time 10:00:00
    #SBATCH --output=boltzgen_slurm%A.log

    {args.boltzgen}

    boltzgen run input.yaml \\
      --output {args.outdir} \\
      --protocol {args.protocol} \\
      --num_designs {args.num_designs} \\
      --budget {args.budget} \\
    """)
    return template

def parse_slurm(slurm, args, parseflags=None):
    """
    Parses a BoltzGen slurm file and updates the args with the parameters set in the file
    if parseflags is not specified, only the budget, num_designs and protocol are read
    """
    if parseflags == None:
        parseflags = {'budget':'budget', 'num_designs':'num_designs', 'protocol':'protocol'}
    for arg in vars(args):
        if arg not in parseflags.keys():
            continue
        flag = parseflags[arg]
        entry = re.search(r'(--{}\s+)(\S+)'.format(flag), slurm)
        if entry == None:
            continue
        setattr(args, arg,  entry[2])
    #outdir = re.search(r'(--output\s+)(\S+)', slurm)
    #budget = re.search(r'(--budget\s+)(\S+)', slurm)[2]
    #num_designs = re.search(r'(--budget\s+)(\S+)', slurm)[2]
    return args


def default_settings(base_name, struc, design_root=".", chain="A"):
    """
    A minimal BoltzGen settings dict for a structure.
    """
    template = {'entities': [{'protein': {'id': 'B', 'sequence': '80..140'}},
                {'file': {'path': str(struc),
                 'include': [{'chain': {'id': chain}}],
                 'binding_types': [{'chain': {'id': chain, 'binding': ''}}],
                 'structure_groups': 'all'}}]}
    return template

def load_settings(path):
    """
    Load and return a BoltzGen settings yaml as a dict.
    """
    return yaml.safe_load(Path(path).read_text())


def yaml_path_exists(data, path):
    """
    check if path exists within a yaml architecture. 
    if lists in path, check each entry in list by recursively calling function
    """
    if not path:
        return True
    key = path[0]
    rest = path[1:]
    # if dict, check if key in dict
    if isinstance(data, dict):
        if key in data:
            return yaml_path_exists(data[key], rest)
        return False
    # if list rather than dict, recusively call func
    elif isinstance(data, list):
        return any(yaml_path_exists(item, path) for item in data)
    return False

def validate_settings(path, required_paths_in=None):
    """
    Load a settings yaml and raise if any required key is missing.
    """
    # paths the yaml should contain  for the screen to run it.
    required_paths = [("entities", "protein", "id"),
                      ("entities", "protein", "sequence"),
                      ("entities", "file", "path"),
                      ("entities", "file", "include"),
                      ("entities", "file", "binding_types", "chain", "binding"),]
    # use default if not specified
    if required_paths_in != None:
        required_paths = required_paths_in
    # load in data and check required paths in the yaml file
    data = load_settings(path)
    missing = [' -> '.join(p) for p in required_paths if not yaml_path_exists(data, p)]
    if missing:
        raise ValueError("{}: missing required paths: {}".format(path, missing))
    return data

def write_patch_settings(patch, out_yaml, base_name_out, struc, template=None,
                         design_root=".", chain="A"):
    """
    Write one BoltzGen settings YAML for a hotspot patch.
    If no template is specified, generate default .yaml
    patch:     a list of residue numbers
    out_yaml:  the path to the output .json
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
    # add hotspot to yaml file
    file_entity = [e['file'] for e in data['entities'] if 'file' in e][0]
    binding_chain = file_entity['binding_types'][0]['chain']
    binding_chain['binding'] = ",".join(str(i) for i in patch)
    # set structure path
    file_entity['path'] = str(struc)
    # write output
    out_yaml = Path(out_yaml).with_suffix('.yaml')
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    with open(out_yaml, 'w') as yaml_file:
        yaml.dump(data, yaml_file)
    return str(out_yaml)

