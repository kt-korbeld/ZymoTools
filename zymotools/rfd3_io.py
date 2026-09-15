"""
Reading, writing and validating RFdiffusion3 (foundry) settings YAML files.
Used by hotspots.py (which writes one settings YAML per patch)
and rfd3_slurm_slurm.py (which validates them and resubmits).
Unlike the other pipelines, foundry requires an in-house pipeline 
for running through all the separate foundry models, which can be found in rfd3_slurm.py 
"""

import re
import textwrap
import yaml
from pathlib import Path

from .util import activation_command, env_python

def target_residue_range(struc, chain):
    """
    First and last residue number of chain in struc.
    Used to specify the whole input region inside the yaml
    """
    from .structure import load_universe
    u_full, at_sel = load_universe(struc, chain=chain)
    resids = at_sel.residues.resids
    return int(min(resids)), int(max(resids))


def build_contig(struc, chain, lengths):
    """
    Contig string for binder design: 
    for a binder with length specified in (lengths), 
    """
    first, last = target_residue_range(struc, chain)
    return "{}-{},/0,{}{}-{}".format(lengths[0], lengths[1], chain, first, last)


def contig_binder_lengths(contig):
    """
    check the allowed binder lengths in the contig. 
    The designed segment is the leading, chain-less component: i.e 
    65-100 in 65-100,/0,C1-250.
    """
    first = str(contig).split(",")[0].strip()
    m = re.fullmatch(r"(\d+)(?:-(\d+))?", first)
    if not m:
        return None
    lo = int(m.group(1))
    return (lo, int(m.group(2)) if m.group(2) else lo)


def default_settings(base_name, struc, design_root=".", chain="A",
                     lengths=(65, 100), hotspot_atoms="ALL"):
    """
    A minimal RFD3 input dict for a structure, based on (base_name).
    design_root is currently unused but kept in for consistency with other pipelines.
    """
    return {base_name: {
        "input": str(struc),
        "contig": build_contig(struc, chain, lengths),
        # Place the ORI token relative to the hotspots rather than the target COM;
        # the foundry docs report this works best for PPI.
        "infer_ori_strategy": "hotspots",
        "is_non_loopy": True,
        "select_hotspots": {},}}


def load_settings(path):
    """
    Load and return an RFD3 input file (JSON or YAML) as a dict.
    """
    return yaml.safe_load(Path(path).read_text())


def iter_specs(data):
    """
    give every (name, spec) every design entry in the input YAML.
    """
    for name, spec in data.items():
        # global_args is part of the entry, not an entry itself.
        if name == "global_args":
            continue
        if isinstance(spec, dict):
            yield name, spec


def validate_settings(path, required_keys_in=None):
    """
    Load an RFD3 input file and raise if any entry is missing a required key.
    """
    # Keys every entry must carry for the screen to run it.
    required_keys = ["input", "contig", "select_hotspots"]
    # use default if not specified
    if required_keys_in != None:
        required_keys = required_keys_in
    # load in data and check required keys for each design entry
    data = load_settings(path)
    if not isinstance(data, dict):
        raise ValueError("{}: expected a mapping of design names".format(path))
    specs = list(iter_specs(data))
    if not specs:
        raise ValueError("{}: contains no design entries".format(path))
    for name, spec in specs:
        missing = [k for k in required_keys if k not in spec]
        if missing:
            raise ValueError("{} [{}]: missing required keys: {}".format(
                path, name, missing))
        if not spec["select_hotspots"]:
            raise ValueError("{} [{}]: select_hotspots is empty".format(path, name))
    return data


def hotspots_in_contig(spec):
    """
    True if every residue in spec["select_hotspots"] falls inside a fixed segment of
    the contig. RFD3 requires this so the model is aware of the hotspots.
    """
    # collect the fixed (chain-labelled) ranges of the contig
    ranges = []
    for part in str(spec["contig"]).split(","):
        part = part.strip()
        m = re.fullmatch(r"([A-Za-z])(\d+)(?:-(\d+))?", part)
        if m:
            lo = int(m.group(2))
            hi = int(m.group(3)) if m.group(3) else lo
            ranges.append((m.group(1), lo, hi))
    # every hotspot must sit inside one of them
    for res in spec["select_hotspots"]:
        m = re.fullmatch(r"([A-Za-z])(\d+)", str(res))
        if not m:
            return False
        chain, num = m.group(1), int(m.group(2))
        if not any(c == chain and lo <= num <= hi for c, lo, hi in ranges):
            return False
    return True


def write_patch_settings(patch, out_yaml, base_name_out, struc, template=None,
                         design_root=".", chain="A",
                         lengths=(65, 100), hotspot_atoms="ALL"):
    """
    Write one RFD3 input YAML for a hotspot patch.
    If no template is specified, generate a default one.
    patch:     a list of residue numbers
    out_yaml:  the path to the output .yaml
    base_name_out: the design name, used as the top-level key
    struc:     the path to the input PDB file
    template:  a template RFD3 input file to copy (contig, sampler hints, ...)
    design_root: unused, kept for signature parity with the other backends
    """
    # if no template is provided, generate default settings
    if template is None:
        data = default_settings(base_name_out, struc, design_root=design_root,
                                chain=chain, lengths=lengths,
                                hotspot_atoms=hotspot_atoms)
    # if template is provided, keep its settings but re-key it to this patch
    else:
        loaded = load_settings(template)
        specs = list(iter_specs(loaded))
        if not specs:
            raise ValueError("{}: contains no design entries".format(template))
        data = {base_name_out: dict(specs[0][1])}
        if "global_args" in loaded:
            data["global_args"] = loaded["global_args"]
    spec = data[base_name_out]
    # point at this structure, and rebuild the contig if the template had none
    spec["input"] = str(struc)
    if not spec.get("contig"):
        spec["contig"] = build_contig(struc, chain, lengths=lengths)
    # add the hotspots: RFD3 selects per residue, per atom
    spec["select_hotspots"] = {
        "{}{}".format(chain, res): hotspot_atoms for res in patch}
    # warn rather than fail: a hand-written template contig may be intentional
    if not hotspots_in_contig(spec):
        print("  warning: hotspots of {} fall outside its contig '{}'".format(
            base_name_out, spec["contig"]))
    # write output
    out_yaml = Path(out_yaml).with_suffix('.yaml')
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    with open(out_yaml, 'w') as yaml_file:
        yaml.dump(data, yaml_file, sort_keys=False)
    return str(out_yaml)


def default_slurm(args, wrapper_cmd):
    """
    Minimal template for running RFdiffusion3 jobs.
    runs the rfd3-job command specified in the cli, which runs the entire
    pipeline from RFD3->MPNN->RF3 and saves the output. 
    the input.yaml, output dir and job prefix are handled by update_sbatch()
    the exact amount of memory and partition name are dependent on the specific system,
    and might need to be adjusted. in case 32GB memory is not sufficient, it can be 
    increased, or the num_desings, batch_size, or mpnn_seqs can be decreased. 
    """
    # turn the environment flag into a line the job's shell understands, and
    # name its interpreter for the in-job driver where that is possible
    activate = activation_command(args.foundry)
    python = env_python(args.foundry)
    export_python = "export FOUNDRY_PYTHON={}\n".format(python) if python else ""
    # generate template sbatch script
    template = textwrap.dedent("""\
    #!/bin/bash
    #SBATCH --job-name=rfd3
    #SBATCH --nodes=1
    #SBATCH --ntasks=1
    #SBATCH --cpus-per-task=6
    #SBATCH --gres=gpu:1
    #SBATCH --mem=32G
    #SBATCH --time={time}
    #SBATCH --output=rfd3_slurm%A.log

    {activate}
    {export_python}export FOUNDRY_CHECKPOINT_DIRS={checkpoints}

    {wrapper} rfd3-job \\
      --settings input.yaml \\
      --outdir {outdir} \\
      --job-prefix job0_ \\
      --num-designs {num_designs} \\
      --batch-size {batch_size} \\
      --mpnn-seqs {mpnn_seqs} \\
      --min-iptm {min_iptm} \\
      --min-plddt {min_plddt} \\
      --max-ipae {max_ipae} \\
      --max-rmsd {max_rmsd} \\
      --foundry-shim {foundry_shim} \\
      --step-scale {step_scale} \\
      --gamma-0 {gamma_0}
    """).format(
        time=getattr(args, "job_time", "04:00:00"),
        activate=activate,
        export_python=export_python,
        checkpoints=args.checkpoints,
        wrapper=wrapper_cmd,
        outdir=args.outdir,
        num_designs=args.num_designs,
        batch_size=args.batch_size,
        mpnn_seqs=args.mpnn_seqs,
        min_iptm=args.min_iptm,
        min_plddt=args.min_plddt,
        max_ipae=args.max_ipae,
        max_rmsd=args.max_rmsd,
        foundry_shim=getattr(args, "foundry_shim", "on"),
        step_scale=args.step_scale,
        gamma_0=args.gamma_0,)
    return template


def parse_slurm(slurm, args, parseflags=None):
    """
    Parses an RFD3 slurm file and updates the args with the parameters in the file.
    If parseflags is not specified, the design counts and filter thresholds are read.
    """
    if parseflags == None:
        parseflags = {'num_designs': 'num-designs', 'batch_size': 'batch-size',
                      'mpnn_seqs': 'mpnn-seqs', 'min_iptm': 'min-iptm',
                      'min_plddt': 'min-plddt', 'max_ipae': 'max-ipae',
                      'max_rmsd': 'max-rmsd', 'foundry_shim': 'foundry-shim',
                      'step_scale': 'step-scale', 'gamma_0': 'gamma-0',}
    for arg in vars(args):
        if arg not in parseflags.keys():
            continue
        flag = parseflags[arg]
        entry = re.search(r'(--{}\s+)(\S+)'.format(flag), slurm)
        if entry == None:
            continue
        # keep the type argparse gave the flag, so downstream arithmetic still works
        caster = type(getattr(args, arg, None))
        try:
            setattr(args, arg, caster(entry[2]) if caster in (int, float) else entry[2])
        except (TypeError, ValueError):
            setattr(args, arg, entry[2])
    return args
