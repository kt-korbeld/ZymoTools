"""
Command-line interface and pipeline orchestrator for the generation of Zymogen designs.

Subcommands
-----------
    hotspots            Generate hotspot-patch BindCraft JSONs.
    rfd3-job            Run one RFdiffusion3 design round.
    screen              Run/continue the self-resubmitting SLURM screen.
    status              Read-only progress of a screen.
    linker              Compute fusion-linker length.
    fuse                Fuse binder+target with linker
    view-hotspots       Launch the Shiny hotspot viewer.
    view-linker         Launch the Shiny linker viewer.
    patch-bindcraft     Implement a patch to BindCraft to add new termini loss
    run                 Run the whole pipeline: hotspots -> screen.
"""

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

from . import __version__
from .config import PipelineState

# Each backend for generating binders has its own slurm module
STATUS_REPORTS = {"bindcraft": ".bindcraft_slurm",
                  "bindcraft2": ".bindcraft2_slurm",
                  "boltzgen": ".boltzgen_slurm",
                  "rfd3": ".rfd3_slurm",}
# The screen handler and flag for each backend
SCREEN_COMMANDS = {"bindcraft": ("cmd_screen_bc", "bindcraft"),
                   "bindcraft2": ("cmd_screen_bc2", "bindcraft2"),
                   "boltzgen": ("cmd_screen_bg", "boltzgen"),
                   "rfd3": ("cmd_screen_rfd3", "foundry"),}
# --------------------------------------------------------------------------
# launch resubmit script
# --------------------------------------------------------------------------
def find_wrapper_executable():
    """
    Command that re-invokes the current CLI in an environment-independent manner.
    This way, the pipeline can be run without being in the correct venv.
    This is used in the resubmission pipeline, where the correct venv might not be present.
    """
    repo_root = Path(__file__).resolve().parent.parent
    wrapper_path = repo_root / "zymotools.py"
    # if pipeline.py is excecuted, this file should exist at repo root
    if wrapper_path.is_file():
        return [sys.executable, str(wrapper_path)]
    # if installed through pip as zymotools cli tool, use that
    return [sys.executable, "-m", "zymotools"]


def verify_launcher(cmd):
    """
    Check if the executable found using find_wrapper_executable actually runs,
    by calling --version. The input cmd should be just the executable found by
    this function, not a full subcommand, as --version crashes when combined with other flags.
    """
    # try running the cmd with --version
    try:
        proc = subprocess.run(list(cmd) + ["--version"], capture_output=True, text=True, timeout=120)
    # return error if the cmd cannot be run
    except (OSError, subprocess.SubprocessError) as exc:
        raise SystemExit("cannot run the resubmit launcher: {}\n  {}".format(
            exc, shlex.join([str(c) for c in cmd])))
    # if it fails but does not cause an OSError, cmd does not work for other reasons.
    if proc.returncode != 0:
        raise SystemExit("the resubmit launcher does not work:\n  {}\n{}".format(
            shlex.join([str(c) for c in cmd] + ["--version"]), (proc.stderr or proc.stdout).strip()[:500]))
    return cmd

def build_screen_resubmit_cmd(args, pipeline='bindcraft'):
    """
    Reconstruct the exact screen command, including path and arguments
    so each self-resubmission behaves identically to the first.
    """
    launcher = verify_launcher(find_wrapper_executable())
    # reconstruct bindcraft screen command
    if pipeline == "bindcraft":
        cmd = launcher + ["screen", "bindcraft",
                "--inputs", str(args.inputs),
                "--bindcraft", str(args.bindcraft),
                "--slurm-bc", str(args.slurm_bc),
                "--filters", str(args.filters),
                "--advanced", str(args.advanced),]
    # reconstruct bindcraft2 screen command
    if pipeline == "bindcraft2":
        cmd = launcher + ["screen", "bindcraft2",
                "--inputs", str(args.inputs),
                "--bindcraft2", str(args.bindcraft2),
                "--slurm-bc2", str(args.slurm_bc2),]
        if getattr(args, "af2_params", None):
            cmd += ["--af2-params", str(args.af2_params)]
    # reconstruct boltzgen screen command
    if pipeline == "boltzgen":
        cmd = launcher + ["screen", "boltzgen",
                "--inputs", str(args.inputs),
                "--boltzgen", str(args.boltzgen),
                "--slurm-bg", str(args.slurm_bg),
                "--protocol", str(args.protocol),
                "--num-designs", str(args.num_designs),
                "--budget", str(args.budget),]
    # reconstruct rfd3 screen command
    if pipeline == "rfd3":
        cmd = launcher + ["screen", "rfd3",
                "--inputs", str(args.inputs),
                "--foundry", str(args.foundry),
                "--checkpoints", str(args.checkpoints),
                "--outdir", str(args.outdir),
                "--slurm-rfd3", str(args.slurm_rfd3),
                "--budget", str(args.budget),
                "--num-designs", str(args.num_designs),
                "--batch-size", str(args.batch_size),
                "--mpnn-seqs", str(args.mpnn_seqs),
                "--min-iptm", str(args.min_iptm),
                "--min-plddt", str(args.min_plddt),
                "--max-ipae", str(args.max_ipae),
                "--max-rmsd", str(args.max_rmsd),
                "--foundry-shim", str(args.foundry_shim),
                "--step-scale", str(args.step_scale),
                "--gamma-0", str(args.gamma_0),
                "--job-time", str(args.job_time),]
    # settings for controller script are the same across commands
    cmd += ["--min-ratio", str(args.min_ratio),
                "--min-traj", str(args.min_traj),
                "--nr-jobs", str(args.nr_jobs),
                "--controller-time", str(args.controller_time),
                "--controller-mem", str(args.controller_mem),
                "--controller-cpus", str(args.controller_cpus),]
    return cmd

# --------------------------------------------------------------------------
# Add defaults
# --------------------------------------------------------------------------

def add_bindcraft_defaults(args):
    """
    add paths for default bindcraft files. Cannot be included from the start 
    in argparse since the bindcraft path must be specified first.
    """
    bc = Path(args.bindcraft)
    if getattr(args, "slurm_bc", None) is None:
        args.slurm_bc = bc / "bindcraft.slurm"
    if getattr(args, "filters", None) is None:
        args.filters = bc / "settings_filters" / "default_filters.json"
    if getattr(args, "advanced", None) is None:
        args.advanced = bc / "settings_advanced" / "default_4stage_multimer.json"
    return args

def add_bindcraft2_defaults(args):
    """
    add path for default BindCraft2 slurm script. Cannot be included from the
    start in argparse since the bindcraft2 path must be specified first.
    Unlike BindCraft1, BindCraft2 has no separate filters/advanced settings.
    everything is specified in the input .json.
    """
    bc2 = Path(args.bindcraft2)
    if getattr(args, "slurm_bc2", None) is None:
        args.slurm_bc2 = bc2 / "bindcraft.slurm"
    return args

def add_boltzgen_defaults(args):
    """
    generate default boltzgen files if neccesary. 
    Default slurm scripts are not provided the same way 
    they are in bindcraft, so must be generated
    """
    from .boltzgen_io import default_slurm, parse_slurm
    # if not specified, set an output root beside the inputs list. It is a root
    # holding one directory per input, not one input's directory: the finish check
    # counts rows per hotspot.
    if getattr(args, "outdir", None) is None:
        args.outdir = Path(args.inputs).resolve().parent / "boltzgen-screen"
    args.outdir = Path(args.outdir)
    args.outdir.mkdir(parents=True, exist_ok=True)
    # if not specified, generate default slurm to use as template
    if getattr(args, "slurm_bg", None) is None:
        slurm = default_slurm(args)
        slurmpath = args.outdir / "boltzgen_slurm_default.sh"
        with open(slurmpath, "w") as f:
            f.write(slurm)
        args.slurm_bg = slurmpath
    else:
        slurm = Path(args.slurm_bg).read_text()
    # a hand-edited template stays authoritative: read its settings back out
    args = parse_slurm(slurm, args)
    return args

def check_foundry_env(spec):
    """
    Check the environment specified in the --foundry flag. 
    Fail before anything is submitted
    """
    from .util import env_root, find_executable_in_env, is_env_path
    root = env_root(spec)
    if root is None:
        return spec
    if not Path(root).is_dir():
        problem = "no such directory: {}".format(root)
    else:
        missing = [t for t in ("rfd3", "mpnn", "rf3")
                   if find_executable_in_env(root, t) is None]
        problem = "{} not found in {}".format(", ".join(missing), root) if missing else None
    if problem is None:
        return spec
    if is_env_path(spec):
        raise SystemExit("--foundry {}: {}.\nPass the foundry venv or conda "
                         "environment (pip install \"rc-foundry[all]\" into it), its "
                         "activate script, or a command that activates it.".format(
                             spec, problem))
    print("warning: --foundry {}: {}.\n  the command is used as written; check it "
          "if every job fails at its first foundry call.".format(spec, problem),
          file=sys.stderr)
    return spec

def add_rfd3_defaults(args):
    """
    generate default RFdiffusion3 files if neccesary
    Default slurm scripts are not provided the same way 
    they are in bindcraft, so must be generated
    """
    from .rfd3_io import default_slurm, parse_slurm
    # the environment flag is what every foundry call in the job depends on, so
    # it is checked before a default job script is written around it
    check_foundry_env(args.foundry)
    # if not specified, set output root based on the directory holding the inputs
    if getattr(args, "outdir", None) is None:
        args.outdir = Path(args.inputs).resolve().parent / "rfd3-screen"
    args.outdir = Path(args.outdir)
    args.outdir.mkdir(parents=True, exist_ok=True)
    # if not specified, generate default slurm to use as template
    if getattr(args, "slurm_rfd3", None) is None:
        slurmpath = args.outdir / "rfd3_slurm_default.sh"
        slurm = default_slurm(args, shlex.join(find_wrapper_executable()))
        slurmpath.write_text(slurm)
        args.slurm_rfd3 = slurmpath
    else:
        slurm = Path(args.slurm_rfd3).read_text()
    # overwrite with input slurm if provided
    args = parse_slurm(slurm, args)
    return args


# --------------------------------------------------------------------------
# command implementations
# --------------------------------------------------------------------------
def cmd_hotspots(args):
    """
    run the generate_hotspots function with the arguments provided in ``args``
    save the result as a pipeline state in a json file. 
    """
    # run the generate_hotspots function, returning the csv, txt and json paths
    from .hotspots import generate_hotspots
    patch_csv, inputs_txt, inputs_paths = generate_hotspots(
        struc=args.struc, pipeline=args.pipeline, template=args.json, 
        target=args.target, step=args.step, depth=args.depth, 
        minsize=args.minsize, outdir=args.outdir, chain=args.chain,)
    if not inputs_paths:
        return 1
    # update the pipeline state and save paths of output
    state = PipelineState.load_or_empty(args.outdir)
    state.set(struc=args.struc, pipeline=args.pipeline, outdir=args.outdir, template=args.json,
              patch_csv=patch_csv, inputs_txt=inputs_txt, n_patches=len(inputs_paths))
    state.save()
    # print outcome of command
    print("\nGenerated {} hotspot patches".format(len(inputs_paths)))
    print("  patches CSV : {}".format(patch_csv))
    print("  ordered list: {}".format(inputs_txt))
    print("  pipeline    : {}".format(state.path))
    return 0


def cmd_screen_bc(args):
    """
    Run the BindCraft slum screen command with the arguments provided in args
    """
    # run the run_screen function to analyze outputs and (re)run slurm jobs
    from .bindcraft_slurm import run_screen
    args = add_bindcraft_defaults(args)
    resubmit_cmd = build_screen_resubmit_cmd(args, pipeline='bindcraft')
    run_screen(
        inputs_file=args.inputs, slurm_bc=args.slurm_bc, filters=args.filters,
        advanced=args.advanced, resubmit_cmd=resubmit_cmd,
        min_ratio=args.min_ratio, min_traj=args.min_traj, nr_jobs=args.nr_jobs,
        controller_time=args.controller_time, controller_mem=args.controller_mem,
        controller_cpus=args.controller_cpus,)
    return 0

def cmd_screen_bc2(args):
    """
    Run the BindCraft2 slum screen command with the arguments provided in args
    """
    from .bindcraft2_slurm import run_screen
    args = add_bindcraft2_defaults(args)
    resubmit_cmd = build_screen_resubmit_cmd(args, pipeline='bindcraft2')
    run_screen(
        inputs_file=args.inputs, slurm_bc2=args.slurm_bc2, bindcraft2_dir=args.bindcraft2,
        resubmit_cmd=resubmit_cmd, af2_params=args.af2_params,
        min_ratio=args.min_ratio, min_traj=args.min_traj, nr_jobs=args.nr_jobs,
        controller_time=args.controller_time, controller_mem=args.controller_mem,
        controller_cpus=args.controller_cpus,)
    return 0

def cmd_screen_bg(args):
    """
    Run the boltzgen slum screen command with the arguments provided in args
    """
    # run the run_screen function to analyze outputs and (re)run slurm jobs
    from .boltzgen_slurm import run_screen
    args = add_boltzgen_defaults(args)
    resubmit_cmd = build_screen_resubmit_cmd(args, pipeline='boltzgen')
    run_screen(
        inputs_file=args.inputs, slurm_bg=args.slurm_bg, output_path=args.outdir, 
        resubmit_cmd=resubmit_cmd, boltzgen_cmd=args.boltzgen, max_des=args.budget,
        min_ratio=args.min_ratio, min_traj=args.min_traj, nr_jobs=args.nr_jobs,
        controller_time=args.controller_time, controller_mem=args.controller_mem,
        controller_cpus=args.controller_cpus,)
    return 0

def cmd_screen_rfd3(args):
    """
    Run the RFdiffusion 3 screen command with the arguments provided in args  
    """
    # run the run_screen function to analyze outputs and (re)run slurm jobs
    from .rfd3_slurm import run_screen
    args = add_rfd3_defaults(args)
    resubmit_cmd = build_screen_resubmit_cmd(args, pipeline='rfd3')
    run_screen(
        inputs_file=args.inputs, slurm_rfd3=args.slurm_rfd3, outdir_root=args.outdir,
        resubmit_cmd=resubmit_cmd, budget=args.budget,
        min_ratio=args.min_ratio, min_traj=args.min_traj, nr_jobs=args.nr_jobs,
        controller_time=args.controller_time, controller_mem=args.controller_mem,
        controller_cpus=args.controller_cpus,)
    return 0


def cmd_rfd3_job(args):
    """
    Run one RFdiffusion3 design round inside a SLURM job:
    diffuse backbones, design sequences with ProteinMPNN, refold with RF3, and
    write the per-job metrics CSV the screen controller counts.
    """
    from .rfd3_slurm import run_design_job
    run_design_job(
        settings=args.settings, outdir=args.outdir, job_prefix=args.job_prefix,
        num_designs=args.num_designs, batch_size=args.batch_size,
        mpnn_seqs=args.mpnn_seqs, min_iptm=args.min_iptm,
        min_plddt=args.min_plddt, max_ipae=args.max_ipae, max_rmsd=args.max_rmsd,
        step_scale=args.step_scale, gamma_0=args.gamma_0,
        use_shim=args.foundry_shim,)
    return 0

def resolve_pipeline_state(args):
    """
    Returns (inputs_list, pipeline, screen_outdir) from the flags, filling anything
    unset from the pipeline.json that hotspots wrote.
    """
    inputs = args.inputs
    pipeline = getattr(args, "pipeline", None)
    outdir = getattr(args, "screen_outdir", None)
    if args.outdir is not None:
        state = PipelineState.load_or_empty(args.outdir)
        inputs = inputs or state.get("inputs_txt")
        pipeline = pipeline or state.get("pipeline")
        outdir = outdir or state.get("screen_outdir")
    # the screen's output root defaults the same way each backend's own screen
    # command defaults it, so status looks where the jobs actually wrote
    if outdir is None and inputs is not None:
        outdir = Path(inputs).resolve().parent / "{}-screen".format(pipeline or "rfd3")
    return inputs, pipeline, outdir

def cmd_status(args):
    """
    Print read-only progress for every input of a screen, for any backend.
    """
    import importlib
    inputs, pipeline, screen_outdir = resolve_pipeline_state(args)
    if inputs is None:
        print("Provide --inputs or --outdir (with a pipeline.json).", file=sys.stderr)
        return 2
    if pipeline not in STATUS_REPORTS:
        print("Unknown pipeline {!r}; pass --pipeline {{{}}}.".format(
            pipeline, ",".join(sorted(STATUS_REPORTS))), file=sys.stderr)
        return 2
    status_report = importlib.import_module(
        STATUS_REPORTS[pipeline], package=__package__).status_report

    print("pipeline: {}   inputs: {}".format(pipeline, inputs))
    rows = status_report(inputs, screen_outdir, budget=args.budget,
                         min_ratio=args.min_ratio, min_traj=args.min_traj)
    if not rows:
        print("No inputs found in {}".format(inputs))
        return 0

    header = "{:<9} {:>6} {:>7} {:>7} {:>8}  {}".format(
        "STATE", "traj", "design", "target", "ratio", "input")
    print(header)
    print("-" * len(header))
    n_done = 0
    for r in rows:
        if r["state"] == "finished":
            n_done += 1
        print("{:<9} {:>6} {:>7} {:>7} {:>8}  {}".format(
            r["state"][:9],
            r.get("trajectories", "-"), r.get("designs", "-"),
            r.get("target", "-"),
            "n/a" if r.get("ratio") is None else r["ratio"],
            r["input"]))
    print("\n{}/{} inputs finished.".format(n_done, len(rows)))
    return 0


def cmd_fuse(args):
    """
    Fuse binder+target designs into single-chain constructs, skipping any whose
    termini the linker cannot reach around the protein.
    """
    from .fuse import fuse_designs, collect_designs
    designs = collect_designs(args.struc)
    if not designs:
        print("No structures found in {}".format(args.struc), file=sys.stderr)
        return 2
    outdir = args.outdir or Path(args.struc).parent / "fused"
    print("Fusing {} design(s) with a {} aa linker ({})".format(
        len(designs), len(args.linker), args.linker))
    rows, report = fuse_designs(
        designs, outdir, linker=args.linker, order=args.order,
        binder_chain=args.binder_chain, target_chain=args.target_chain,
        check_only=args.check_only, opt_term=args.optimize_term, 
        replace_term=args.replace_term, num_conf=args.num_conf, keep=args.keep,
        sidechains=not args.no_sidechains, seed=args.seed,
        gridstep=args.gridstep, padding=args.padding, rad=args.rad,
        verbose=args.verbose, buffer=args.buffer)
    n_bridge = sum(1 for r in rows if r["bridgeable"])
    n_built = sum(1 for r in rows if r["output"])
    print("\n{}/{} designs can be bridged by a {} aa linker".format(
        n_bridge, len(rows), len(args.linker)))
    if not args.check_only:
        print("{}/{} constructs built".format(n_built, len(rows)))
    print("  report: {}".format(report))
    return 0


def cmd_linker(args):
    """
    calculate the required linker length from the command line
    """
    from .linker import compute_linker, residues_for_length

    length, _ = compute_linker(
        struc=args.struc, start_res=args.start, end_res=args.end,
        chain_start=args.chain_start, chain_end=args.chain_end,
        gridstep=args.gridstep, padding=args.padding, rad=args.rad,
        progress=lambda m: print("[linker]", m),)
    minimum, buffered = residues_for_length(length, buffer=args.buffer)
    print("\nShortest solvent path: {:.2f} Å".format(length))
    print("Minimum linker: {} aa (3.8 Å/aa)".format(minimum))
    print("With +{} buffer: {} aa".format(args.buffer, buffered))
    return 0

def launch_shiny(app_ref, args):
    """
    wrapper to launch shiny app, checks if shiny is properly installed
    """
    try:
        from shiny import run_app
    except ImportError:
        print("shiny is not installed. Try: pip install shiny", file=sys.stderr)
        return 1
    print("Launching {} on http://{}:{}".format(app_ref, args.host, args.port))
    run_app(app_ref, host=args.host, port=args.port,
            launch_browser=not args.no_browser)
    return 0

def cmd_view_hotspots(args):
    """
    use launch_shiny to launch hotspot inspection app
    """
    return launch_shiny("zymotools.apps.hotspot_viewer:app", args)

def cmd_view_linker(args):
    """
    use launch_shiny to launch linker distance app
    """
    return launch_shiny("zymotools.apps.linker_viewer:app", args)


def cmd_patch_bindcraft(args):
    """
    Apply the target-binder termini distance loss feature to a BindCraft installation.
    """
    from .bindcraft_patch import patch_bindcraft
    print("Patching BindCraft install at {}".format(args.bindcraft))
    try:
        result = patch_bindcraft(args.bindcraft, backend=args.backend)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print("Error: {}".format(exc), file=sys.stderr)
        return 1
    print("  backend: {}".format(result["backend"]))
    if not result["colabdesign_utils_patched"] and not result["json_files_updated"]:
        print("Already up to date; nothing to change.")
    elif result["backend"] == "bindcraft2":
        print("Done: bindcraft/loss.py {}.".format(
            "patched" if result["colabdesign_utils_patched"] else "unchanged"))
    else:
        print("Done: BindCraft functions {}, {} settings_advanced json file(s) updated.".format(
            "patched" if result["colabdesign_utils_patched"] else "unchanged",
            len(result["json_files_updated"])))
    return 0


def cmd_run(args):
    """
    Orchestrate: generate hotspots, then either pause for curation or screen.
    Works for any of the three backends specified with --pipeline
    """
    # generate hotspots, stop if hotspots failed
    rc = cmd_hotspots(args)
    if rc != 0:
        return rc
    # load current status from status json
    state = PipelineState.load(args.outdir)
    inputs_txt = state.get("inputs_txt")
    handler, env_flag = SCREEN_COMMANDS[args.pipeline]
    # auto mode used in full pipeline
    if args.auto:
        if getattr(args, env_flag, None) is None:
            print("--auto with --pipeline {} requires --{}.".format(
                args.pipeline, env_flag), file=sys.stderr)
            return 2
        print("\n=== Auto mode: launching the {} screen on all generated patches ==="
              .format(args.pipeline))
        args.inputs = Path(inputs_txt)
        # the screen writes into its own root; record it so status can find it
        # later without being told again
        state.set(screen_outdir=getattr(args, "screen_outdir", None)
                  or Path(args.outdir) / "{}-screen".format(args.pipeline))
        args.outdir = Path(state.get("screen_outdir"))
        state.save()
        return globals()[handler](args)
    # if not on auto, stop here and hand off to the user.
    print("\n=== Hotspots generated. Pausing for curation. ===")
    print("Next steps:")
    print("  1. Inspect/curate patches:")
    print("       python pipeline.py view-hotspots")
    print("     (load the PDB and {})".format(state.get("patch_csv")))
    print("  2. Start the SLURM screen on your chosen list:")
    print("       python pipeline.py screen {} --inputs {} \\".format(
        args.pipeline, inputs_txt))
    print("           --{} <environment>".format(env_flag))
    print("  3. Track progress without submitting:")
    print("       python pipeline.py status --outdir {}".format(state.path.parent))
    return 0


# --------------------------------------------------------------------------
# argument parsing
# --------------------------------------------------------------------------
def add_hotspot_args(p, with_pipeline=True):
    p.add_argument("--struc", required=True, type=Path,
                   help="Input structure to generate hotspots on.")
    p.add_argument("--json", type=Path, default=None,
                   help="Template BindCraft JSON or BoltzGen YAML. If omitted, a default is generated.")
    p.add_argument("--target", type=int, nargs="+", default=None,
                   help="Target residues the hotspot patch should cover.")
    # disable if arguments are folded into the whole pipeline to prevent argument clashes
    if with_pipeline:
        p.add_argument("--pipeline", type=str, default="bindcraft",
                       choices=["bindcraft", "bindcraft2", "boltzgen", "rfd3"],
                       help="Pipeline for which the files and hotspots will be generated. (default: bindcraft)")
    p.add_argument("--chain", type=str, default=None,
                   help="Restrict scoring to this chain/segid.")
    p.add_argument("--step", type=float, default=6,
                   help="Patch expansion step size in Å (default: 6).")
    p.add_argument("--depth", type=int, default=None,
                   help="Number of expansion steps (default bindcraft: 3, boltzgen/rfd3: 1).")
    p.add_argument("--minsize", type=int, default=None,
                   help="Minimum residues for a patch (default bindcraft: 6, boltzgen/rfd3: 2).")
    p.add_argument("--outdir", type=Path, default=Path("."),
                   help="Output directory (default: current dir).")

def add_rfd3_job_args(p):
    p.add_argument("--settings", type=Path, required=True,
                   help="RFdiffusion3 input YAML for one hotspot patch.")
    p.add_argument("--outdir", type=Path, required=True,
                   help="Output directory for this hotspot.")
    p.add_argument("--job-prefix", type=str, default="job0_",
                   help="Prefix keeping parallel jobs apart in a shared outdir.")
    p.add_argument("--num-designs", type=int, default=8,
                   help="Backbones to diffuse (default: 8).")
    p.add_argument("--batch-size", type=int, default=8,
                   help="Designs per diffusion batch (default: 8).")
    p.add_argument("--mpnn-seqs", type=int, default=4,
                   help="ProteinMPNN sequences per backbone (default: 4).")
    p.add_argument("--min-iptm", type=float, default=0.8,
                   help="Minimum RF3 ipTM for a design to pass (default: 0.8).")
    p.add_argument("--min-plddt", type=float, default=80.0,
                   help="Minimum RF3 pLDDT of the binder chain for a design to pass (default: 80")
    p.add_argument("--max-ipae", type=float, default=10.0,
                   help="Maximum RF3 *minimum* binder-target interface PAE, in A (default: 10).")
    p.add_argument("--max-rmsd", type=float, default=2.0,
                   help="Maximum binder CA-RMSD to the diffused backbone, in A (default: 2).")
    p.add_argument("--foundry-shim", choices=["on", "off"], default="on",
                   help="Run the foundry CLIs through foundry_shim.py (default: on); ")
    p.add_argument("--step-scale", type=float, default=3.0,
                   help="RFD3 sampler step scale (default: 3).")
    p.add_argument("--gamma-0", type=float, default=0.2,
                   help="RFD3 sampler gamma_0 (default: 0.2).")

def add_screen_bc_args(p, require_inputs=True, require_bindcraft=True):
    p.add_argument("--inputs", type=Path, required=require_inputs,
                   help="Text file with one path to an input JSON or YAML file per line.")
    p.add_argument("--bindcraft", required=require_bindcraft, type=Path, default=None,
                   help="Path to bindcraft directory (used to derive defaults).")
    p.add_argument("--slurm-bc", type=Path, default=None,
                   help="BindCraft SLURM script (default: <bindcraft>/bindcraft.slurm).")
    p.add_argument("--filters", type=Path, default=None,
                   help="Filters JSON (default: <bindcraft>/settings_filters/default_filters.json).")
    p.add_argument("--advanced", type=Path, default=None,
                   help="Advanced settings JSON (default: <bindcraft>/settings_advanced/default_4stage_multimer.json).")

def add_screen_bc2_args(p, require_inputs=True, require_bindcraft2=True):
    p.add_argument("--inputs", type=Path, required=require_inputs,
                   help="Text file with one path to an input campaign JSON per line.")
    p.add_argument("--bindcraft2", required=require_bindcraft2, type=Path, default=None,
                   help="Path to the BindCraft2 install directory (used to derive defaults).")
    p.add_argument("--slurm-bc2", type=Path, default=None,
                   help="BindCraft2 SLURM script (default: <bindcraft2>/bindcraft.slurm).")
    p.add_argument("--af2-params", type=Path, default=None,
                   help="optional directory holding model parameters")

def add_screen_bg_args(p, with_outdir=True, require_inputs=True, require_boltzgen=True):
    p.add_argument("--inputs", type=Path, required=require_inputs,
                   help="Text file with one path to an input YAML file per line.")
    p.add_argument("--boltzgen", type=str, required=require_boltzgen,
                   help="command to activate BoltzGen venv or conda environment.")
    # disable if arguments are folded into the whole pipeline to prevent argument clashes
    if with_outdir:
        p.add_argument("--outdir", type=Path, default=None,
                       help="Output directory. If none is provided, the path inside the inputs file is used.")
    p.add_argument("--slurm-bg", type=Path, default=None,
                   help="BoltzGen SLURM script. If none is provided, a default one will be generated")
    p.add_argument("--protocol", type=str, default='protein-anything',
                   help="BoltzGen protocol. only specify if no template slurm is provided (default: protein-anything)")
    p.add_argument("--num-designs", type=int, default=10000,
                   help="Number of total BoltzGen designs. only specify if no template slurm is provided (default: 10000)")
    p.add_argument("--budget", type=int, default=100,
                   help="Number of final BoltzGen designs. only specify if no template slurm is provided (default: 100)")

def add_screen_rfd3_args(p, with_outdir=True, require_inputs=True, require_foundry=True):
    p.add_argument("--inputs", type=Path, required=require_inputs,
                   help="Text file with one path to an RFdiffusion3 input YAML per line.")
    p.add_argument("--foundry", type=str, required=require_foundry,
                   help="foundry's venv or conda environment: its directory, its "
                        "activate script, or a command that activates it. The job "
                        "script activates it and calls its interpreter by path.")
    p.add_argument("--checkpoints", type=Path, default=Path.home() / ".foundry" / "checkpoints",
                   help="foundry checkpoint directory (default: ~/.foundry/checkpoints).")
    # disable if arguments are folded into the whole pipeline to prevent argument clashes
    if with_outdir:
        p.add_argument("--outdir", type=Path, default=None, 
                       help="Output directory if none provided, input directory is used")
    p.add_argument("--slurm-rfd3", type=Path, default=None,
                   help="RFdiffusion3 SLURM script. If none is provided, a default one will be generated")
    p.add_argument("--budget", type=int, default=100,
                   help="Number of final designs that count as finished (default: 100).")
    p.add_argument("--num-designs", type=int, default=8,
                   help="Backbones diffused per job (default: 8).")
    p.add_argument("--batch-size", type=int, default=4,
                   help="Designs per diffusion batch; one batch shares one sampled "
                        "binder length (default: 4).")
    p.add_argument("--mpnn-seqs", type=int, default=8,
                   help="ProteinMPNN sequences designed per backbone (default: 8).")
    p.add_argument("--min-iptm", type=float, default=0.8,
                   help="Minimum RF3 ipTM for a design to pass (default: 0.8).")
    p.add_argument("--min-plddt", type=float, default=80.0,
                   help="Minimum RF3 pLDDT of the binder chain for a design to pass "
                        "(default: 80). Whole-complex pLDDT is recorded, not filtered.")
    p.add_argument("--max-ipae", type=float, default=10.0,
                   help="Maximum RF3 minimum binder-target interface PAE, in A (default: 10).")
    p.add_argument("--max-rmsd", type=float, default=2.0,
                   help="Maximum binder CA-RMSD to the diffused backbone, in A (default: 2).")
    p.add_argument("--foundry-shim", choices=["on", "off"], default="on",
                   help="Run the foundry CLIs through foundry_shim.py, which disables "
                        "the checkpoints' unreachable residue cache and unusable fused "
                        "kernels (default: on). Set to off where foundry runs unpatched.")
    p.add_argument("--step-scale", type=float, default=3.0,
                   help="RFD3 sampler step scale; higher is lower-temperature (default: 3).")
    p.add_argument("--gamma-0", type=float, default=0.2,
                   help="RFD3 sampler gamma_0; lower is lower-temperature (default: 0.2).")
    p.add_argument("--job-time", type=str, default="04:00:00",
                   help="Walltime for each design job (default: 04:00:00).")

def add_screen_c_args(p):
    p.add_argument("--min-ratio", type=float, default=0.01,
                   help="Stop an input if n_pass/n_fail < min-ratio (default: 0.01).")
    p.add_argument("--min-traj", type=int, default=300,
                   help="Don't apply min-ratio until this many failed trajectories (default: 300).")
    p.add_argument("--nr-jobs", type=int, default=5,
                   help="Parallel SLURM jobs per input (default: 5).")
    p.add_argument("--controller-time", default="00:05:00",
                   help="Walltime for the controller job (default: 00:05:00).")
    p.add_argument("--controller-mem", default="512M",
                   help="Memory for the controller job (default: 512M).")
    p.add_argument("--controller-cpus", type=int, default=1,
                   help="CPUs for the controller job (default: 1).")

def add_status_args(p):
    p.add_argument("--inputs", type=Path, default=None,
                   help="Inputs list (default: inputs_txt from --outdir's pipeline.json).")
    p.add_argument("--outdir", type=Path, default=None,
                   help="Pipeline output dir containing pipeline.json.")
    p.add_argument("--pipeline", type=str, default=None,
                   choices=["bindcraft", "bindcraft2", "boltzgen", "rfd3"],
                   help="Backend that ran the screen (default: from pipeline.json).")
    p.add_argument("--screen-outdir", type=Path, default=None,
                   help="Root the screen wrote into. BoltzGen and RFdiffusion3 "
                        "need it; BindCraft records its own path per input. "
                        "(default: from pipeline.json, else <inputs dir>/"
                        "<pipeline>-screen)")
    p.add_argument("--budget", type=int, default=None,
                   help="Final designs that count as finished. Unset, BindCraft "
                        "uses each settings JSON's own number_of_final_designs "
                        "and the other two use 300.")
    p.add_argument("--min-ratio", type=float, default=0.01,
                   help="Stop an input if n_pass/n_fail < min-ratio (default: 0.01).")
    p.add_argument("--min-traj", type=int, default=300,
                   help="Don't apply min-ratio until this many failed trajectories (default: 300).")

def add_linker_args(p):
    p.add_argument("--struc", required=True, type=Path,
                   help="Structure (PDB).")
    p.add_argument("--start", required=True, type=int,
                   help="Start residue number.")
    p.add_argument("--end", required=True, type=int,
                   help="End residue number.")
    p.add_argument("--chain-start", required=True,
                   help="Chain ID of start residue.")
    p.add_argument("--chain-end", required=True,
                   help="Chain ID of end residue.")
    p.add_argument("--gridstep", type=float, default=1,
                   help="Grid spacing in Å (default: 1).")
    p.add_argument("--padding", type=float, default=4,
                   help="Grid padding in Å (default: 4).")
    p.add_argument("--rad", type=float, default=3,
                   help="Protein exclusion radius in Å (default: 3).")
    p.add_argument("--buffer", type=int, default=5,
                   help="Extra residues added as a buffer (default: 5).")

def add_fuse_args(p):
    p.add_argument("--struc", required=True, type=Path,
                   help="A design (PDB or mmCIF, optionally gzipped), or a directory of them.")
    p.add_argument("--outdir", type=Path, default=None,
                   help="Where to write constructs and the report (default: /fused).")
    p.add_argument("--linker", type=str, default="GGGSGGGSENLYFQS",
                   help="Linker sequence (default:, a 15 aa GS spacer plus a TEV site).")
    p.add_argument("--buffer", type=int, default=5, 
                   help="nr. of residues on top of minimal linker length. (default: 5)")
    p.add_argument("--order", choices=["auto", "binder-target", "target-binder"], default="binder-target",
                   help="whether target/binder comes first. 'auto' measures both (default: binder-target).")
    p.add_argument("--binder-chain", type=str, default=None,
                   help="Chain of the binder (default: the shorter chain).")
    p.add_argument("--target-chain", type=str, default=None,
                   help="Chain of the target (default: the longest one).")
    p.add_argument("--optimize-term", action='store_true',
                   help="Optimize flexible residues on termini of the binder/target during linker construction")
    p.add_argument("--replace-term", action='store_true',
                  help="Replace flexible residues on the termini of the binder/target with linker during construction")
    p.add_argument("--check-only", action="store_true",
                   help="Only report which designs the linker can bridge, much faster")
    p.add_argument("--num-conf", type=int, default=5000,
                   help="Linker conformations pyDisgro attempts (default: 5000).")
    p.add_argument("--keep", type=int, default=1,
                   help="Conformations to write per design (default: 1).")
    p.add_argument("--no-sidechains", action="store_true",
                   help="Do not sample linker side chains (faster, backbone only).")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed for pyDisgro (default: nondeterministic).")
    p.add_argument("--gridstep", type=float, default=1.0,
                   help="Solvent-grid spacing in A (default: 1).")
    p.add_argument("--padding", type=float, default=4.0,
                   help="Solvent-grid padding around the structure (default: 4).")
    p.add_argument("--rad", type=float, default=3.0,
                   help="Clearance from protein atoms a linker path needs (default: 3).")
    p.add_argument("--verbose", action="store_true",
                   help="Report both junction distances and pyDisgro progress.")

def add_shiny_args(p):
    p.add_argument("--host", default="127.0.0.1",
                   help="Bind host (default: 127.0.0.1).")
    p.add_argument("--port", type=int, default=8000,
                   help="Bind port (default: 8000).")
    p.add_argument("--no-browser", action="store_true",
                   help="Do not auto-open a browser.")

def build_parser():
    parser = argparse.ArgumentParser(
        prog="zymotools",
        description="tool suite for zymogen design")
    parser.add_argument("--version", action="version", version="%(prog)s " + __version__)
    sub = parser.add_subparsers(dest="command", required=True)

    # generation of hotspots
    p = sub.add_parser("hotspots", help="Generate hotspot-patch BindCraft JSONs.")
    add_hotspot_args(p)
    p.set_defaults(func=cmd_hotspots)
    
    # create pipeline for a single rfd3+mpnn+rf3 job
    p = sub.add_parser("rfd3-job", help="Run one RFdiffusion3 design round (used inside screen).")
    add_rfd3_job_args(p)
    p.set_defaults(func=cmd_rfd3_job)
    # general resubmission pipelines
    p = sub.add_parser("screen", help="Run/continue the self-resubmitting SLURM screen.")
    subsub = p.add_subparsers(dest="command", required=True)
    # create resubmission pipeline for bindcraft
    p_bc = subsub.add_parser("bindcraft", help="Run/continue BindCraft SLURM screen.")
    add_screen_bc_args(p_bc, require_inputs=True)
    add_screen_c_args(p_bc)
    p_bc.set_defaults(func=cmd_screen_bc)
    # create resubmission pipeline for bindcraft2
    p_bc2 = subsub.add_parser("bindcraft2", help="Run/continue BindCraft2 SLURM screen.")
    add_screen_bc2_args(p_bc2, require_inputs=True)
    add_screen_c_args(p_bc2)
    p_bc2.set_defaults(func=cmd_screen_bc2)
    # create resubmission pipeline for boltzgen
    p_bg = subsub.add_parser("boltzgen", help="Run/continue BoltzGen SLURM screen.")
    add_screen_bg_args(p_bg, require_inputs=True, require_boltzgen=True)
    add_screen_c_args(p_bg)
    p_bg.set_defaults(func=cmd_screen_bg)
    # create resubmission pipeline for rfd3+mpnn+rf3
    p_rfd3 = subsub.add_parser("rfd3", help="Run/continue RFdiffusion3 SLURM screen.")
    add_screen_rfd3_args(p_rfd3, require_inputs=True, require_foundry=True)
    add_screen_c_args(p_rfd3)
    p_rfd3.set_defaults(func=cmd_screen_rfd3)
    # check status of pipeline
    p = sub.add_parser("status", help="Read-only progress of a screen.")
    add_status_args(p)
    p.set_defaults(func=cmd_status)

    # compute linker length
    p = sub.add_parser("linker", help="Compute fusion-linker length.")
    add_linker_args(p)
    p.set_defaults(func=cmd_linker)
     # fuse binder+target into one chain
    p = sub.add_parser("fuse", help="Fuse binder+target with linker")
    add_fuse_args(p)
    p.set_defaults(func=cmd_fuse)
    
    # visualization of hotspots
    p = sub.add_parser("view-hotspots", help="Launch the Shiny hotspot viewer.")
    add_shiny_args(p)
    p.set_defaults(func=cmd_view_hotspots)
    # visualization of linker length
    p = sub.add_parser("view-linker", help="Launch the Shiny linker viewer.")
    add_shiny_args(p)
    p.set_defaults(func=cmd_view_linker)

    # apply the target-binder termini distance loss feature to a BindCraft install
    p = sub.add_parser("patch-bindcraft",
                       help="Add the target-binder termini distance loss (both fusion orders) to a BindCraft install.")
    p.add_argument("--bindcraft", required=True, type=Path,
                   help="Path to the BindCraft installation to patch.")
    p.add_argument("--backend", choices=["bindcraft", "bindcraft2"], default=None,
                   help="Which BindCraft this install is (default: guessed from its directory layout).")
    p.set_defaults(func=cmd_patch_bindcraft)
    # the whole pipeline
    p = sub.add_parser("run", help="Run the whole pipeline: hotspots -> screen.")
    runsub = p.add_subparsers(dest="backend", required=True)
    for name, adder in (("bindcraft", add_screen_bc_args),
                        ("bindcraft2", add_screen_bc2_args),
                        ("boltzgen", add_screen_bg_args),
                        ("rfd3", add_screen_rfd3_args)):
        p_run = runsub.add_parser(name, help="hotspots -> {} screen.".format(name))
        add_hotspot_args(p_run, with_pipeline=False)
        # --inputs comes from the hotspots step, and the environment flag is only
        # needed with --auto, so neither is required up front
        if name == "bindcraft":
            adder(p_run, require_inputs=False, require_bindcraft=False)
        elif name == "bindcraft2":
            adder(p_run, require_inputs=False, require_bindcraft2=False)
        elif name == "boltzgen":
            adder(p_run, with_outdir=False, require_inputs=False, require_boltzgen=False)
        else:
            adder(p_run, with_outdir=False, require_inputs=False, require_foundry=False)
        p_run.add_argument("--screen-outdir", type=Path, default=None,
                           help="Root the screen writes into "
                                "(default: <outdir>/{}-screen).".format(name))
        add_screen_c_args(p_run)
        mode = p_run.add_mutually_exclusive_group()
        mode.add_argument("--pause", dest="auto", action="store_false",
                          help="Stop after hotspots for manual curation (default).")
        mode.add_argument("--auto", dest="auto", action="store_true",
                          help="Immediately screen all generated patches.")
        p_run.set_defaults(auto=False, func=cmd_run, pipeline=name)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)

if __name__ == "__main__":
    sys.exit(main())
