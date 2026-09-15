"""
SLURM screen controller for RFDiffusion3 hotspot screening as implemented in foundry.
Since foundry has no integrated pipeline including RFD3, MPNN and RF3, an in-house
pipeline for generating the designs and scoring them is provided.
As foundry (currently) produces an error in its checkpoints, a shim is used to avoid this
"""

import csv
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from .rfd3_io import validate_settings, load_settings, contig_binder_lengths
from .util import env_python
from .structure import read_ca_coords, read_chain_sequence, kabsch_rmsd

# --------------------------------------------------------------------------
# Global variables
# --------------------------------------------------------------------------

# RFD3 reassigns chainID, so binder should always be A. this is verified by check_binder_chain
BINDER_CHAIN = "A"
# ...and the target (known structure) is always the other chain of the pair.
TARGET_CHAIN = "B"
# glob name of the outputs from the parallel jobs
METRICS_GLOB = "metrics_*.csv"
# output name of the combined jobs
ALL_DESIGNS_CSV = "all_designs_metrics.csv"
# output names of the designs passing thresholds
FINAL_DESIGNS_CSV = "final_design_stats.csv"

# all metric fields used to filter final design
METRICS_FIELDS = ["design", "backbone", "sequence", "mpnn_confidence",
                  "iptm", "plddt", "pae", "binder_plddt", "interface_pae",
                  "interface_pae_min", "rmsd", "pass"]
# RF3 confidence metrics that have to be extracted from the RF3 outputs
CONF_FIELDS = ["iptm", "plddt", "pae", "binder_plddt", "interface_pae",
               "interface_pae_min"]

# variables for running foundry. due to some errors, a shim must be used. 
# The foundry shim is run as a script so the full package does not have to be imported
FOUNDRY_SHIM = Path(__file__).resolve().parent / "foundry_shim.py"
SHIM_ENV = "BINDER_PIPELINE_FOUNDRY_SHIM"
USE_SHIM = None
OFF_VALUES = ("0", "off", "no", "false", "none")
FOUNDRY_LOG = None

# Each job's designs are named after its prefix, which must be unique per *round*
# as well as per parallel job. RFdiffusion3 defaults to ``skip_existing``, so a
# second round submitted under the same prefix regenerates example IDs that already
# exist, skips every one of them ("No design specifications to run"), and produces
# no new backbones -- the attempt count never grows, and the controller resubmits
# that hotspot forever without advancing. Folding in the SLURM job ID fixes that
# while keeping what ``skip_existing`` is actually for: a job SLURM *requeues*
# keeps its job ID, so it resumes instead of restarting from scratch.
# Expanded by the job script's shell, not here.
JOB_PREFIX_TEMPLATE = "job{}_${{SLURM_JOB_ID:-manual}}_"


# --------------------------------------------------------------------------
# track progress
# --------------------------------------------------------------------------
def count_rows(csv_path):
    """
    Number of data rows in a CSV (excluding the header) 0 if missing.
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        return 0
    with csv_path.open() as fh:
        n = sum(1 for _ in fh)
    return max(0, n - 1)


def get_progress(final_design_path, trajectory_path, verbose=False):
    """
    From the data loaded in from an input .yaml file, use the output path
    to get the number of trajectories and number of successful designs.
    if verbose is set to True, print number of trajectories, designs, and ratio
    """
    # get number of trajectories and number of successful designs, print nrs if verbose
    traj = count_rows(trajectory_path)
    des = count_rows(final_design_path)
    if verbose:
        ratio_str = str(round(des / traj, 3)) if traj else "n/a"
        print("  trajectories: {}, designs: {}, ratio: {}".format(traj, des, ratio_str))
    return traj, des


def check_if_finished(output_path, max_des, min_ratio=0.01, min_traj=300, verbose=True):
    """
    Check whether an input has been screened sufficiently based on if:
    min_traj (default: 300) designs exist with less than min_ratio (default: 0.01)
    success rate, or a sufficient number of final designs (max_des) is reached.
    Returns True if sufficiently run, and False if it requires more runs
    """
    # get paths of neccesary output files
    final_design_path = Path(output_path) / FINAL_DESIGNS_CSV
    trajectory_path = Path(output_path) / ALL_DESIGNS_CSV
    # if nothing has run yet, it is not finished, re-submit.
    if not (os.path.exists(final_design_path) and os.path.exists(trajectory_path)):
        return False
    # get number of trajectories and number of successful designs, print data if verbose
    trajectory_nr, design_nr = get_progress(final_design_path, trajectory_path, verbose=verbose)
    # Return True if sufficient number of designs or ratio too bad, otherwise return False
    if design_nr >= int(max_des):
        if verbose:
            print("  sufficient number of final designs, finished")
        return True
    if trajectory_nr < min_traj:
        if verbose:
            print("  insufficient number of trajectories, running")
        return False
    if design_nr / trajectory_nr < min_ratio:
        if verbose:
            print("  poor ratio of designs to trajectories, finished")
        return True
    if verbose:
        print("  good ratio of designs to trajectories, insufficient final designs, running")
    return False

def read_input_txt(inputs_file):
    """
    Read an inputs list file into a list of settings-YAML paths.
    """
    with open(inputs_file) as f:
        return [line.strip() for line in f if line.strip()]


def outdir_for_input(input_yaml, outdir_root):
    """
    Output directory for one hotspot input. Each input gets its own directory
    under (outdir_root) so that per-hotspot progress can be counted separately.
    """
    return Path(outdir_root) / Path(input_yaml).stem


def aggregate_metrics(outdir, metrics_glob=METRICS_GLOB,
                      all_des=ALL_DESIGNS_CSV,
                      pass_des=FINAL_DESIGNS_CSV):
    """
    compile all output files specified with metrics_glob, and combine into the two csvs:
    all designs go in the all_des file, the successful designs go into the pass_des file.
    Deduplicates on the design id, so a requeued job cannot inflate the counts.
    """
    outdir = Path(outdir)
    rows, seen = [], set()
    # go over all output files, skip duplicates
    for path in sorted(outdir.glob(metrics_glob)):
        with open(path, newline="") as f:
            # go over each row in file and save to seen
            for row in csv.DictReader(f):
                key = row.get("design")
                if key in seen:
                    continue
                seen.add(key)
                rows.append(row)
    # return 0 if all files are empty
    if not rows:
        return 0, 0
    # check if metric passed is true
    passed = [r for r in rows if str(r.get("pass")).lower() == "true"]
    for target, subset in ((all_des, rows), (pass_des, passed)):
        with open(outdir / target, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=METRICS_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(subset)
    return len(rows), len(passed)

# --------------------------------------------------------------------------
# job submission
# --------------------------------------------------------------------------
def update_sbatch(slurm, yaml_path, output, job_prefix):
    """
    Updates a given RFdiffusion3 jobscript with a given input yaml file, output
    directory and job prefix, so that parallel jobs share one output directory
    without overwriting each other's designs.
    """
    # read provided template slurm
    with open(slurm, 'r') as f:
        slurm_new = f.read()
    # matches anything following --outdir, .group(1) extracts just the following bit
    slurm_new = re.sub(r'(--outdir\s+)(\S+)',
                       lambda m: m.group(1) + str(output), slurm_new)
    slurm_new = re.sub(r'(--job-prefix\s+)(\S+)',
                       lambda m: m.group(1) + str(job_prefix), slurm_new)
    slurm_new = re.sub(r'(--settings\s+)(\S+)',
                       lambda m: m.group(1) + str(yaml_path), slurm_new)
    return slurm_new

def submit_sbatch(input_yaml, slurm_rfd3, outdir, nr_jobs):
    """
    Submit (nr_jobs) RFdiffusion3 jobs for one input. Returns job IDs.
    All jobs write into (outdir), kept apart by their --job-prefix.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    job_ids = []
    for ind in range(int(nr_jobs)):
        job_prefix = JOB_PREFIX_TEMPLATE.format(ind)
        # the script's own name stays round-independent; only its contents carry
        # the unexpanded ${SLURM_JOB_ID}
        slurm_out = outdir / "{}-job{}.sh".format(Path(input_yaml).stem, ind)
        slurm_out.write_text(update_sbatch(
            slurm_rfd3, Path(input_yaml).resolve(), outdir.resolve(), job_prefix))
        cmd = ["sbatch", str(slurm_out)]
        print("  {}".format(" ".join(cmd)))
        out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
        # Expect "Submitted batch job 12345".
        job_ids.append(int(out.split()[-1]))
    return job_ids


def submit_controller(jids, resubmit_cmd, time="00:05:00", mem="512M", cpus=1):
    """
    Schedule the next controller iteration after all slurm (jids) finish.
    """
    dep_str = "afterany:" + ":".join(str(i) for i in jids)
    wrap_cmd = shlex.join(resubmit_cmd)
    cmd = ["sbatch",
           "--dependency={}".format(dep_str),
           "--time={}".format(time),
           "--cpus-per-task={}".format(cpus),
           "--mem={}".format(mem),
           "--wrap={}".format(wrap_cmd)]
    print("  scheduling controller (depends on {})".format(dep_str))
    subprocess.run(cmd, check=True)


def run_screen(inputs_file, slurm_rfd3, outdir_root, resubmit_cmd, budget=300,
               min_ratio=0.01, min_traj=300, nr_jobs=5,
               controller_time="00:05:00", controller_mem="512M",
               controller_cpus=1):
    """
    Run one controller iteration: go over each .yaml file in the input txt file,
    check if the file exists and is valid, aggregate what the jobs produced,
    then submit jobs for the first unfinished yaml file and reschedule controller.
    """
    # iterate over the .yaml files in input file in given order
    input_yamls = read_input_txt(inputs_file)
    for input_yaml in input_yamls:
        print("[{}]".format(input_yaml))
        # skip if .yaml file does not exist
        if not os.path.exists(input_yaml):
            print("  {} does not exist, skipping".format(input_yaml))
            continue
        # check if settings in yaml file are valid
        try:
            validate_settings(input_yaml)
        except Exception as e:
            print("  failed to parse: {}".format(e))
            continue
        # combine the per-job metrics written by the jobs of the previous iteration
        outdir = outdir_for_input(input_yaml, outdir_root)
        n_rows, n_pass = aggregate_metrics(outdir)
        if n_rows:
            print("  aggregated {} attempts, {} passing".format(n_rows, n_pass))
        # check if output exists and if the success ratio is high enough
        if check_if_finished(outdir, budget, min_ratio=min_ratio, min_traj=min_traj):
            print("  {} has been run sufficiently".format(input_yaml))
            continue
        # submit new jobs if not finished
        print("  submitting {} jobs for {}".format(nr_jobs, input_yaml))
        jids = submit_sbatch(input_yaml, slurm_rfd3, outdir, nr_jobs)
        print("  submitted job ids: {}".format(jids))
        submit_controller(jids, resubmit_cmd, time=controller_time,
                          mem=controller_mem, cpus=controller_cpus)
        return False  # work remains
    print("All inputs are finished; no further jobs scheduled.")
    return True  # everything finished


def status_report(inputs_file, outdir_root, budget=None, min_ratio=0.01, min_traj=300):
    """
    Read-only progress for every input. Returns a list of per-input dicts.
    """
    budget = 300 if budget is None else budget
    rows = []
    for input_yaml in read_input_txt(inputs_file):
        row = {"input": input_yaml}
        if not os.path.exists(input_yaml):
            row["state"] = "missing"
            rows.append(row)
            continue
        try:
            validate_settings(input_yaml)
        except Exception as e:
            row["state"] = "invalid: {}".format(e)
            rows.append(row)
            continue
        outdir = outdir_for_input(input_yaml, outdir_root)
        traj, des = get_progress(outdir / FINAL_DESIGNS_CSV, outdir / ALL_DESIGNS_CSV)
        finished = check_if_finished(outdir, budget, min_ratio=min_ratio,
                                     min_traj=min_traj, verbose=False)
        row.update({
            "design_path": str(outdir),
            "trajectories": traj,
            "designs": des,
            "target": budget,
            "ratio": round(des / traj, 4) if traj else None,
            "state": "finished" if finished else "running",})
        rows.append(row)
    return rows

# --------------------------------------------------------------------------
# Code dealing with the foundry environment / shim
# --------------------------------------------------------------------------

def read_shebang_interpreter(path):
    """
    read the shebang, the first line of code in a python executable indicating the intepreter,
    which shoud be something like: #!/venv-foundry/bin/python3 (all pip installed code should include it)
    """
    # try reading the shebang, limit to 512 bytes, read as utf-8, replace unknown characters
    try:
        with open(path, "rb") as fh:
            first = fh.readline(512).decode("utf-8", "replace").strip()
    # return none if error, or doesnt start with #!
    except OSError:
        return None
    if not first.startswith("#!"):
        return None
    # extract the path
    parts = first[2:].strip().split()
    if not parts:
        return None
    # check if the interpreter is actually python
    name = Path(parts[0]).name
    if name == "env" or not name.startswith("python"):
        return None
    return parts[0] if Path(parts[0]).is_file() else None


def foundry_python(tool="rfd3"):
    """
    Return the interpreter that has foundry installed, which is not necessarily this one.
    Assumes the script is currently in a venv or conda env that contains the foundry cli interface.
    this is needed because this code may live in its own venv, while the jobscript activates a foundry venv
    So sys.executable cannot be assumed to import the right foundry models.
    If the environment variable FOUNDY_PYTHON is set (which the default jobscript does), use that instead.
    """
    # try to get foundry location set in global FOUNDRY_PYTHON variable
    explicit = os.environ.get("FOUNDRY_PYTHON")
    if explicit:
        return explicit
    # if not specified, try to find location of rfd3 executable
    script = shutil.which(tool)
    if script:
        from_shebang = read_shebang_interpreter(script)
        if from_shebang:
            return from_shebang
        for name in ("python3", "python"):
            candidate = Path(script).parent / name
            if candidate.is_file():
                return str(candidate)
        return sys.executable
    # if no rfd3 executable, fall back to whichever environment is activated
    for var in ("VIRTUAL_ENV", "CONDA_PREFIX"):
        candidate = env_python(os.environ.get(var))
        if candidate:
            return str(candidate)
    return sys.executable

def shim_enabled():
    """
    Check whether global python variable USE_SHIM is set. 
    if not, check SHIM_ENV, which should contain the environment variable that sets the shim
    if SHIM_ENV is empty, simply return 'on' and assume shim is on by default
    """
    if USE_SHIM is not None:
        return USE_SHIM
    return os.environ.get(SHIM_ENV, "on").strip().lower() not in OFF_VALUES

def set_foundry_shim(setting):
    """
    enable or disable the shim by checking/setting the global variable USE_SHIM
    """
    global USE_SHIM
    if setting is None:
        USE_SHIM = None
    elif isinstance(setting, str):
        USE_SHIM = setting.strip().lower() not in OFF_VALUES
    else:
        USE_SHIM = bool(setting)
    return shim_enabled()

def foundry_cmd(tool, *args):
    """
    Build a command that runs one foundry CLI through the foundry_shim
    """
    if not shim_enabled():
        return [tool] + list(args)
    return [foundry_python(tool), str(FOUNDRY_SHIM), tool] + list(args)


def run_cmd(cmd, label):
    """
    Run one foundry command. Returns True on success; a failure is logged and
    reported rather than raised, so one bad backbone does not kill the job.
    """
    print("  [{}] {}".format(label, " ".join(str(c) for c in cmd)), flush=True)
    proc = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if FOUNDRY_LOG is not None:
        with open(FOUNDRY_LOG, "a") as fh:
            fh.write("$ {}\n{}{}\n".format(
                " ".join(str(c) for c in cmd), proc.stdout, proc.stderr))
    if proc.returncode != 0:
        print("  [{}] failed (rc={}):\n{}".format(
            label, proc.returncode, proc.stderr[-2000:]), flush=True)
        return False
    return True


def checkpoint_path(name):
    """
    Locate a foundry checkpoint by filename across FOUNDRY_CHECKPOINT_DIRS,
    also check the default ~/.foundry/checkpoints.
    """
    dirs = (os.environ.get("FOUNDRY_CHECKPOINT_DIRS")
            or os.environ.get("FOUNDRY_CHECKPOINTS_DIR", "")).split(":")
    dirs = [d for d in dirs if d] + [str(Path.home() / ".foundry" / "checkpoints")]
    for d in dirs:
        candidate = Path(d) / name
        if candidate.is_file():
            return candidate
    return None

# --------------------------------------------------------------------------
# The in-house pipeline for foundry: RFD3-MPNN-RF3
# --------------------------------------------------------------------------

def run_rfd3(settings, rfd3_dir, job_prefix, num_designs, batch_size,
             step_scale=3.0, gamma_0=0.2):
    """
    Run RFDifussion 3 to generate binder backbones for one hotspot input.
    All designs within one batch share a single binder length, so use several batches. 
    step_scale and gamma_0 default to the reccomended settings for PPI designability.
    """
    rfd3_dir.mkdir(parents=True, exist_ok=True)
    n_batches = max(1, math.ceil(int(num_designs) / int(batch_size)))
    cmd = foundry_cmd("rfd3", "design",
                      "inputs={}".format(Path(settings).resolve()),
                      "out_dir={}".format(rfd3_dir.resolve()),
                      "global_prefix={}".format(job_prefix),
                      "n_batches={}".format(n_batches),
                      "diffusion_batch_size={}".format(int(batch_size)),
                      "inference_sampler.step_scale={}".format(step_scale),
                      "inference_sampler.gamma_0={}".format(gamma_0))
    run_cmd(cmd, "rfd3")
    # rfd3 writes <prefix><key>_<batch>_model_<n>.cif.gz alongside a .json
    return sorted(p for p in rfd3_dir.glob("{}*.cif.gz".format(job_prefix))
                  if "_model_" in p.name and "denoised" not in p.name
                  and "noisy" not in p.name)


def run_mpnn(backbone, mpnn_dir, name, n_seqs, temperature=0.1):
    """
    Design n_seqs sequences on the binder chain of one backbone with
    ProteinMPNN, keeping the target fixed. Returns a list of
    (design_id, sequence, confidence, structure_path).
    """
    mpnn_dir.mkdir(parents=True, exist_ok=True)
    ckpt = checkpoint_path("proteinmpnn_v_48_020.pt")
    if ckpt is None:
        print("  [mpnn] proteinmpnn_v_48_020.pt not found; "
              "set FOUNDRY_CHECKPOINT_DIRS", flush=True)
        return []
    # the fasta is opened in append mode by mpnn, so clear it to keep retries clean
    fasta = mpnn_dir / "{}.fa".format(name)
    if fasta.exists():
        fasta.unlink()
    cmd = foundry_cmd("mpnn",
                      "--model_type", "protein_mpnn",
                      "--checkpoint_path", ckpt,
                      "--is_legacy_weights", "True",
                      "--structure_path", Path(backbone).resolve(),
                      # without an explicit --name, "x.cif.gz" would be named "x.cif"
                      "--name", name,
                      "--designed_chains", BINDER_CHAIN,
                      "--batch_size", int(n_seqs),
                      "--number_of_batches", 1,
                      "--temperature", temperature,
                      "--write_fasta", "True",
                      "--write_structures", "True",
                      "--out_directory", mpnn_dir.resolve())
    if not run_cmd(cmd, "mpnn") or not fasta.exists():
        return []
    return parse_mpnn_fasta(fasta, mpnn_dir)


def parse_mpnn_fasta(fasta, mpnn_dir):
    """
    Read an MPNN fasta into (design_id, sequence, confidence, structure_path).
    Headers look like >name_b0_d0, sequence_recovery=0.5352, older versions
    use confidence isntead of sequence_recovery, so accept both. 
    the fasta reports the whole complex, so the binder seq should be
    read from the CIF using read_chain_seq()
    """
    designs, design_id, conf, seq = [], None, None, []

    def flush():
        if design_id is not None:
            designs.append((design_id, "".join(seq), conf,
                            mpnn_dir / "{}.cif".format(design_id)))

    with open(fasta) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                flush()
                seq = []
                parts = line[1:].split(",")
                design_id = parts[0].strip()
                conf = None
                for part in parts[1:]:
                    key = part.strip().split("=", 1)[0]
                    if key in ("confidence", "sequence_recovery"):
                        try:
                            conf = float(part.split("=", 1)[1])
                        except ValueError:
                            conf = None
                continue
            if line:
                seq.append(line)
    flush()
    return designs


def run_rf3_batch(structures_dir, rf3_dir, diffusion_batch_size=1, template_target=True):
    """
    Refold every design in structures_dir with a single rf3 fold call.
    diffusion_batch_size is 1 by default
    because RF3 otherwise samples 5 structures per input
    and ranks them, five times the cost for a number we then threshold anyway.

    template_target (default True) passes RF3's own ``template_selection``
    override for the whole target chain (TARGET_CHAIN, "B"), so RF3 conditions
    on the target's known structure via a distogram template while the binder
    chain (A) still folds unconstrained. This is the same target-templated,
    binder-free convention as foundry's own binder-design example
    (docs/examples/*_template_antigen_and_framework.json in the rf3 model).
    Selection syntax and behaviour: rf3.utils.inference.apply_template_selection
    / rf3.data.ground_truth_template (the "is_input_file_templated" annotation).
    """
    rf3_dir.mkdir(parents=True, exist_ok=True)
    cmd = foundry_cmd("rf3", "fold",
                      "inputs={}".format(Path(structures_dir).resolve()),
                      "out_dir={}".format(rf3_dir.resolve()),
                      "diffusion_batch_size={}".format(int(diffusion_batch_size)))
    if template_target:
        cmd.append("template_selection=[{}]".format(TARGET_CHAIN))
    return run_cmd(cmd, "rf3")


def rf3_results(rf3_dir, design_id, conf_fields=CONF_FIELDS):
    """
    return (confidences, model_path) for one design of a batched RF3 run. 
    A failed RF3 run simply has no output directory so its metrics stay None
    """
    out_dir = Path(rf3_dir) / design_id
    if not out_dir.is_dir():
        return dict.fromkeys(conf_fields), None
    return parse_rf3_confidences(out_dir, conf_fields=conf_fields), find_rf3_model(out_dir)


def find_rf3_model(out_dir):
    """
    Find predicted RF3 structures in output directory and return first candidate
    """
    models = sorted(list(Path(out_dir).rglob("*.cif.gz")) + list(Path(out_dir).rglob("*.cif")))
    return models[0] if models else None


def parse_rf3_confidences(out_dir, binder_chain_index=0, conf_fields=CONF_FIELDS):
    """
    Pull RF3's confidence numbers out of the summary JSON it wrote as a dict
    RF3 reports plddt on a 0-1 scale so pLDDT is rescaled
    (binder_plddt) is extracted from RF3's chain_ptm which list holds per-chain pLDDT, 
    not pTM. (interface_pae_min) is the best binder-target PAE, from (chain_pair_pae_min)
    (interface_pae) is the mean binder-target PAE, from the off-diagonal of(chain_pair_pae)
    Recorded for context, but not used. (iptm) is already interface-only by construction.
    (plddt) and (pae) are average over the whole complex and currently not used. 
    """
    
    keys = {"iptm": ("iptm",), "plddt": ("overall_plddt", "plddt"), "pae": ("overall_pae", "pae")}
    found = dict.fromkeys(conf_fields)
    # required confidences can all be found in summary_confidences.json
    paths = sorted(Path(out_dir).rglob("*.json"),
                   key=lambda q: (not q.name.endswith("summary_confidences.json"), q))
    for path in paths:
        try:
            data = json.loads(Path(path).read_text())
        except (ValueError, OSError):
            continue
        if not isinstance(data, dict):
            continue
        blocks = [data]
        if isinstance(data.get("summary_confidences"), dict):
            blocks.append(data["summary_confidences"])
        for block in blocks:
            for metric, names in keys.items():
                if found[metric] is not None:
                    continue
                for name in names:
                    value = block.get(name)
                    if isinstance(value, (int, float)):
                        found[metric] = float(value)
                        break
            if found["binder_plddt"] is None:
                chain_plddt = block.get("chain_ptm")
                if (isinstance(chain_plddt, list)
                        and len(chain_plddt) > binder_chain_index
                        and isinstance(chain_plddt[binder_chain_index], (int, float))):
                    found["binder_plddt"] = float(chain_plddt[binder_chain_index])
            if found["interface_pae"] is None:
                found["interface_pae"] = _mean_interface_pae(block.get("chain_pair_pae"))
            if found["interface_pae_min"] is None:
                found["interface_pae_min"] = _mean_interface_pae(
                    block.get("chain_pair_pae_min"))
        if all(v is not None for v in found.values()):
            break
    for key in ("plddt", "binder_plddt"):
        if found[key] is not None and found[key] <= 1.0:
            found[key] *= 100.0
    return found


def _mean_interface_pae(matrix):
    """
    Mean of the **off-diagonal** entries of one of RF3's ``chain_pair_pae*``
    matrices: for a two-chain complex that is the binder-target error, and nothing
    else. None if unavailable.

    RF3 as installed writes ``null`` everywhere but the upper triangle, so skipping
    the diagonal changes no number today; it is skipped because the diagonal, if a
    later version fills it in, holds each chain's error against *itself*, which for
    a 250-residue target is large and near-constant and would flatten the metric.
    """
    if not isinstance(matrix, list):
        return None
    values = [v for i, row in enumerate(matrix) if isinstance(row, list)
              for j, v in enumerate(row)
              if i != j and isinstance(v, (int, float))]
    return sum(values) / len(values) if values else None


def passes_filters(conf, rmsd, min_iptm=0.8, min_plddt=80.0, max_ipae=10.0,
                   max_rmsd=2.0):
    """
    Whether one refolded design passes, given its parse_rf3_confidences dict.
    only checks iptm, binder_plddt and interface_pae_min. 
    """
    thresholds = [(conf.get("iptm"), min_iptm, False),
                  (conf.get("binder_plddt"), min_plddt, False),
                  (conf.get("interface_pae_min"), max_ipae, True),
                  (rmsd, max_rmsd, True),]
    for value, limit, is_maximum in thresholds:
        if value is None:
            return False
        if (value > limit) if is_maximum else (value < limit):
            return False
    return True


def check_binder_chain(backbone, settings, binder_chain=BINDER_CHAIN):
    """
    Confirm the diffused backbone really has the binder on the chain specified in
    binder_chain. Returns None if it does, or a message explaining what is wrong.
    """
    coords, _ = read_ca_coords(backbone, chain=binder_chain)
    if not coords:
        return "no chain {} in {}".format(binder_chain, Path(backbone).name)
    try:
        spec = next(iter(load_settings(settings).values()))
        bounds = contig_binder_lengths(spec.get("contig", ""))
    except Exception:
        bounds = None
    # a binder-design run always has a target alongside the binder
    others = [c for c in ("B", "C", "D") if read_ca_coords(backbone, chain=c)[0]]
    if not others:
        return ("only chain {} in {}; expected the target alongside the binder"
                .format(binder_chain, Path(backbone).name))
    if bounds and not (bounds[0] <= len(coords) <= bounds[1]):
        return ("chain {} of {} has {} residues, outside the contig's designed "
                "range {}-{}; chain {} is probably not the binder"
                .format(binder_chain, Path(backbone).name, len(coords),
                        bounds[0], bounds[1], binder_chain))
    return None


def run_design_job(settings, outdir, job_prefix="job0_", num_designs=8, batch_size=8,
                   mpnn_seqs=4, min_iptm=0.8, min_plddt=80.0, max_ipae=10.0,
                   max_rmsd=2.0, step_scale=3.0, gamma_0=0.2, temperature=0.1,
                   use_shim=None):
    """
    One design round for one hotspot input: diffuse backbones, design sequences on
    each, refold them, and write one metrics row per (backbone, sequence) to
    ``<outdir>/metrics_<job_prefix>.csv``. The controller concatenates those.
    """
    global FOUNDRY_LOG
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = job_prefix.rstrip("_") or "job0"
    FOUNDRY_LOG = outdir / "foundry_{}.log".format(prefix)
    print("  foundry shim: {}".format("on" if set_foundry_shim(use_shim) else "off"),
          flush=True)

    backbones = run_rfd3(settings, outdir / "rfd3", job_prefix, num_designs,
                         batch_size, step_scale=step_scale, gamma_0=gamma_0)
    print("  {} backbones from rfd3".format(len(backbones)), flush=True)

    # Fail here rather than write a CSV full of metrics for the wrong chain.
    if backbones:
        problem = check_binder_chain(backbones[0], settings)
        if problem:
            raise SystemExit(
                "[rfd3-job] {}.\n"
                "  Everything downstream assumes RFD3 puts the designed binder on "
                "chain {}. Check the contig in {}: the designed segment must come "
                "first, before the '/0' and the target.".format(
                    problem, BINDER_CHAIN, settings))

    # Per-job directories: parallel jobs share ``outdir``, and the RF3 batch below
    # folds a whole directory, so it must not be able to see a sibling job's
    # designs -- or race with a sibling still writing them.
    mpnn_dir = outdir / "mpnn" / prefix
    rf3_dir = outdir / "rf3" / prefix

    designs = []
    for backbone in backbones:
        # ".cif.gz" -> Path.stem leaves a trailing ".cif", so strip both suffixes
        bb_name = backbone.name.split(".")[0]
        for design_id, _fasta_seq, conf, design_struct in run_mpnn(
                backbone, mpnn_dir, bb_name, mpnn_seqs, temperature=temperature):
            if not Path(design_struct).is_file():
                print("  [mpnn] {} has no structure, skipping".format(design_id))
                continue
            designs.append((design_id, bb_name, conf, design_struct))
    print("  {} sequences from mpnn".format(len(designs)), flush=True)

    # One rf3 call for the whole job, so the checkpoint loads once rather than
    # once per design.
    if designs:
        run_rf3_batch(mpnn_dir, rf3_dir)

    rows = []
    for design_id, bb_name, conf, design_struct in designs:
        # the mpnn fasta concatenates binder and target; record only the binder
        sequence = read_chain_sequence(design_struct)
        conf_metrics, model = rf3_results(rf3_dir, design_id)
        rmsd = None
        if model is not None:
            coords_des, res_des = read_ca_coords(design_struct)
            coords_mod, res_mod = read_ca_coords(model)
            rmsd = kabsch_rmsd(coords_des, coords_mod)
        row = {"design": design_id,
               "backbone": bb_name,
               "sequence": sequence,
               "mpnn_confidence": conf,
               "rmsd": rmsd,
               "pass": passes_filters(conf_metrics, rmsd, min_iptm=min_iptm,
                                      min_plddt=min_plddt, max_ipae=max_ipae,
                                      max_rmsd=max_rmsd),}
        row.update(conf_metrics)
        rows.append(row)

    metrics_csv = outdir / "metrics_{}.csv".format(prefix)
    with open(metrics_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=METRICS_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    n_pass = sum(1 for r in rows if r["pass"])
    print("  wrote {} ({} attempts, {} passing)".format(metrics_csv, len(rows), n_pass))
    return metrics_csv
