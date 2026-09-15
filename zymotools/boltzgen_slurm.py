"""
SLURM screen controller for BoltzGen hotspot screening.
after finishing the boltzgen jobs the results are merged an analyzed
the screen adds a self-resubmitting script in order to rerun independently
until conditions are met. get_progress() checks how far one patch has been run. 
"""

import os
import shlex
import subprocess
from pathlib import Path

from .boltzgen_io import validate_settings

def count_rows(csv_path):
    """
    Number of data rows in a CSV (excluding the header); 0 if missing.
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        return 0
    with csv_path.open() as fh:
        n = sum(1 for _ in fh)
    return max(0, n - 1)


def get_progress(final_design_path, trajectory_path, verbose=False):
    """
    From the data loaded in from an input .json file, use the output path 
    to get the number of trajectories and number of successful designs.
    if verbose is set to True, print number of trajectories, designs, and ratio
    """
    # get number of trajectories and number of successful designs, print nrs if verbose
    traj = count_rows(trajectory_path)
    des = count_rows(final_design_path)
    if verbose:
        ratio_str = str(round(des / traj, 3)) if traj else "n/a"
        print("  trajectories: {}, designs: {}, ratio: {}".format(
            traj, des, ratio_str))
    return traj, des


def progress_paths(output_path, max_des):
    """
    ``(final_designs_csv, all_designs_csv)`` BoltzGen writes for one input.
    One definition, so the finish check and the status report cannot disagree
    about where the numbers live.
    """
    ranked = Path(output_path) / "final_ranked_designs"
    return (ranked / "final_designs_metrics_{}.csv".format(max_des),
            ranked / "all_designs_metrics.csv")


def outdir_for_input(input_yaml, outdir_root):
    """
    Output directory for one hotspot input.

    Each input needs its *own* directory: the finish check counts rows in that
    directory, so pooling every hotspot into one lets the designs of the first
    hotspot mark all the others finished before they have run at all.
    """
    return Path(outdir_root) / Path(input_yaml).stem


def check_if_finished(output_path, max_des, min_ratio=0.01, min_traj=300, verbose=True):
    """
    Check whether an input has been screened sufficiently based on if:
    min_traj (default: 300) designs exist with less than min_ratio (default: 0.01) sucess rate,
    or a sufficient number of final designs specified in the input .json has been reached.
    uses data from .json input file to get the output directory and sufficient nr of final designs. 
    Returns True if sufficiently run, and False if it requires more runs
    """
    # get paths of neccesary output files
    final_design_path, trajectory_path = progress_paths(output_path, max_des)
    # if nothing has run yet, it is not finished, re-submit.
    if not (os.path.exists(final_design_path) and os.path.exists(trajectory_path)):
        return False      
    # get number of trajectories and number of successful designs, print data if verbose
    trajectory_nr, design_nr = get_progress(final_design_path, trajectory_path, verbose=verbose)
    # Return True if sufficient number of designs or ratio too bad, otherwise return False
    if design_nr >= max_des:
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


def check_jobdir(output_path):
    """
    for a given output path, find all the directories for parallel jobs 
    named after the output path but with "-job-N" appended
    """
    output_path = Path(output_path)
    out_jobs = []
    # return empty if output_path does not exist
    if not output_path.parent.is_dir():
        return out_jobs
    # for each entry in output_path, check for -job- and if dir
    for entry in os.listdir(output_path.parent):
        if entry.startswith(output_path.name + "-job-"):
            output_path_job = output_path.parent / entry
            if os.path.isdir(output_path_job):
                out_jobs.append(output_path_job)
    return out_jobs


def boltzgen_merge(activate_cmd, dirs, output):
    """
    Runs a subprocess that calls the boltzgen merge command,
    and merges the directories from parallel jobs.
    """
    if len(dirs) == 0:
        return 1
    dirstring = " ".join([str(i) for i in dirs])
    print('merging: ', dirstring)
    # run command boltzgen merge after activating boltzgen venv/conda
    cmd = f"{activate_cmd} && boltzgen merge {dirstring} --output {output}"
    subprocess.run(cmd, shell=True, executable="/bin/bash", check=True)
    return 1

def update_sbatch(slurm, yaml, output):
    """
    updates a given BoltzGen jobscript with a given yaml and output dir
    Both substitutions are anchored on the flag
    """
    import re
    with open(slurm, 'r') as f:
        slurm_new = f.read()
    # replace --output <output>
    slurm_new = re.sub(r'(--output\s+)(\S+)',
                       lambda m: m.group(1) + str(output), slurm_new)
    # replace the input yaml, which boltzgen takes positionally after "run"
    slurm_new = re.sub(r'(boltzgen\s+run\s+)(\S+)',
                       lambda m: m.group(1) + str(yaml), slurm_new)
    return slurm_new


def submit_sbatch(input_yaml, slurm_bg, outdir, nr_jobs):
    """
    Submit (nr_jobs) BindCraft jobs for one input. Returns job IDs.
    """
    job_ids = []
    for ind in range(int(nr_jobs)):
        # create new outdir and .yaml for each run
        outdir_job = Path(outdir).parent / (Path(outdir).name+f"-job-{ind}")
        outdir_job.mkdir(parents=True, exist_ok=True)
        slurm_out = outdir_job / (os.path.splitext(Path(input_yaml).name)[0]+'.sh')
        slurm_out.write_text(update_sbatch(slurm_bg, Path(input_yaml).resolve(), Path(outdir_job).resolve()))
        print(slurm_out)
        cmd = ["sbatch", str(slurm_out)]
        print(cmd)
        out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
        # Expect "Submitted batch job 12345".
        job_ids.append(int(out.split()[-1]))
    return job_ids


def submit_controller(jids, resubmit_cmd, time="00:05:00", mem="512M", cpus=1):
    """
    Schedule the next controller iteration after jids all finish.
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


def run_screen(inputs_file, slurm_bg, output_path, resubmit_cmd, boltzgen_cmd,
               min_ratio=0.01, min_traj=300, nr_jobs=5, max_des=300,
               controller_time="00:05:00", controller_mem="512M",
               controller_cpus=1):
    """
    Run one controller iteration: 
    go over each .json file in the input txt file,
    check if the file is exists, is valid, and check output of the runs
    submit jobs for the first unfinished json file and reschedule controller. 
    """
    # iterate over the .json files in input file in given order
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
        # combine results from separate jobs using boltzgen merge
        outdir = outdir_for_input(input_yaml, output_path)
        print('output path:', outdir)
        out_jobs = check_jobdir(outdir)
        boltzgen_merge(boltzgen_cmd, out_jobs, outdir)
        # check if output exists and if the success ratio is high enough
        if check_if_finished(outdir, max_des, min_ratio=min_ratio, min_traj=min_traj):
            print("  {} has been run sufficiently".format(input_yaml))
            continue
        # submit new jobs if not finished
        print("  submitting {} jobs for {}".format(nr_jobs, input_yaml))
        jids = submit_sbatch(input_yaml, slurm_bg, outdir, nr_jobs)
        print("  submitted job ids: {}".format(jids))
        submit_controller(jids, resubmit_cmd, time=controller_time,
                          mem=controller_mem, cpus=controller_cpus)
        return False  # work remains
    print("All inputs are finished; no further jobs scheduled.")
    return True  # everything finished


def status_report(inputs_file, outdir_root, budget=None, min_ratio=0.01, min_traj=300):
    """
    Read-only progress for every input. Returns a list of per-input dicts.
    outdir_root and budget are screen-level settings rather than anything
    the input YAML carries, so unlike BindCraft they have to be passed in.
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
        final_design_path, trajectory_path = progress_paths(outdir, budget)
        traj, des = get_progress(final_design_path, trajectory_path)
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
