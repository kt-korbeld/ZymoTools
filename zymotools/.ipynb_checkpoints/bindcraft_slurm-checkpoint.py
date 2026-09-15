"""
SLURM screen controller for Bindcraft hotspot screening.
after finishing the Bindcraft jobs the results are merged an analyzed
the screen adds a self-resubmitting script in order to rerun independently
until conditions are met. get_progress() checks how far one patch has been run. 
"""


import os
import shlex
import subprocess
from pathlib import Path

from .bindcraft_io import validate_settings


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


def progress_paths(output_path):
    """
    ``(final_designs_csv, trajectories_csv)`` BindCraft writes for one input.
    One definition, so the finish check and the status report cannot disagree
    about where the numbers live.
    """
    output_path = Path(output_path)
    return (output_path / "final_design_stats.csv",
            output_path / "trajectory_stats.csv")


def check_if_finished(output_path, max_des, min_ratio=0.01, min_traj=300, verbose=True):
    """
    Check whether an input has been screened sufficiently based on if:
    min_traj (default: 300) designs exist with less than min_ratio (default: 0.01) sucess rate,
    or a sufficient number of final designs specified in the input .json has been reached.
    uses data from .json input file to get the output directory and sufficient nr of final designs. 
    Returns True if sufficiently run, and False if it requires more runs
    """

    final_design_path, trajectory_path = progress_paths(output_path)
    
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


def submit_sbatch(input_json, slurm_bc, filters, advanced, nr_jobs):
    """
    Submit ``nr_jobs`` BindCraft jobs for one input. Returns job IDs.
    """
    cmd = ["sbatch", str(slurm_bc),
           "--settings", str(input_json),
           "--filters", str(filters),
           "--advanced", str(advanced)]
    job_ids = []
    for _ in range(int(nr_jobs)):
        out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
        # Expect "Submitted batch job 12345".
        job_ids.append(int(out.split()[-1]))
    return job_ids


def submit_controller(jids, resubmit_cmd, time="00:05:00", mem="512M", cpus=1):
    """
    Schedule the next controller iteration after ``jids`` all finish.
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


def read_input_txt(inputs_file):
    """
    Read an inputs list file into a list of settings-JSON paths.
    """
    with open(inputs_file) as f:
        return [line.strip() for line in f if line.strip()]


def run_screen(inputs_file, slurm_bc, filters, advanced, resubmit_cmd,
               min_ratio=0.01, min_traj=300, nr_jobs=5,
               controller_time="00:05:00", controller_mem="512M",
               controller_cpus=1):
    """
    Run one controller iteration: 
    go over each .json file in the input txt file,
    check if the file is exists, is valid, and check output of the runs
    submit jobs for the first unfinished json file and reschedule controller. 
    """
    # iterate over the .json files in input file in given order
    input_jsons = read_input_txt(inputs_file)
    for input_json in input_jsons:
        print("[{}]".format(input_json))
        # skip if .jsion file does not exist
        if not os.path.exists(input_json):
            print("  {} does not exist, skipping".format(input_json))
            continue
        # check if settings in json file are valid
        try:
            data = validate_settings(input_json)
            output_path = data["design_path"]
            max_des = data["number_of_final_designs"]
        except Exception as e:
            print("  failed to parse: {}".format(e))
            continue
        # check if output exists and if the success ratio is high enough
        if check_if_finished(output_path, max_des, min_ratio=min_ratio, min_traj=min_traj):
            print("  {} has been run sufficiently".format(input_json))
            continue
        # submit new jobs if not finished
        print("  submitting {} jobs for {}".format(nr_jobs, input_json))
        jids = submit_sbatch(input_json, slurm_bc, filters, advanced, nr_jobs)
        print("  submitted job ids: {}".format(jids))
        submit_controller(jids, resubmit_cmd, time=controller_time,
                          mem=controller_mem, cpus=controller_cpus)
        return False  # work remains
    print("All inputs are finished; no further jobs scheduled.")
    return True  # everything finished


def status_report(inputs_file, outdir_root=None, budget=None,
                  min_ratio=0.01, min_traj=300):
    """
    Read-only progress for every input. Returns a list of per-input dicts.

    ``outdir_root`` is accepted for signature parity with the other two backends
    and ignored: BindCraft's settings JSON carries its own ``design_path``.
    """
    rows = []
    for input_json in read_input_txt(inputs_file):
        row = {"input": input_json}
        if not os.path.exists(input_json):
            row["state"] = "missing"
            rows.append(row)
            continue
        try:
            data = validate_settings(input_json)
            output_path = data["design_path"]
            max_des = data["number_of_final_designs"] if budget is None else budget
        except Exception as e:
            row["state"] = "invalid: {}".format(e)
            rows.append(row)
            continue
        # unlike the other two backends, BindCraft records its own output path and
        # design target in each settings JSON, so outdir_root is not needed here
        final_design_path, trajectory_path = progress_paths(output_path)
        traj, des = get_progress(final_design_path, trajectory_path)
        finished = check_if_finished(output_path, max_des, min_ratio=min_ratio, min_traj=min_traj,
                                     verbose=False)
        row.update({
            "design_path": output_path,
            "trajectories": traj,
            "designs": des,
            "target": max_des,
            "ratio": round(des / traj, 4) if traj else None,
            "state": "finished" if finished else "running",})
        rows.append(row)
    return rows
