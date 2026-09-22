"""
SLURM screen controller for Bindcraft 2 hotspot screening.
after finishing the Bindcraft jobs the results are merged an analyzed
the screen adds a self-resubmitting script in order to rerun independently
until conditions are met. BindCraft 2 is still similar enough to borrow 
most functions from bindcraft_slurm.py.
"""

import os
import subprocess
from pathlib import Path

from .bindcraft2_io import validate_settings
from .bindcraft_slurm import (
    check_if_finished,
    get_progress,
    read_input_txt,
    submit_controller,
)
from .util import MAX_STALL_CYCLES, check_stalled, stall_state_path


def progress_paths(output_path):
    """
    return the (ranked_csv, trajectories_csv) paths for the BindCraft2 output
    """
    output_path = Path(output_path)
    return (output_path / "3_Ranked" / "!_Ranked.csv",
            output_path / "1_Trajectories" / "!_Trajectories.csv")


def submit_sbatch(input_json, slurm_bc2, bindcraft2_dir, nr_jobs, af2_params=None):
    """
    Submit nr_jobs number of BindCraft2 jobs for one input. Returns job IDs
    also pass the BINDCRAFT_HOME variable and BINDCRAFT_AF2_PARAMS when given
    """
    export = "ALL,BINDCRAFT_HOME={}".format(bindcraft2_dir)
    if af2_params:
        export += ",BINDCRAFT_AF2_PARAMS={}".format(af2_params)
    cmd = ["sbatch", "--export={}".format(export), str(slurm_bc2), str(input_json)]
    job_ids = []
    for _ in range(int(nr_jobs)):
        out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
        # Expect "Submitted batch job 12345".
        job_ids.append(int(out.split()[-1]))
    return job_ids


def run_screen(inputs_file, slurm_bc2, bindcraft2_dir, resubmit_cmd, af2_params=None,
               min_ratio=0.01, min_traj=300, nr_jobs=5,
               controller_time="00:05:00", controller_mem="512M", controller_cpus=1):
    """
    Run one controller iteration: 
    go over each .json file in the input txt file,
    check if the file is exists, is valid, and check output of the runs
    submit jobs for the first unfinished json file and reschedule controller. 
    """
    input_jsons = read_input_txt(inputs_file)
    for input_json in input_jsons:
        print("[{}]".format(input_json))
        if not os.path.exists(input_json):
            print("  {} does not exist, skipping".format(input_json))
            continue
        try:
            data = validate_settings(input_json)
            output_path = data["project_folder"]
            max_des = data["number_of_final_designs"]
        except Exception as e:
            print("  failed to parse: {}".format(e))
            continue
        if check_if_finished(*progress_paths(output_path), max_des=max_des,
                             min_ratio=min_ratio, min_traj=min_traj):
            print("  {} has been run sufficiently".format(input_json))
            continue
        # prevent resubmission from looping in case of error
        traj_nr, _ = get_progress(*progress_paths(output_path), verbose=False)
        if check_stalled(output_path, traj_nr):
            print("no new trajectories after {} resubmissions."
                  "fix it, then delete {} and rerun screen to resume.".format(
                      MAX_STALL_CYCLES, stall_state_path(output_path)))
            return True
        print("  submitting {} jobs for {}".format(nr_jobs, input_json))
        jids = submit_sbatch(input_json, slurm_bc2, bindcraft2_dir, nr_jobs, af2_params=af2_params)
        print("  submitted job ids: {}".format(jids))
        submit_controller(jids, resubmit_cmd, time=controller_time,
                          mem=controller_mem, cpus=controller_cpus)
        return False
    print("All inputs are finished; no further jobs scheduled.")
    return True


def status_report(inputs_file, outdir_root=None, budget=None,
                  min_ratio=0.01, min_traj=300):
    """
    Read-only progress for every input. Returns a list of per-input dicts.
    outdir_root is not used, but kept as an option for consistency with other backends.
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
            output_path = data["project_folder"]
            max_des = data["number_of_final_designs"] if budget is None else budget
        except Exception as e:
            row["state"] = "invalid: {}".format(e)
            rows.append(row)
            continue
        final_design_path, trajectory_path = progress_paths(output_path)
        traj, des = get_progress(final_design_path, trajectory_path)
        finished = check_if_finished(final_design_path, trajectory_path, max_des,
                                     min_ratio=min_ratio, min_traj=min_traj, verbose=False)
        row.update({
            "design_path": output_path,
            "trajectories": traj,
            "designs": des,
            "target": max_des,
            "ratio": round(des / traj, 4) if traj else None,
            "state": "finished" if finished else "running",})
        rows.append(row)
    return rows
