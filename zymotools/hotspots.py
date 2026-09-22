"""
Hotspot patch generation.
Scores surface residues for binder-design suitability, groups them into
spatially connected patches, ranks the patches, and exports them as 
input files for different binder pipelines depending on the type specified. 
"""

import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from MDAnalysis.analysis.dssp import DSSP
from Bio.PDB.kdtrees import KDTree

from .structure import iterate_sasa_trajectory

# different default patch size settings depending on the pipeline used
PATCH_DEFAULTS = {"bindcraft": {"depth": 3, "minsize": 6},
                  "bindcraft2": {"depth": 3, "minsize": 6},
                  "boltzgen": {"depth": 1, "minsize": 2},
                  "rfd3": {"depth": 1, "minsize": 2},}

def compute_secondary_structure(u_full, at_sel, window=1):
    """
    Per-residue secondary-structure score 
    Give preference to stable structures: (H/E -> 1.0, coil -> 0.2).
    the window specifies how many res around each res is used to 
    average the score. window=0: just the reside is used.
    scores of 1 do not get averaged. 
    """
    ss_score = defaultdict(lambda: 0.2, {"H": 1.0, "E": 1.0})
    # calculate dssp for input universe
    dssp = DSSP(u_full).run()
    dssp_data = dssp.results.dssp  # (n_frames, n_residues), chars H/E/-
    sel_resids = at_sel.residues.resids
    n_frames = dssp_data.shape[0]
    # make dict with sec structure score for each residue
    ss_scores = {}
    for i, resid in enumerate(sel_resids):
        if i >= dssp_data.shape[1]:
            break
        if n_frames == 1:
            ss_scores[resid] = ss_score[dssp_data[0, i]]
        else:
            structured_frac = np.mean(
                [(dssp_data[f, i] in ("H", "E")) for f in range(n_frames)])
            ss_scores[resid] = round(max(structured_frac, 0.2), 3)
    ss_score_dict = defaultdict(float, ss_scores)
    # average over set window if window > 0
    if window > 0:
        for resid in sorted(ss_score_dict.keys()):
            #if ss_score_dict[resid] == 1:
            #    continue
            wind_min = max(min(ss_score_dict.keys()), int(resid)-window)
            wind_max = min(int(resid)+window+1, max(ss_score_dict.keys()))
            print(wind_min, wind_max)
            ss_score_dict[resid] = np.round(np.mean([ss_score_dict[i] for i in range(wind_min, wind_max)]),2)
    return ss_score_dict


def score_residues(u_full, at_sel, sasa_cutoff=0.2):
    """
    Rank residues for binder design.
    Score = residue-type weight * SASA bin factor * secondary-structure score,
    for residues that are accessible to a protein-sized probe.
    """
    # Graded preference weights, informed by the BindCraft wiki and the rotamer
    # variety reported in A.D. Scouras 2010.
    res_weights = defaultdict(float, {
        "PHE": 1.0, "TYR": 1.0, "TRP": 1.0,    # preferred: low rotamer variety, hydrophobic
        "LEU": 1.0, "ILE": 1.0, "MET": 0.7,    # Met higher rotamer variety but still preferred
        "VAL": 0.5, "ALA": 0.5, "PRO": 0.5,    # hydrophobic + low rotamer variety, not preferred
        "SER": 0.3, "THR": 0.3, "LYS": -0.5,}) # S/T hydrophilic; K should be avoided
    
    # Relative SASA: used to score residues from most to least accessible
    sasa_rel = iterate_sasa_trajectory(u_full, at_sel, relative=True, probe_radius=1.4)
    # Absolute SASA with a larger probe: is the residue reachable by a protein?
    sasa_prot = iterate_sasa_trajectory(u_full, at_sel, relative=False, probe_radius=3)
    # Secondary structure score: prefer residues with stable structure
    ss_scores = compute_secondary_structure(u_full, at_sel)

    # score each residue
    residue_data = {}
    for residue in at_sel.residues:
        resid = residue.resid
        resname = residue.resname
        rel_sasa = sasa_rel[resid]
        prot_sasa = sasa_prot[resid]
        ss = ss_scores[resid]
        # Hard-filter residues completely inaccessible to a protein partner.
        if prot_sasa == 0 or rel_sasa == 0:
            continue

        # Bin relative SASA into 3 levels (0.3 / 0.6 / 1.0).
        sasa_factor = 0.3
        if rel_sasa > 0.2:
            sasa_factor += 0.3
        if rel_sasa > 0.3:
            sasa_factor += 0.4

        # assign score based on residue type
        resweight = res_weights[resname]
        chain_id = residue.segid if residue.segid else ""
        residue_data[resid] = {
            "chain": chain_id,
            "resname": resname,
            "resid": resid,
            "sasa_rel": round(rel_sasa, 3),
            "sasa_factor": sasa_factor,
            "resweight": resweight,
            "ss": round(ss, 3),
            "score": resweight * sasa_factor * ss if resweight > 0 else resweight,}
    return residue_data


def find_patches(u_full, at_sel, surface_resids, candidate_resids,
                 target_resids=None, max_radius=12, step_radius=6, max_depth=3):
    """
    Find spatially connected patches via a CB-neighbour graph, 
    using breath first search.
    """
    # surface residues are all accesible res, candidate all accessile res with score > 0
    target_resids = list(target_resids) if target_resids else []
    if len(candidate_resids) == 0 or len(surface_resids) == 0:
        print("empty selection")
        return []

    # If targets are given, restrict to surface residues in a broad shell (max_radius).
    if len(target_resids) != 0:
        near_target_sel = "around {} resid {}".format(
            max_radius, " ".join(np.array(target_resids).astype(str)))
        surface_resids_near_target = at_sel.select_atoms(near_target_sel).residues.resids
        print("near target", " ".join(surface_resids_near_target.astype(str)))
        surface_resids_in = list(surface_resids_near_target) + target_resids
    else:
        surface_resids_in = surface_resids
        
    # generate dict of CB (or CA for glycine) coordinates for the patch graph.
    cb_sel = at_sel.select_atoms("name CB or (resname GLY and name CA)")
    cb_by_resid = {atom.resid: atom.position for atom in cb_sel}
    coords = np.array([cb_by_resid[res] for res in surface_resids_in]).astype(np.float64)
    kdt = KDTree(coords, 10)
    
    # create KDtree of all neighboring residues
    neighbors = defaultdict(set)
    for i in range(len(surface_resids_in)):
        for hit in kdt.search(coords[i], step_radius):
            j = hit.index
            if i != j:
                neighbors[i].add(j)
                neighbors[j].add(i)

    # Seeds: use narrow shell (step_radius) around targets, else all candidates.
    if len(target_resids) != 0:
        candidate_resids_in = []
        target_ix = np.where(np.isin(surface_resids_in, target_resids))[0]
        for t_ix in target_ix:
            candidate_resids_in.extend([surface_resids_in[i] for i in neighbors[t_ix]])
        candidate_resids_in = [i for i in candidate_resids_in if i not in target_resids]
        print(candidate_resids_in)
    else:
        candidate_resids_in = candidate_resids

    # go over each seed, extend using BFS (max_depth) times
    candidate_ix = np.where(np.isin(surface_resids_in, candidate_resids_in))[0]
    patches = []
    for start in candidate_ix:
        visited = {start}
        component = []
        queue = [(start, 0)]
        while queue:
            node, depth = queue.pop(0)
            component.append(surface_resids_in[node])
            if max_depth is not None and depth >= max_depth:
                continue
            for nb in neighbors[node]:
                if nb not in visited:
                    visited.add(nb)
                    queue.append((nb, depth + 1))
        patches.append(component)
    # sort patches by length
    patches.sort(key=len, reverse=True)
    return patches


def compute_target_coverage(at_sel, patches, target_resids, sel_radius=8):
    """
    Fraction of target residues each patch covers (CB within sel_radius).
    """
    cb_sel = at_sel.select_atoms("name CB or (resname GLY and name CA)")
    cb_by_resid = {atom.resid: atom.position for atom in cb_sel}
    target_coords = np.array([cb_by_resid[res] for res in target_resids]).astype(np.float64)
    n_targets = len(target_resids)
    fractions = []
    for patch in patches:
        covered = set()
        for resid in patch:
            if resid not in cb_by_resid:
                continue
            pos = cb_by_resid[resid]
            dists = np.linalg.norm(target_coords - pos, axis=1)
            for k in range(n_targets):
                if dists[k] <= sel_radius:
                    covered.add(target_resids[k])
        fractions.append(len(covered) / n_targets if n_targets > 0 else 0.0)
    return fractions


def build_patch_summaries(at_sel, patches, residue_data, target_resids=None,
                          min_patch=3, cover_rad=8):
    """
    Summarise/score patches, dropping ones smaller than (min_patch).
    Lysines are counted in the patch score but excluded from the exported
    hotspot residue list (BindCraft should not target them).
    """
    coverage = None
    if target_resids:
        fracs = compute_target_coverage(at_sel, patches, target_resids, sel_radius=cover_rad)
        coverage = dict(enumerate(fracs))

    patch_summaries = []
    for pid, patch_resids in enumerate(patches):
        patch_scores_vals = [residue_data[r]["score"] for r in residue_data if r in patch_resids]
        tot = np.sum(patch_scores_vals) if patch_scores_vals else 0.0
        if len(patch_resids) < min_patch:
            print("patch too small")
            continue
        hotspot_resids, residue_labels = [], []
        for rid in patch_resids:
            if rid in residue_data:
                if residue_data[rid]["resname"] == "LYS":
                    continue
                hotspot_resids.append(rid)
                residue_labels.append("{}{}".format(residue_data[rid]["resname"], rid))
        summary = {
            "patch_id": pid,
            "residues": hotspot_resids,
            "residue_labels": residue_labels,
            "tot_score": round(tot, 4),
            "patch_size": len(hotspot_resids),}
        if coverage is not None:
            summary["target_coverage"] = round(coverage.get(pid, 0.0), 3)
        patch_summaries.append(summary)
    return patch_summaries


def generate_hotspots(struc, pipeline='bindcraft', template=None, target=None, step=6, depth=None,
                      minsize=None, outdir=".", chain=None):
    """
    End-to-end hotspot generation for one structure.
    Returns (patch_csv, inputs_txt, json_paths) and writes patch_summaries.csv,
    one settings file per patch, and inputs-out.txt (patches in score order),
    in outdir. the depth and minsize parameters can be changed to tune patch size. 
    per-pipeline defaults are used when left unset.
    """
    from .structure import load_universe
    # load in correct function for patch generation depending on pipeline used
    if pipeline == "bindcraft":
        from .bindcraft_io import write_patch_settings
    elif pipeline == "bindcraft2":
        from .bindcraft2_io import write_patch_settings
    elif pipeline == "boltzgen":
        from .boltzgen_io import write_patch_settings
    elif pipeline == "rfd3":
        from .rfd3_io import write_patch_settings
    else:
        raise ValueError("unknown pipeline: {}".format(pipeline))
    # only override the patch size when the caller did not ask for one
    patch_defaults = PATCH_DEFAULTS[pipeline]
    if depth is None:
        depth = patch_defaults["depth"]
    if minsize is None:
        minsize = patch_defaults["minsize"]

    # define neccesary variables
    struc = Path(struc).resolve()
    outdir = Path(outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    u_full, at_sel = load_universe(struc, chain=chain)
    residue_data = score_residues(u_full, at_sel)
    residues_surface = pd.DataFrame(residue_data).T
    residues_candidate = residues_surface[residues_surface.score >= 0]

    # generate patches
    patches = find_patches(
        u_full, at_sel, residues_surface.index, residues_candidate.index,
        target_resids=target, max_radius=8, step_radius=step, max_depth=depth,)
    # score and sort patches
    patch_summaries = build_patch_summaries(
        at_sel, patches, residue_data, target_resids=target, min_patch=minsize,)

    # define output paths and save summaries in csv
    patch_csv = outdir / "patch_summaries.csv"
    inputs_txt = outdir / "inputs-out.txt"
    inputs_paths = []
    if not patch_summaries:
        print("No patches passed the filters; nothing written.")
        return None, None, []
    df = pd.DataFrame(patch_summaries).sort_values(by="tot_score", ascending=False)
    df.to_csv(patch_csv)

    # write patch settings into output files
    base_name = os.path.basename(os.path.splitext(str(struc))[0])
    for ind, patch in enumerate(df.residues.values):
        id_out = "-patch-" + str(ind)
        base_name_out = base_name + id_out
        if template is None:
            out_file = outdir / base_name_out
        else:
            out_file = outdir / (Path(template).stem + id_out)
        path = write_patch_settings(
            patch, out_file, base_name_out, struc, template=template,
            design_root=str(outdir), chain=(chain or "A"),)
        inputs_paths.append(path)

    inputs_txt.write_text("".join(p + "\n" for p in inputs_paths))
    return str(patch_csv), str(inputs_txt), inputs_paths
