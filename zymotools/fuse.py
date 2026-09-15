"""
Fuse a binder-target complex into a single-chain construct.

A design comes out of every backend as two chains: the designed binder and the
target it was designed against. To express it as one gene it has to become one
chain, with a flexible linker carrying the C-terminus of one segment round to the
N-terminus of the other. This module picks the junction, checks the linker can
actually reach, and builds its conformation with pyDisgro.

**The reach check is the point.** A linker cannot cut through the protein, so the
distance that matters is the shortest path *through solvent*, which
``binder_pipeline.linker`` already computes with a grid and Dijkstra. Over the 97
accepted designs of one real screen, the default 15-residue linker could bridge
54 of them binder-first and 40 target-first (71 in one orientation or the other):
the check is what separates a construct that can exist from one that cannot, and
it is why designs are filtered here rather than all being handed to pyDisgro.
Building a conformation for an unreachable junction does not fail loudly -- the
sampler simply returns no closed conformation after a long run.

Reading order is deliberate:

    load -> pick chains -> measure both junctions -> filter -> merge -> build
"""

import csv
import math
from pathlib import Path

from .linker import residues_for_length
from .structure import (load_atom_records, read_pdb_records,
                        records_to_pdb_lines, records_to_universe)


# default linker: flexible GS linker + TEV-cleavable motif
DEFAULT_LINKER = "GGGSGGGSENLYFQS"
MAX_LINKER = 18
# The chain the fused construct is written as. Everything is renumbered into it.
FUSED_CHAIN = "A"

# Fields of the per-design report. Written even for designs that are rejected,
# so the reason a design did not make it is in the same table as the ones that did.
REPORT_FIELDS = ["design", "binder_chain", "target_chain", "binder_len",
                 "order", "junction", "distance", "min_linker_aa", "linker_aa",
                 "bridgeable", "output", "note"]


def pick_chains(records, binder_chain=None, target_chain=None):
    """
    return (binder_chain, target_chain) for a two-chain complex.
    Different pipelines output different orders of target-binder
    either manually select the right chain, or assume the binder is shorter
    """
    # get lengths of all chains in records
    lengths = {}
    for rec in records:
        lengths.setdefault(rec["chain"], set()).add(rec["resnum"])
    lengths = {c: len(r) for c, r in lengths.items()}
    if binder_chain and target_chain:
        pass
    elif len(lengths) < 2:
        raise ValueError("expected a binder and a target, found chain(s) {}"
                         .format(", ".join(sorted(lengths)) or "none"))
    elif binder_chain:
        others = [c for c in lengths if c != binder_chain]
        if len(others) != 1:
            raise ValueError("more than one chain could be the target ({}); "
                             "name it with --target-chain".format(", ".join(sorted(others))))
        target_chain = others[0]
    elif target_chain:
        others = [c for c in lengths if c != target_chain]
        if len(others) != 1:
            raise ValueError("more than one chain could be the binder ({}); "
                             "name it with --binder-chain".format(", ".join(sorted(others))))
        binder_chain = others[0]
    elif len(lengths) > 2:
        raise ValueError("{} chains present ({}); name the two to fuse with "
                         "--binder-chain/--target-chain"
                         .format(len(lengths), ", ".join(sorted(lengths))))
    else:
        binder_chain = min(lengths, key=lambda c: (lengths[c], c))
        target_chain = [c for c in lengths if c != binder_chain][0]
    for chain in (binder_chain, target_chain):
        if chain not in lengths:
            raise ValueError("chain {} is not in the structure (have: {})"
                             .format(chain, ", ".join(sorted(lengths))))
    return binder_chain, target_chain

def chain_residues(records, chain, fullrecord=False):
    """
    Sorted residue numbers of one chain.
    if fullrecord is set to True, return entire record
    """
    # if fullrecord is set to True, return full records of selected chain
    if fullrecord:
        return sorted([rec for rec in records if rec["chain"] == chain], key=lambda x: x["resnum"])
    # if false, return just the resnum
    else:
        return sorted([rec["resnum"] for rec in records if rec["chain"] == chain])

def count_flexterm(dssp_list):
    """
    count number of coiled residue at the end of a list of dssp assignments 
    """
    count = 0
    for res in dssp_list:
        if res == '-':
            count+=1
        else:
            break
    return count

def confident_dssp(dssp, H_min=4, E_min=3):
    """
    DSSP assignments can contain flukes, where a few coiled residues 
    happen to match the h-bond/backbone requirements. to obtain a 
    confident assignment, turn H/E elements under a certain length 
    to coil. by default this 4 for H (one helical turn) 
    and 3 for E (minimum for a short strand)
    """
    import numpy as np
    dssp_out = []
    prev = ''
    for r in dssp:
        # keep record of previous elements
        prev+=r
        # once record is non-identical, remove last element and save
        if not all([i == prev[0] for i in prev]):
            out = prev[:-1]
            # if length is less than min, save as coil
            if out[0] == 'E' and len(out) < E_min:
                dssp_out.extend(len(out)*'-')
            elif out[0] == 'H' and len(out) < H_min:
                dssp_out.extend(len(out)*'-')
            # else, save normally
            else:
                dssp_out.extend(out)
            # start new record with new non-identical element
            prev = prev[-1]
    return np.array(dssp_out)

def junction_for_order(records, order, binder, target, opt_term=False, smooth_dssp=True):
    """
    return a list of (chain, resnum, seq, chain, resnum, seq) of the two residues 
    a linker must join, depending on if the order is binder-target or target-binder.
    if opt_term = True, the start of the junction is moved to the first non-coiled 
    res on the selected N- and C- termini, and the seq reports the excluded termini 
    """
    # if incl_term not enabled, simply return start and end of chain
    first, second = (binder, target) if order == "binder-target" else (target, binder)
    res_first = chain_residues(records, first)[-1]
    res_second = chain_residues(records, second)[0]
    if not opt_term:
        return [(first, res_first, '', second, res_second, '')]
    # if enabled, check if termini are flexible by calculating secondary structure 
    import MDAnalysis
    from MDAnalysis.analysis.dssp import DSSP 
    u_first = records_to_universe(chain_residues(records, first, fullrecord=True))
    u_second = records_to_universe(chain_residues(records, second, fullrecord=True))
    # since mmcif/pdb, assume only one frame, hence just take [0]
    dssp_first = DSSP(u_first).run().results.dssp[0]
    dssp_second = DSSP(u_second).run().results.dssp[0]
    # if set to true, smooth dssp assignment
    if smooth_dssp:
        dssp_first = confident_dssp(dssp_first)
        dssp_second = confident_dssp(dssp_second)
    # calculate how many of the N and C termini should be included
    nflex_cterm = count_flexterm(dssp_first) # reverse [::-1]
    nflex_nterm = count_flexterm(dssp_second)
    # save every possible permutation
    list_out = []
    for nc in range(nflex_cterm+1):
        for nn in range(nflex_nterm+1):
            # get the correct residue
            res_first_temp = res_first - nc
            res_second_temp = res_second + nn
            # find termini
            seq_first, seq_second = '', ''
            if nc > 0:
                seq_first = str(u_first.residues.sequence().seq[-nc:])
            if nn > 0:
                seq_second = str(u_second.residues.sequence().seq[:nn])
            list_out.append((first, res_first_temp, seq_first, second, res_second_temp, seq_second))
    return list_out

def solvent_distance(struc_in, chain_a, res_a, chain_b, res_b,
                     gridstep=1, padding=4, rad=3):
    """
    Shortest solvent path between two residues' CA atoms.
    returns None when no path exists.
    Accept either a PDB or a pre-loaded MDA universe
    """
    import warnings
    import MDAnalysis
    from .linker import shortest_linker_path
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        # accept either path to pdb or MDAnalysis universe
        if type(struc_in) == MDAnalysis.core.universe.Universe:
            u_in = struc_in
        else:
            u_in = MDAnalysis.Universe(str(struc_in))
        # calculate distance across selection
        try:
            distance, _ = shortest_linker_path(
                u_in,
                "chainID {} and resid {}".format(chain_a, res_a),
                "chainID {} and resid {}".format(chain_b, res_b),
                gridstep=gridstep, padding=padding, rad=rad)
        except (ValueError, IndexError):
            # no route out of the protein for one of the two termini
            distance = None
        return distance

def merge_to_single_chain(records, c1, r1, c2, r2, linker_len,
                          chain=FUSED_CHAIN):
    """
    Create one chain from two, renumbered from 1 with a numbered gap for the linker.
    Returns (records, start_anchor, end_anchor). The anchors are the residue
    numbers either side of the gap. pyDisgro grows the linker between them, and
    needs them in the same chain with (linker_len) free numbers between.
    """
    merged, number = [], 0
    start_anchor = None
    for chain_id, res in zip([c1, c2], [r1, r2]):
        renumbered = {}
        for rec in records:
            if rec["chain"] != chain_id:
                continue
            if chain_id == c1 and rec["resnum"] > r1:
                continue
            if chain_id == c2 and rec["resnum"] < r2:
                continue
            if rec["resnum"] not in renumbered:
                number += 1
                renumbered[rec["resnum"]] = number
            new = dict(rec)
            new["chain"] = chain
            new["resnum"] = renumbered[rec["resnum"]]
            merged.append(new)
        if start_anchor is None:
            start_anchor = number
            number += linker_len          # leave the linker its residue numbers
    return merged, start_anchor, start_anchor + linker_len + 1        

def build_linker_conformation(pdb_lines, start, end, linker, chain=FUSED_CHAIN,
                              num_conf=5000, keep=1, sidechains=True,
                              num_sc_states=5, seed=None, verbose=False):
    """
    Grow the linker with pyDisgro's sequential Monte Carlo sampler.
    Returns a list of Structure objects, best-scoring first, which is empty
    when no conformation closed.
    """
    import pydisgro
    from pydisgro.geom import seed as disgro_seed
    from pydisgro.smc import SMC
    from pydisgro.structure import Structure, blank_loop, resnum_to_index

    pydisgro.init_parameters(sidechains=sidechains)
    if seed is not None:
        disgro_seed(seed)
    # placeholder atoms tell DisGro which residues to build
    blanked = blank_loop(pdb_lines, start, end, linker, chain)
    conf = Structure.readPdb(blanked)
    conf._ProtName = "fused"
    smc = SMC(conf, resnum_to_index(conf, start, chain),
              resnum_to_index(conf, end, chain),
              num_conf=num_conf, confkeep=keep, sample_sc=sidechains,
              num_sc_states=num_sc_states, verbose=verbose)
    return [smc.to_structure(result) for result in smc.run()[:keep]]


def fuse_design(path, outdir, linker="X", order="binder-target", report_fields=REPORT_FIELDS,
                max_linker=MAX_LINKER, binder_chain=None, target_chain=None, 
                check_only=False, opt_term=False, replace_term=False,
                num_conf=5000, keep=1, sidechains=True, seed=None,
                gridstep=1, padding=4, rad=3, verbose=False, buffer=5):
    """
    Fuse one design. Returns a report row, rows are based on REPORT_FIELDS.
    A design that the linker cannot bridge is reported and skipped, not built.
    """
    # raise error if max linker length is exceeded
    if len(linker) > max_linker:
        raise ValueError("Disgro can only handle linkers up to 18 residues."
                         " current linker exceeds limit: {}".format(linker))  
    # load design from path and create row with output fields
    path = Path(path)
    row = dict.fromkeys(report_fields)
    row["design"] = path.name
    row["linker_aa"] = len(linker)
    row["linker"] = linker
    row["bridgeable"] = False
    
    # load atom record from PDB or CIF files
    records = load_atom_records(path)
    if not records:
        row["note"] = "no atoms could be read"
        return row
    binder, target = pick_chains(records, binder_chain, target_chain)
    row["binder_chain"], row["target_chain"] = binder, target
    row["binder_len"] = len(chain_residues(records, binder))
        
    # parse into MDAnalysis
    u_in = records_to_universe(records)
    # if order == 'auto', check best order of binder-target. 
    orders = ["binder-target", "target-binder"] if order == "auto" else [order]
    measured = []
    for candidate in orders:
        # find residues to act as anchor of linker
        anchorlist = junction_for_order(records, candidate, binder, target, opt_term=(opt_term or replace_term))
        measured_temp = []
        for c1, r1, s1, c2, r2, s2 in anchorlist:
            # if replace_term is set, the flexible termini are replaced by the linker 
            if replace_term:
                linker_out = linker
            else:
                linker_out = s1+linker+s2
            # cap to max disgro linker length
            if len(linker_out) > max_linker:
                continue
            # calculate distance between linkers
            print(c1, r1, c2, r2)
            distance = solvent_distance(u_in, c1, r1, c2, r2, gridstep=gridstep, padding=padding, rad=rad)
            # calculate required nr of residues of linker
            min_aa, min_aa_buf = residues_for_length(distance, buffer=buffer)
            measured.append((candidate, c1, r1, c2, r2, linker_out, distance, min_aa_buf))
            # if verbose, print if path is impossible
            if verbose:
                print("    {:<14} {}{} -> {}{}: {} Å".format(candidate, c1, r1, c2, r2,
                    "unreachable" if distance is None or not math.isfinite(distance)
                    else round(distance, 1)))

    # rank by ratio of linker res / distance needed
    def rank(item, verbose=verbose):
        link = len(item[-3])
        needed = item[-1]
        if not math.isfinite(link) or not math.isfinite(needed):
            return False, 0
        if verbose:
            print(item[0:5], 'linker length:', link, 'needed:', needed, 'ratio:', link/needed)
        return (needed is None, link/needed if needed is not None else 0)

    # get best linker position, extract data
    best = sorted(measured, key=rank, reverse=True)[0]
    c1, r1, c2, r2 = best[1:5]
    linker_full, distance, needed = best[5:]
    # save data in row
    row["order"] = best[0] 
    row["junction"] = "{}{}->{}{}".format(*best[1:5])
    row["distance"] = None if not math.isfinite(distance) else round(distance, 1)
    row["min_linker_aa"] = needed
    row["linker_built"] = linker_full

    if needed is None:
        row["note"] = "no solvent path between the termini"
        return row
    if needed > len(linker_full):
        row["note"] = ("needs {} residues to span {} A, linker(+termini) is {}"
                       .format(needed, row["distance"], len(linker_full)))
        return row
    row["bridgeable"] = True
    if check_only:
        row["note"] = "bridgeable; not built (--check-only)"
        return row

    merged, start, end = merge_to_single_chain(records, c1,r1,c2,r2, len(linker_full))
    lines = records_to_pdb_lines(merged)
    built = build_linker_conformation(lines, start, end, linker_full,
                                      num_conf=num_conf, keep=keep,
                                      sidechains=sidechains, seed=seed,
                                      verbose=verbose)
    # report if no succesful conformations were 
    if not built:
        row["note"] = ("pyDisgro closed no conformation in {} attempts"
                       .format(num_conf))
        return row
    # write out successful conformations
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written = []
    for i, structure in enumerate(built):
        out = outdir / "{}_fused{}.pdb".format(
            path.name.split(".")[0], "" if i == 0 else "_{}".format(i + 1))
        structure.writePdb(str(out), 1, structure.numRes)
        # pyDisgro writes a blank chain ID; give the construct a real one so the
        # result is a chain that other tools will select on
        stamped = read_pdb_records(out)
        for rec in stamped:
            rec["chain"] = FUSED_CHAIN
        out.write_text("".join(records_to_pdb_lines(stamped)) + "END\n")
        written.append(out)
    row["output"] = str(written[0])
    row["note"] = "built {} conformation(s)".format(len(written))
    return row


def fuse_designs(paths, outdir, report=None, **kwargs):
    """
    Fuse every design in a list of paths, writing a report CSV. 
    Returns the rows with metrics and success/failure.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in paths:
        print("[{}]".format(Path(path).name), flush=True)
        try:
            row = fuse_design(path, outdir, **kwargs)
        except Exception as exc:
            row = dict.fromkeys(REPORT_FIELDS)
            row.update({"design": Path(path).name, "bridgeable": False,
                        "note": "failed: {}".format(exc)})
        rows.append(row)
        print("  {}".format(row["note"] or "ok"), flush=True)
    report = Path(report) if report else outdir / "fusion_report.csv"
    with open(report, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=REPORT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return rows, report


def collect_designs(struc):
    """
    collect the design structure files in input dir struc. 
    """
    struc = Path(struc)
    if struc.is_dir():
        return sorted(p for p in struc.iterdir()
                      if p.name.endswith((".pdb", ".cif", ".cif.gz", ".pdb.gz")))
    return [struc]
