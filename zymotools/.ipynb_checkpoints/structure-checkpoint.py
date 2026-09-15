"""
some general functions for dealing with structure parsing.
MDAnalysis is used to parse the structure for most inputs, mainly for the patch
generation and the linker calculations. The RFD3 pipeline generates .mmcif files 
which are not compatible with MDAnalysis, so some additional code is provided for
"""

# --------------------------------------------------------------------------
# code for MDAnalysis
# --------------------------------------------------------------------------

def load_universe(topology, trajectory=None, chain=None, sel="protein"):
    """
    Load an MDAnalysis Universe and return 
    the full universe and a specified selection.
    """
    import MDAnalysis
    if trajectory:
        u_full = MDAnalysis.Universe(str(topology), str(trajectory))
    else:
        u_full = MDAnalysis.Universe(str(topology))
    sel_full = sel
    # only load in one of the chains if specified
    if chain:
        sel_full += " and (segid {0} or chainID {0})".format(chain)
    at_sel = u_full.select_atoms(sel_full)
    # raise error if no atoms are found within selection
    if len(at_sel) == 0:
        raise ValueError("No atoms found with selection '{}'. "
            "Check your chain ID or file format.".format(sel_full))
    return u_full, at_sel


def get_target_binder_sel(u_in, target_u):
    """
    Split a target+binder complex into a target_sel and binder_sel.
    Given a universe with a target and binder, and a universe containing only
    the target, return the atom selections for the target and binder chains by
    matching chain sequences against the target sequence.
    """
    import numpy as np
    import MDAnalysis
    if str(target_u.residues.sequence().seq) not in str(u_in.residues.sequence().seq):
        raise ValueError("target not present in input structure")
    target_id, binder_id = [], []
    chainids = list(set(np.hstack(u_in.segments.chainIDs)))
    for chainid in chainids:
        chainsel = u_in.select_atoms("chainID {}".format(chainid))
        if str(chainsel.residues.sequence().seq) in str(target_u.residues.sequence().seq):
            target_id.append(chainid)
        else:
            binder_id.append(chainid)
    target_sel = u_in.select_atoms("chainid {}".format(" ".join(target_id)))
    binder_sel = u_in.select_atoms("chainid {}".format(" ".join(binder_id)))
    return target_sel, binder_sel


def unique_chain_ids(u_in):
    """
    Return the sorted set of chain IDs present in a universe.
    """
    import MDAnalysis
    return sorted(set().union(*[set(i) for i in u_in.segments.chainIDs]))


def suppress_empty_universe_warning():
    """
    Context-manager-free helper to silence MDAnalysis empty-universe noise.
    """
    import warnings
    import MDAnalysis
    warnings.filterwarnings("ignore", message="there is no reference attributes")

# --------------------------------------------------------------------------
# compute SASA using MDAnalysis universe
# --------------------------------------------------------------------------

def compute_sphere(n_points=100):
    """
    Return n_points of 3D coordinates 'evenly' placed on a unit sphere.
    Uses the golden-spiral algorithm. Computed once, then translated to each
    atom as SASA is evaluated.
    """
    import math
    import numpy as np
    n = n_points
    dl = np.pi * (3 - 5 ** 0.5)
    dz = 2.0 / n
    longitude = 0
    z = 1 - dz / 2
    coords = np.zeros((n, 3), dtype=np.float32)
    for k in range(n):
        r = (1 - z * z) ** 0.5
        coords[k, 0] = math.cos(longitude) * r
        coords[k, 1] = math.sin(longitude) * r
        coords[k, 2] = z
        z -= dz
        longitude += dl
    return coords


def compute_shrakerupley(atomgroup, probe_radius=1.40, n_points=100,
                         radii_dict=None, level="A"):
    """
    Per-atom accessible surface area for an AtomGroup.
    """
    import numpy as np
    from Bio.PDB.kdtrees import KDTree

    # van der Waals radii per element (Angstrom).
    radii_atomtypes = {"H": 1.20, "HE": 1.40, "C": 1.70, "N": 1.55, "O": 1.52, 
                       "F": 1.47, "NA": 2.27, "MG": 1.73, "P": 1.80, "S": 1.80, 
                       "CL": 1.75, "K": 2.75, "CA": 2.31, "NI": 1.63, "CU": 1.40, 
                       "ZN": 1.39, "SE": 1.90, "BR": 1.85, "CD": 1.58, "I": 1.98, "HG": 1.55,}
    # set radii_atomtypes as default if no specific dict is provided
    if radii_dict is None:
        radii_dict = radii_atomtypes
    # compute sphere
    sphere = compute_sphere(n_points)
    atoms = atomgroup.atoms
    n_atoms = len(atoms)
    if not n_atoms:
        raise ValueError("Entity has no child atoms.")
    # construct KDTtree from coodinates
    coords = atomgroup.atoms.positions.astype(np.float64)
    kdt = KDTree(coords, 10)
    radii = np.array([radii_dict[a.element] for a in atoms], dtype=np.float64)
    radii += probe_radius
    twice_maxradii = np.max(radii) * 2
    asa_array = np.zeros((n_atoms, 1), dtype=np.int64)
    ptset = set(range(n_points))
    for i in range(n_atoms):
        r_i = radii[i]
        s_on_i = (np.array(sphere, copy=True) * r_i) + coords[i]
        available_set = ptset.copy()
        kdt_sphere = KDTree(s_on_i, 10)
        for jj in kdt.search(coords[i], twice_maxradii):
            j = jj.index
            if i == j:
                continue
            if jj.radius < (r_i + radii[j]):
                available_set -= {
                    pt.index for pt in kdt_sphere.search(coords[j], radii[j])
                }
        asa_array[i] = len(available_set)
    f = radii * radii * (4 * np.pi / n_points)
    asa_array = asa_array * f[:, np.newaxis]
    return asa_array.flatten()


def compute_sasa(atomgroup, probe_radius=1.4, n_points=100, relative=True):
    """
    Per-residue (relative or absolute) SASA for the current frame.
    """
    import MDAnalysis
    import numpy as np
    # Max sasa per res type in a Gly-X-Gly tripeptide
    # from Tien et al. 2013, used to normalise raw SASA to relative SASA.
    max_sasa = {"ALA": 129, "ARG": 274, "ASN": 195, "ASP": 193, "CYS": 167,
                "GLN": 225, "GLU": 223, "GLY": 104, "HIS": 224, "ILE": 197,
                "LEU": 201, "LYS": 236, "MET": 224, "PHE": 240, "PRO": 159,
                "SER": 155, "THR": 172, "TRP": 285, "TYR": 263, "VAL": 174,}
    # compute sasa values
    atom_sasa = compute_shrakerupley(atomgroup, probe_radius=probe_radius, n_points=n_points)
    residue_sasa = {}
    # sum up per residue and scale into rSASA if relative=True
    for residue in atomgroup.residues:
        resname = residue.resname
        if resname not in max_sasa:
            continue
        atom_indices = residue.atoms.ix - atomgroup.atoms.ix[0]
        raw_sasa = atom_sasa[atom_indices].sum()
        if relative:
            residue_sasa[residue.resid] = raw_sasa / max_sasa[resname]
        else:
            residue_sasa[residue.resid] = raw_sasa
    return residue_sasa


def iterate_sasa_trajectory(u_full, at_sel, relative=True, probe_radius=1.4):
    """
    Average per-residue SASA across all trajectory frames.
    """
    import MDAnalysis
    from collections import defaultdict
    sasa_accum = defaultdict(float)
    n_frames = u_full.trajectory.n_frames
    for _ in u_full.trajectory:
        frame_sasa = compute_sasa(at_sel, relative=relative, probe_radius=probe_radius)
        for resid, val in frame_sasa.items():
            sasa_accum[resid] += val
    return defaultdict(float, {resid: val / n_frames for resid, val in sasa_accum.items()})

# --------------------------------------------------------------------------
# code for parsing CIF files
# --------------------------------------------------------------------------

def open_gz(path):
    """
    mmcif files are sometimes outputed as gz by pipeline,
    small function to open file regardless of whether it is .gz or not
    """
    import gzip
    path = str(path)
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "r")

def read_atom_records(path, chain=None, atom=None):
    """
    Every atom of an mmCIF as a list of dicts with the keys
    ``atom, resname, chain, resnum, x, y, z, element``.
    Optionally filtered to one chain and/or one atom name.
    """
    # read in data from cif
    with open_gz(path) as f:
        lines = f.readlines()
    # initialize variables
    records = []
    in_loop = False
    header, cols = [], None
    # go over each line in cif
    for line in lines:
        line = line.rstrip("\n")
        # start saving  lines following the first _atom_site. marker
        if line.startswith("_atom_site."):
            in_loop, cols = True, None
            header.append(line.strip().split(".", 1)[1])
            continue
        if not in_loop:
            continue
        # if not atom record, break if header already exists, else continue
        if not line.strip() or line.startswith("#") or line.startswith("_"):
            if header:
                break
            continue
        # column names are specified by entries in header, create map
        if cols is None:
            cols = {name: i for i, name in enumerate(header)}
        fields = line.split()
        # skip lines that dont correspond to header
        if len(fields) < len(header):
            continue
        # extract information from line
        atom_key = cols.get("label_atom_id", cols.get("auth_atom_id"))
        chain_key = cols.get("auth_asym_id", cols.get("label_asym_id"))
        seq_key = cols.get("auth_seq_id", cols.get("label_seq_id"))
        name_key = cols.get("label_comp_id", cols.get("auth_comp_id"))
        # skip if information is missing, or not the requested chain/atom
        if None in [atom_key, chain_key, seq_key, name_key]:
            break
        atom_id = fields[atom_key].strip('"')
        if atom is not None and atom_id != atom:
            continue
        if chain is not None and fields[chain_key] != chain:
            continue
        # try obtaining coordinates, skip if missing
        try:
            xyz = (float(fields[cols["Cartn_x"]]),
                   float(fields[cols["Cartn_y"]]),
                   float(fields[cols["Cartn_z"]]))
        except (ValueError, KeyError):
            continue
        # RF3 and MPNN write unplaced atoms as NaN rather, so include check
        if not all(v == v for v in xyz):
            continue
        records.append({
            "atom": atom_id,
            "resname": fields[name_key],
            "chain": fields[chain_key],
            "resnum": int(fields[seq_key]),
            "x": xyz[0], "y": xyz[1], "z": xyz[2],
            "element": (fields[cols["type_symbol"]]
                        if "type_symbol" in cols else atom_id[:1]),})
    return records


def read_ca_coords(path, chain="A"):
    """
    extract just the CA coordinates of sepcified chain from an mmCIF 
    as an ordered dict of {resnum: (x, y, z)}.
    """
    coords, residues = {}, {}
    for rec in read_atom_records(path, chain=chain, atom="CA"):
        coords[rec["resnum"]] = (rec["x"], rec["y"], rec["z"])
        residues[rec["resnum"]] = rec["resname"]
    return coords, residues

def read_chain_sequence(path, chain="A"):
    """
    One-letter sequence of specified chain in mmcif, based on the CA atoms.
    MPNN reports the whole complex, so sequence from output cif is used
    """
    three_to_one = {"ALA":"A", "ARG":"R", "ASN":"N", "ASP":"D", "CYS":"C", 
                    "GLN":"Q", "GLU":"E", "GLY":"G", "HIS":"H", "ILE":"I", 
                    "LEU":"L", "LYS":"K", "MET":"M", "PHE":"F", "PRO":"P",
                    "SER":"S", "THR":"T", "TRP":"W", "TYR":"Y", "VAL":"V",}
    coords, residues = read_ca_coords(path, chain=chain)
    # Anything unknown becomes X
    return "".join(three_to_one.get(name, "X") for _, name in sorted(residues.items()))


def kabsch_rmsd(coords_a, coords_b):
    """
    simple function for calculating rmsd between two coordinate sets,
    after doing a simple kabsch alignment. 
    """
    import numpy as np
    shared = sorted(set(coords_a) & set(coords_b))
    if len(shared) < 3:
        # some predictors renumber; fall back to a positional match
        if len(coords_a) != len(coords_b) or len(coords_a) < 3:
            return None
        P = np.array([coords_a[k] for k in sorted(coords_a)], dtype=float)
        Q = np.array([coords_b[k] for k in sorted(coords_b)], dtype=float)
    else:
        P = np.array([coords_a[k] for k in shared], dtype=float)
        Q = np.array([coords_b[k] for k in shared], dtype=float)
    P = P - P.mean(axis=0)
    Q = Q - Q.mean(axis=0)
    V, _, Wt = np.linalg.svd(P.T @ Q)
    d = np.sign(np.linalg.det(V @ Wt))
    U = V @ np.diag([1.0, 1.0, d]) @ Wt
    P = P @ U
    return float(np.sqrt(((P - Q) ** 2).sum() / len(P)))


# --------------------------------------------------------------------------
# code for parsing PDB independently from MDAnalysis
# --------------------------------------------------------------------------

def format_atom_name(atom, element=""):
    """
    Place an atom name in the PDB's four-column atom field.
    """
    atom = str(atom).strip()
    if len(atom) >= 4 or len(str(element).strip()) == 2:
        return atom[:4]
    return " " + atom

def records_to_pdb_lines(records, serial_start=1):
    """
    Turn read_atom_records dicts into PDB ATOM lines.
    """
    # template for an ATOM record
    pdb_atom = ("ATOM  {serial:>5} {atom:<4}{altloc:1}{resname:>3} {chain:1}"
            "{resnum:>4}{icode:1}   {x:8.3f}{y:8.3f}{z:8.3f}"
            "{occ:6.2f}{bfac:6.2f}          {element:>2}\n")
    lines = []
    # for every entry in record, write PDB line
    for i, rec in enumerate(records):
        element = rec.get("element", "") or rec["atom"][:1]
        lines.append(pdb_atom.format(
            serial=(serial_start + i) % 100000,
            atom=format_atom_name(rec["atom"], element),
            altloc=" ",
            resname=str(rec["resname"])[:3],
            chain=str(rec["chain"])[:1],
            resnum=rec["resnum"],
            icode=" ",
            x=rec["x"], y=rec["y"], z=rec["z"],
            occ=rec.get("occupancy", 1.0), bfac=rec.get("bfactor", 0.0),
            element=str(element)[:2],))
    return lines

def read_pdb_records(path):
    """
    read_atom_records-shaped dicts from a PDB file, read by column.
    """
    records = []
    with open_gz(path) as fh:
        for line in fh:
            if not line.startswith("ATOM  "):
                continue
            records.append({
                "atom": line[12:16].strip(),
                "resname": line[17:20].strip(),
                "chain": line[21],
                "resnum": int(line[22:26]),
                "x": float(line[30:38]), "y": float(line[38:46]),
                "z": float(line[46:54]),
                "occupancy": float(line[54:60] or 1.0),
                "bfactor": float(line[60:66] or 0.0),
                "element": line[76:78].strip() or line[12:16].strip()[:1],})
    return records


def load_atom_records(path):
    """
    Atom records from either a PDB or an (gzipped) mmCIF
    to deal with differing pipeline outputs across RFD3/Bindcraft/Boltzgen
    """
    name = str(path)
    if name.endswith(".gz"):
        name = name[:-3]
    if name.lower().endswith((".pdb", ".ent")):
        return read_pdb_records(path)
    return read_atom_records(path)

def records_to_universe(atom_records):
    """
    Build an MDAnalysis Universe from an atom record.
    """
    import numpy as np
    import MDAnalysis
    # group atoms into residues, and residues into segments (chains)
    resindices = []
    # (chain, resnum, resname) per residue, in order
    residue_info = [] 
    current_res_key = None
    res_idx = -1
    for rec in atom_records:
      res_key = (rec['chain'], rec['resnum'], rec['resname'])
      if res_key != current_res_key:
          res_idx += 1
          residue_info.append(res_key)
          current_res_key = res_key
      resindices.append(res_idx)
      
    # define neccesary parameters to make universe
    segindices = []
    segment_chains = []
    current_chain = None
    seg_idx = -1
    for chain, _, _ in residue_info:
      if chain != current_chain:
          seg_idx += 1
          segment_chains.append(chain)
          current_chain = chain
      segindices.append(seg_idx)
    # number of atoms/residues/chains 
    n_atoms = len(atom_records)
    n_residues = len(residue_info)
    n_segments = len(segment_chains)
    
    # build empty universe with correct topology shape
    u = MDAnalysis.Universe.empty(n_atoms, n_residues=n_residues, n_segments=n_segments,
      atom_resindex=resindices, residue_segindex=segindices, trajectory=True,)
    
    # add atom attributes
    u.add_TopologyAttr('name', [a['atom'] for a in atom_records])
    u.add_TopologyAttr('element', [a['element'] for a in atom_records])
    u.add_TopologyAttr('occupancy', [a['occupancy'] for a in atom_records])
    u.add_TopologyAttr('tempfactor', [a['bfactor'] for a in atom_records])
    u.add_TopologyAttr('chainID', [a['chain'] for a in atom_records])
    # add residue attributes
    u.add_TopologyAttr('resname', [r[2] for r in residue_info])
    u.add_TopologyAttr('resid', [r[1] for r in residue_info])
    # add segment attributes
    u.add_TopologyAttr('segid', segment_chains)
    # add coordinates
    positions = np.array([[a['x'], a['y'], a['z']] for a in atom_records], dtype=np.float32)
    u.atoms.positions = positions
    return u
