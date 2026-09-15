""" 
Calculate Minimum fusion-linker length by calculating shortest path 
through a grid that excludes points near the protein. 
shortest path is calculated using Dijkstra. 
"""

import warnings

import numpy as np
import MDAnalysis
from scipy.sparse import csr_array
from scipy.sparse.csgraph import dijkstra
from scipy.ndimage import generate_binary_structure

def get_grid(u_in, padding=3, gridstep=1):
    """
    Create a regular grid of points padded around the protein bounding box.
    """
    pos = u_in.atoms.positions
    x_max, x_min = pos[:, 0].max() + padding, pos[:, 0].min() - padding
    y_max, y_min = pos[:, 1].max() + padding, pos[:, 1].min() - padding
    z_max, z_min = pos[:, 2].max() + padding, pos[:, 2].min() - padding

    xs = np.arange(x_min, x_max, gridstep)
    ys = np.arange(y_min, y_max, gridstep)
    zs = np.arange(z_min, z_max, gridstep)
    x, y, z = np.meshgrid(xs, ys, zs, indexing="ij")
    grid_points = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    grid_dim = (len(xs), len(ys), len(zs))
    return grid_points, grid_dim


def grid_to_u(grid_points):
    """
    Wrap a set of grid coordinates in a throwaway MDAnalysis universe.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="there is no reference attributes")
        grid = MDAnalysis.Universe.empty(
            n_atoms=len(grid_points),
            atom_resindex=len(grid_points) * [0],
            trajectory=True,)
        grid.atoms.positions = grid_points
        grid.add_TopologyAttr("resname", ["UNK"])
    return grid


def get_mask(u_in, grid_points, rad=3):
    """
    Boolean mask True for grid points with rad distance from any protein atom.
    """
    grid_u = grid_to_u(grid_points)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="there is no reference attributes")
        combined = MDAnalysis.Merge(grid_u.atoms, u_in.atoms)
    sel_comb = combined.select_atoms("resname UNK and not around {} protein".format(rad))
    prot_mask = np.zeros(len(grid_points))
    prot_mask[sel_comb.atoms.ix_array] = 1
    return prot_mask.astype(bool)


def match_point_fullgrid(grid, mask, point):
    """
    Nearest unmasked grid point to a given point
    """
    valid_ind = np.where(mask)[0]
    dists = np.sqrt(np.sum((grid[valid_ind] - point) ** 2, axis=1))
    ind = valid_ind[np.argmin(dists)]
    # return index and distance to point
    return ind, np.min(dists)


def create_csr_graph(grid_dim, prot_mask, gridstep=1):
    """
    Build a neighbour graph over the grid.
    """
    lx, ly, lz = grid_dim
    N = lx * ly * lz
    lin_index = np.arange(N).reshape((lx, ly, lz))
    prot_mask3d = prot_mask.reshape((lx, ly, lz))
    # 3x3x3 kernel -> 26 neighbour offsets (origin excluded).
    neighbor_kernel = generate_binary_structure(3, 3)
    neighbor_offsets = np.argwhere(neighbor_kernel) - 1
    neighbor_offsets = neighbor_offsets[~np.all(neighbor_offsets == 0, axis=1)]

    rows, cols, weights = [], [], []
    for dx, dy, dz in neighbor_offsets:
        valid = prot_mask3d & np.roll(prot_mask3d, shift=(-dx, -dy, -dz), axis=(0, 1, 2))
        # Clip wrapped edges so np.roll cannot connect across grid boundaries.
        if dx < 0: valid[:-dx, :, :] = False
        if dx > 0: valid[lx - dx:, :, :] = False
        if dy < 0: valid[:, :-dy, :] = False
        if dy > 0: valid[:, ly - dy:, :] = False
        if dz < 0: valid[:, :, :-dz] = False
        if dz > 0: valid[:, :, lz - dz:] = False
        src_ids = lin_index[valid]
        dst_ids = lin_index[np.roll(valid, shift=(dx, dy, dz), axis=(0, 1, 2))]
        w = gridstep * np.sqrt(dx * dx + dy * dy + dz * dz)
        rows.append(src_ids)
        cols.append(dst_ids)
        weights.append(np.full(src_ids.shape, w))
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    weights = np.concatenate(weights)
    return csr_array((weights, (rows, cols)), shape=(N, N))


def reconstruct_path(predecessors, start, end):
    """
    reconstruct path from Dijkstra predecessor array
    """
    path = []
    i = end
    while True:
        path.append(i)
        if i == start:
            break
        i = predecessors[i]
    return path[::-1]


def shortest_linker_path(u_in, sel_start, sel_end, gridstep=1, padding=4, rad=3,
                         progress=None):
    """
    Shortest solvent path between two residue CAs.
    u_in: MDAnalysis Universe
    sel_start, sel_end: the selection strings for the two endpoint residues.
    progress: callable, optional called with a short status string (for logging/UI).
    Returns distance in Angstrom and a grid universe tracing the path.
    """
    def note(msg):
        if progress:
            progress(msg)

    start = u_in.select_atoms(sel_start)
    end = u_in.select_atoms(sel_end)
    # Exclude the endpoint residues themselves so the grid doesn't mask them out.
    u_no_se = u_in.select_atoms("protein").subtract(start + end)

    note("make grid")
    grid_points, grid_dim = get_grid(u_no_se, gridstep=gridstep, padding=padding)
    note("make mask")
    prot_mask = get_mask(u_no_se, grid_points, rad=rad)
    note("make graph")
    graph = create_csr_graph(grid_dim, prot_mask, gridstep=gridstep)

    coords_start = start.select_atoms("name CA").atoms.positions
    coords_end = end.select_atoms("name CA").atoms.positions
    ind_s, dist_s = match_point_fullgrid(grid_points, prot_mask, coords_start)
    ind_e, dist_e = match_point_fullgrid(grid_points, prot_mask, coords_end)

    note("dijkstra")
    dist_matrix, predecessors = dijkstra(
        csgraph=graph, directed=False, indices=ind_e, return_predecessors=True)
    if not np.isfinite(dist_matrix[ind_s]):
        # start and end not connected, scipy returns -9999 and loops forever
        raise ValueError("no solvent path")
    shortest_dist = dist_matrix[ind_s] + dist_s + dist_e
    path = reconstruct_path(predecessors, ind_e, ind_s)
    path_u = grid_to_u(grid_points[path])
    return float(shortest_dist), path_u


def residues_for_length(distance, a_per_aa=3.8, buffer=5):
    """
    Convert a distance (A) to a minimum and buffered residue count.
    a_per_aa represents the backbone distance covered by one amino acid. (A)
    buffer represents the added number of residues on top of the minimal distance
    """
    import math
    if distance is None or not math.isfinite(distance):
        return math.inf, math.inf
    minimum = int(distance / a_per_aa) + 1
    return minimum, minimum + buffer


def compute_linker(struc, start_res, end_res, chain_start, chain_end,
                   gridstep=1, padding=4, rad=3, progress=None):
    """
    Convenience wrapper: load a structure and compute the linker path.
    """
    u_in = MDAnalysis.Universe(str(struc))
    sel_start = "chainID {} and resid {}".format(chain_start, start_res)
    sel_end = "chainID {} and resid {}".format(chain_end, end_res)
    return shortest_linker_path(
        u_in, sel_start, sel_end, gridstep=gridstep, padding=padding, rad=rad,
        progress=progress,)
