from os.path import split
from datetime import datetime

import numpy as np
from matplotlib.tri import Triangulation

# compute deformation and plot
def get_deformation_on_triangulation(x, y, u, v, t):
    """ Compute deformation for given nodes.

    Input X, Y, U, V are given for individual N nodes. Nodes coordinates are triangulated and
    area, perimeter and deformation is computed for M elements.

    Parameters
    ----------
    x : Nx1 ndarray
        X-coordinates of nodes, m
    y : Nx1 ndarray
        Y-coordinates of nodes, m
    u : Nx1 ndarray
        U-component of nodes, m/s
    v : Nx1 ndarray
        V-component of nodes, m/s
    t : 3xM array
        Triangulation (indices of input nodes for each element)

    Returns
    -------
    e1 : Mx1 array
        Divergence, 1/s
    e2 : Mx1 array
        Shear, 1/s
    e3 : Mx1 array
        Vorticity, 1/s
    a : Mx1 array
        Area, m2
    p : Mx1 array
        Perimeter, m
    """

    # coordinates and speeds of corners of each element
    xt, yt, ut, vt = [i[t].T for i in (x, y, u, v)]

    # side lengths (X,Y,tot)
    tri_x = np.diff(np.vstack([xt, xt[0]]), axis=0)
    tri_y = np.diff(np.vstack([yt, yt[0]]), axis=0)
    tri_s = np.hypot(tri_x, tri_y)
    # perimeter
    tri_p = np.sum(tri_s, axis=0)
    s = tri_p/2
    # area
    tri_a = np.sqrt(s * (s - tri_s[0]) * (s - tri_s[1]) * (s - tri_s[2]))

    # deformation components
    e1, e2, e3 = get_deformation_elems(xt, yt, ut, vt, tri_a)

    return e1, e2, e3, tri_a, tri_p

def get_deformation_elems(x, y, u, v, a):
    """ Compute deformation for given elements.

    Input X, Y, U, V are organized in three columns: for each node of M elements.
    To convert deformation rates from 1/s to %/day outputs should be multiplied by 8640000.

    Parameters
    ----------
    x : 3xM ndarray
        X-coordinates of nodes, m
    y : 3xM ndarray
        Y-coordinates of nodes, m
    u : 3xM ndarray
        U-component of nodes, m/s
    v : 3xM ndarray
        V-component of nodes, m/s
    a : Mx1 ndarray
        area of elements, m2

    Returns
    -------
    e1 : Mx1 array
        Divergence, 1/s
    e2 : Mx1 array
        Shear, 1/s
    e3 : Mx1 array
        Vorticity, 1/s

    """
    # contour integrals of u and v [m/s * m ==> m2/s]
    ux = uy = vx = vy = 0
    for i0, i1 in zip([1, 2, 0], [0, 1, 2]):
        ux += (u[i0] + u[i1]) * (y[i0] - y[i1])
        uy -= (u[i0] + u[i1]) * (x[i0] - x[i1])
        vx += (v[i0] + v[i1]) * (y[i0] - y[i1])
        vy -= (v[i0] + v[i1]) * (x[i0] - x[i1])
    # divide integral by double area [m2/s / m2 ==> 1/day]
    ux, uy, vx, vy =  [i / (2 * a) for i in (ux, uy, vx, vy)]

    # deformation components
    e1 = ux + vy
    e2 = ((ux - vy) ** 2 + (uy + vx) ** 2) ** 0.5
    e3 = vx - uy

    return e1, e2, e3

def get_deformation_nodes(x, y, u, v):
    """ Compute deformation for given nodes.

    Input X, Y, U, V are given for individual N nodes. Nodes coordinates are triangulated and
    area, perimeter and deformation is computed for M elements.

    Parameters
    ----------
    x : Nx1 ndarray
        X-coordinates of nodes, m
    y : Nx1 ndarray
        Y-coordinates of nodes, m
    u : Nx1 ndarray
        U-component of nodes, m/s
    v : Nx1 ndarray
        V-component of nodes, m/s

    Returns
    -------
    e1 : Mx1 array
        Divergence, 1/s
    e2 : Mx1 array
        Shear, 1/s
    e3 : Mx1 array
        Vorticity, 1/s
    a : Mx1 array
        Area, m2
    p : Mx1 array
        Perimeter, m
    t : 3xM array
        Triangulation (indices of input nodes for each element)
    """
    tri = Triangulation(x, y)

    e1, e2, e3, tri_a, tri_p = get_deformation_on_triangulation(x, y, u, v, tri.triangles)

    return e1, e2, e3, tri_a, tri_p, tri.triangles

def get_deformation_2files(n1, n2):
    """ Calculate deformation from two neXtSIM snapshots.

    U/V is calculated as displacement of nodes between snapshots. Deformation is not computed
    for all elements. Matching nodes on the input NextsimBin objects are found, triangulated
    and used for computing deformation.
    Coordinates, triangulation and deformation for N nodes and M elements is returned.

    Parameters
    ----------
    n1 : name of the file with the first snapshot
    n2 : name of the file with the second snapshot

    Returns
    -------
    e1 : Mx1 array
        Divergence, 1/s
    e2 : Mx1 array
        Shear, 1/s
    e3 : Mx1 array
        Vorticity, 1/s
    a : Mx1 array
        New element area, m2
    p : Mx1 array
        New element perimeter, m2
    t : 3xM array
        Triangulation of matching nodes
    x : Nx1 array
        X-coordinates of matching nodes, m
    y : Nx1 array
        Y-coordinates of matching nodes, m
    u : Nx1 array
        U-component of speed of matching nodes, m/s
    v : Nx1 array
        V-component of speed of matching nodes, m/s

    """
    x, y, u, v = get_drift_2files(n1, n2)
    e1, e2, e3, a, p, t = get_deformation_nodes(x, y, u, v)

    return e1, e2, e3, a, p, t, x, y, u, v

def filename2datetime(filename):
    return datetime.strptime(
        split(filename)[-1].split('_')[1],
         '%Y%m%dT%H%M%SZ.npz'
    )

def get_drift_2files(n1, n2):
    """ Calculate ice drift between two neXtSIM snapshots.

    U/V is calculated as displacement of nodes between snapshots. Drift is not computed
    for all elements. Only matching nodes on the input NextsimBin objects are found.
    Coordinates and drift matching nodes is returned.

    Parameters
    ----------
    n1 : name of the file with the first snapshot
    n2 : name of the file with the second snapshot

    Returns
    -------
    x : Nx1 array
        X-coordinates of matching nodes, m
    y : Nx1 array
        Y-coordinates of matching nodes, m
    u : Nx1 array
        U-component of speed of matching nodes, m/s
    v : Nx1 array
        V-component of speed of matching nodes, m/s

    """
    datetime1 = filename2datetime(n1)
    n1 = dict(np.load(n1))
    x1 = n1['x']
    y1 = n1['y']
    i1 = n1['i']

    datetime2 = filename2datetime(n2)
    n2 = dict(np.load(n2))
    x2 = n2['x']
    y2 = n2['y']
    i2 = n2['i']

     # indices of nodes common to 0 and 1
    ids_cmn_12, ids1i, ids2i = np.intersect1d(i1, i2, return_indices=True)

     # coordinates of nodes of common elements
    x1n = x1[ids1i]
    y1n = y1[ids1i]
    x2n = x2[ids2i]
    y2n = y2[ids2i]

    delta_t = (datetime2 - datetime1).total_seconds()
    u = (x2n - x1n) / delta_t
    v = (y2n - y1n) / delta_t
    return x2n, y2n, u, v