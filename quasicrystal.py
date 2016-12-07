# Copyright (c) 2014-2015 and later, Ion Cosma Fulga, Dmitry Pikulin,
# and Terry Loring. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#     1) Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#
#     2) Redistributions in binary form must reproduce the above
#     copyright notice, this list of conditions and the following
#     disclaimer in the documentation and/or other materials provided
#     with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


"""
--------------------------------------------------------------
          Aperiodic Weak Topological Superconductors
--------------------------------------------------------------

This Kwant script generates and computes the properties of a
topological superconductor on a two-dimensional Ammann-Beenker
tiling.

I. C. Fulga, D. I. Pikulin, and T. A. Loring
"Aperiodic Weak Topological Superconductors"
arXiv:XXXX.XXXXX.

For more information on Kwant:

http://kwant-project.org/

C.W. Groth, M. Wimmer, A.R. Akhmerov and X. Waintal
"Kwant: a software package for quantum transport"
arXiv:1309.2916.

For examples of usage, see the main() function, which
reproduces some of our numerical results. This script can be
imported in a python interface or simply run as:

python quasicrystal.py
"""
import numpy as np
import kwant
from kwant.digest import uniform
import pylab as py
py.ion()

# Define Pauli matrices
sx = np.array([[0, 1], [1, 0]], complex)
sy = np.array([[0 , -1j], [1j, 0]])
sz = np.array([[1, 0], [0, -1]], complex)
s0 = np.array([[1, 0], [0, 1]], complex)

class SimpleNamespace(object):
    """A simple container for parameters."""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class Tile(object):
    '''Class describing the rhombus and triangular tiles.'''
    def __init__(self, Type, A, B, C, D=(0, 0)):
        self.Type = Type
        if Type == 0: # Triangle.
            self.A = A
            self.B = B
            self.C = C
        else: # Rhombus.
            self.A = A
            self.B = B
            self.C = C
            self.D = D

p = SimpleNamespace(d=1.0, t=1.0, dis=0, salt='', mu=1.9,
                    stripe_l=1.0, nrwires=13, gapratio=0.9)

def subdivide(tile):
    '''Subdivide a tile as in Fig. 5b.'''
    new_tiles = []
    if tile.Type == 0: # Subdivide a triangular tile.
        AB = tile.B - tile.A
        AC = tile.C - tile.A
        CB = tile.B - tile.C
        AG = AC / (1 + np.sqrt(2))
        CE = CB / (1 + np.sqrt(2))
        AH = AB / (2 + np.sqrt(2))
        AD = AB * (1 + np.sqrt(2)) / (2 + np.sqrt(2))
        AF = AH + AG
        G = tile.A + AG
        F = tile.A + AF
        H = tile.A + AH
        D = tile.A + AD
        E = tile.C + CE
        new_tiles.append(Tile(1, tile.A, H, F, G))
        new_tiles.append(Tile(0, D, H, F))
        new_tiles.append(Tile(0, tile.C, G, F))
        new_tiles.append(Tile(1, tile.C, F, D, E))
        new_tiles.append(Tile(0, tile.B, E, D))

    elif tile.Type == 1: # Subdivide a rhombus tile.
        AB = tile.B - tile.A
        AD = tile.D - tile.A
        CB = tile.B - tile.C
        CD = tile.D - tile.C
        AM = AD / (1 + np.sqrt(2))
        AH = AB / (1 + np.sqrt(2))
        AL = AM + AH
        CE = CB / (1 + np.sqrt(2))
        CF = CD / (1 + np.sqrt(2))
        CG = CE + CF
        E = tile.C + CE
        F = tile.C + CF
        G = tile.C + CG
        H = tile.A + AH
        L = tile.A + AL
        M = tile.A + AM
        new_tiles.append(Tile(1, tile.A, H, L, M))
        new_tiles.append(Tile(0, tile.B, H, L))
        new_tiles.append(Tile(0, tile.D, M, L))
        new_tiles.append(Tile(1, tile.B, G, tile.D, L))
        new_tiles.append(Tile(0, tile.B, E, G))
        new_tiles.append(Tile(0, tile.D, F, G))
        new_tiles.append(Tile(1, tile.C, E, G, F))

    return new_tiles

def join_separated_tiles(tile_array):
    '''Combine pairs of triangular tiles, as in Fig. 5c.'''
    new_tile_array = []
    for ind1, tile1 in enumerate(tile_array):
        if tile1.Type == 1:
            new_tile = tile1
        else:
            for ind2, tile2 in enumerate(tile_array):
                if ind2 > ind1 and \
                   np.linalg.norm(tile1.B - tile2.B) <= 1e-13 and \
                   np.linalg.norm(tile1.A - tile2.A) <= 1e-13:
                    new_tile = Tile(1, tile1.A, tile1.C, tile1.B, tile2.C)
        new_tile_array.append(new_tile)

    return new_tile_array

def unique(s):
    """Return a list containing the elements of s, but without duplicates."""
    n = len(s)
    t = list(s)
    t.sort()
    last = t[0]
    lasti = i = 1
    while i < n:
        if t[i] != last:
            t[lasti] = last = t[i]
            lasti += 1
        i += 1

    return t[:lasti]

def build_tiling(ndiv=4):
    """ Construct a square patch of the Ammann-Beenker tiling by applying
    the subdivision algorithm 'ndiv' times.
    """
    # Create the initial tiling, made of two triangles
    a = Tile(0, np.asarray([0., 0.]),
                np.asarray([1., 1.]),
                np.asarray([1., 0.]))
    b = Tile(0, np.asarray([0., 0.]),
                np.asarray([1., 1.]),
                np.asarray([0., 1.]))

    tile_array = [a, b]

    for i in range(ndiv):
        new_tile_array = []
        for tile in tile_array:
            new_tile_array.extend(subdivide(tile))
        tile_array = new_tile_array

    return join_separated_tiles(tile_array)

def onsite(site, p):
    """Onsite Hamiltonian element of H_QC."""
    dis = p.mu + (2 * uniform(repr(site), p.salt) - 1) * p.dis
    onsite = - dis * sz
    return onsite

def stripe_hopping(x):
    """Selectively reduce hopping amplitudes based on their x position."""
    wire_width = 1.0 / (p.nrwires + p.nrwires * p.gapratio - p.gapratio)
    space_width = wire_width * p.gapratio
    modx = x % (wire_width + space_width)
    hop = 1 if (modx <= wire_width) else p.stripe_l
    return hop

def hop(site1, site2, p):
    """Hopping matrix of H_QC."""
    x1, y1 = site1.pos
    x2, y2 = site2.pos

    stripe = stripe_hopping(x1 / 2 + x2 / 2)
    angle = np.arctan2(y2 - y1, x2 - x1)

    hopping = -p.t * sz
    hopping = hopping - 1j / 2 * p.d * sx * np.cos(angle)
    hopping = hopping - 1j / 2 * p.d * sy * np.sin(angle)
    return  hopping * stripe

def make_modes_func(t, N_orb, W):
    """Construct lead modes.

    Parameters:
    -----------
    t : real.
        Hopping amplitude in the lead.
    N_orb : integer.
        Number of orbitals per site.
    W : integer.
        Number of sites connected to the lead.
    """
    momenta = np.array([ np.sign(t) * np.pi / 2] * N_orb * W +
                       [-np.sign(t) * np.pi / 2] * N_orb * W)
    velocities = np.array([-2 * np.abs(t)] * N_orb * W +
                          [ 2 * np.abs(t)] * N_orb * W)
    wave_functions = np.array(np.bmat(
        [[ 1/np.sqrt(np.abs(2 * t)) * np.eye(N_orb * W),
           1/np.sqrt(np.abs(2 * t)) * np.eye(N_orb * W)]]))
    prop_modes = kwant.physics.PropagatingModes(wave_functions,
                                                velocities, momenta)

    nmodes = N_orb * W
    vecs = wave_functions
    vecslmbdainv = t * np.dot(vecs,
        np.diag([-1j * np.sign(t)] * N_orb * W +
                [ 1j * np.sign(t)] * N_orb * W))
    stab_modes = kwant.physics.StabilizedModes(vecs, vecslmbdainv, nmodes)

    def modes_func(energy, args=[]):
        return (prop_modes, stab_modes)

    return modes_func

def get_vertex_hopping(tile_array):
    """Extract the vertices and links of a tiling."""
    coords = []
    hoppings = []
    for tile in tile_array:
        coords.append(list(tile.A))
        coords.append(list(tile.B))
        coords.append(list(tile.C))
        coords.append(list(tile.D))

        hoppings.append(list([list(tile.A), list(tile.B)]))
        hoppings.append(list([list(tile.B), list(tile.C)]))
        hoppings.append(list([list(tile.C), list(tile.D)]))
        hoppings.append(list([list(tile.D), list(tile.A)]))

    return coords, hoppings

def build_system(ndiv=3, leads=[]):
    """Construct a tight-binding model of a topological superconductor on a
    two-dimensional Ammann-Beenker tiling.

    Parameters:
    -----------
    ndiv : integer.
        Number of times the subdivision algorithm is applied.
    leads : list.
        List describing where and how many leads are attached to the system.
        Can be empty or contain 'l', 'r', 't', 'b', corresponding to left,
        right, top, and bottom leads, respectively.

    Returns:
    --------
    sys : <kwant.builder.FiniteSystem> object.
        The finalized system H_QC.
    """
    # generates discretized coorinate system of integers called tiling
    tiling = build_tiling(ndiv)
    # gets list of coordinates on tiling and hopping coordinates as integers,
    # e.g. coords[0] = [0.0, 0.0], hoppings[0] = [[0.0, 0.0], [0.05, 0.05]]
    coords, hoppings = get_vertex_hopping(tiling)
    # filters out duplicates for coords and hoppings
    coords = unique(coords)
    hoppings = unique(hoppings)

    # this class allows us to turn our integer coordinates into the format
    # kwant requires
    class Quasicrystal(kwant.builder.SiteFamily):
        def __init__(self, coords):
            self.coords = coords
            super(Quasicrystal, self).__init__("quasicrystal", "")

        def normalize_tag(self, tag):
            try:
                tag = int(tag[0])
            except:
                raise KeyError

            if 0 <= tag < len(coords):
                return tag
            else:
                raise KeyError

        def pos(self, tag):
            return self.coords[tag]

    qc = Quasicrystal(coords) # here we turns our coords into kwants format
    sys = kwant.Builder() # initialize system
    for i in range(len(coords)):
        sys[qc(i)] = onsite # sets all the coordinates to have onsite energy

    for i in range(len(hoppings)):
        sys[qc(coords.index(hoppings[i][0])),
            qc(coords.index(hoppings[i][1]))] = hop # sets all the hoppings to have hop energy

    left_sites = []
    right_sites = []
    top_sites = []
    bot_sites = []
    # finds the coordinates that are left/right/top/bottom sites
    for ind, c in enumerate(coords):
        if c[0] == 0:
            left_sites.append(ind)
        if c[0] == 1:
            right_sites.append(ind)

        if c[1] == 0:
            top_sites.append(ind)
        if c[1] == 1:
            bot_sites.append(ind)

    if 'l' in leads:
        sites_left = [qc(i) for i in left_sites]
        modes_func_left = make_modes_func(1, 2, len(left_sites))
        lead_left = kwant.builder.ModesLead(modes_func_left, sites_left)
        sys.leads.append(lead_left)

    if 'r' in leads:
        sites_right = [qc(i) for i in right_sites]
        modes_func_right = make_modes_func(1, 2, len(right_sites))
        lead_right = kwant.builder.ModesLead(modes_func_right, sites_right)
        sys.leads.append(lead_right)

    if 't' in leads:
        sites_top = [qc(i) for i in top_sites]
        modes_func_top = make_modes_func(1, 2, len(top_sites))
        lead_top = kwant.builder.ModesLead(modes_func_top, sites_top)
        sys.leads.append(lead_top)

    if 'b' in leads:
        sites_bot = [qc(i) for i in bot_sites]
        modes_func_bot = make_modes_func(1, 2, len(bot_sites))
        lead_bot = kwant.builder.ModesLead(modes_func_bot, sites_bot)
        sys.leads.append(lead_bot)

    return sys.finalized()

def plot_amplitude(sys, wf, sitesize=0.5, hopsize=0.05):
    """Plot the total wavefunction amplitude.

    Parameters:
    -----------
    sys : <kwant.builder.FiniteSystem> object.
        Finalized system, as returned by build_system().
    wf : numpy.ndarray.
        Array of wafefunction amplitudes.
    sitesize : real.
        Scale factor setting the maximum area of a site.
    hopsize : real.
        Scale factor setting the maximum width of a link.
    """
    minwf = np.min(wf)
    maxwf = np.max(wf)
    def site_size(site):
        """Set the size of a site based on total wavefunction amplitude."""
        return (wf[site] - minwf) / (maxwf - minwf) * sitesize

    def site_col(site):
        """Set the color of a site based on total wavefunction amplitude."""
        return (wf[site] - minwf) / (maxwf - minwf)

    def hop_lw(site1, site2):
        """Set the thickness of hoppings to show the positions of Kitaev
        chains.
        """
        x1, y1 = sys.sites[site1].pos
        x2, y2 = sys.sites[site2].pos
        return stripe_hopping(x1/2 + x2/2) * hopsize

    kwant.plot(sys, site_size=site_size, site_color=site_col,
                cmap='gist_heat_r', hop_lw=hop_lw)

def get_transmission(sys):
    """Compute the transmission of a system.

    Parameters:
    -----------
    sys : <kwant.builder.FiniteSystem> object.
        Finalized system, as returned by build_system().

    Returns:
    --------
    transmission : real.
        Transmission through the system, assuming only two leads are attached.
    """
    smatrix = kwant.smatrix(sys, energy=0.0, args=[p])
    n, m = smatrix.data.shape
    tblock = np.asmatrix(smatrix.data[:n//2, m//2:])
    transmission = np.trace(tblock * tblock.H).real
    return transmission

def get_detr(sys):
    """Compute the determinant of the reflection block.

    Parameters:
    -----------
    sys : <kwant.builder.FiniteSystem> object.
        Finalized system, as returned by build_system().

    Returns:
    --------
    detr : real.
        Determinant of the reflection block, assuming only two leads
        are attached.
    """
    smatrix = kwant.smatrix(sys, energy=0.0, args=[p])
    n, m = smatrix.data.shape
    rblock = np.asmatrix(smatrix.data[:n//2, :m//2])
    detr = np.linalg.det(rblock)
    return detr.real

def pos_H(fsys, params, coord):
    """ Calculate the position operator in the 'coord' direction of the
        Hamiltonian of fsys.
    """
    H, ton, fon = fsys.hamiltonian_submatrix(return_norb=True,
                                             args=[params])
    x = np.zeros(H.shape[0])
    ind = 0
    for i in range(len(fsys.sites)):
        for j in range(ind, ind + ton[i]):
            x[j] = fsys.sites[i].pos[coord]

        ind += ton[i]

    return x

def strong_invariant(x, y, h):
    """Compute the strong pseudospectrum invariant.

    Parameters:
    -----------
    x : numpy.ndarray.
        X position operator.
    y : numpy.ndarray.
        Y position operator.
    h : numpy.ndarray.
        Hamiltonian of the system.

    Returns:
    --------
    inv : integer.
        Strong pseudospectrum invariant.
    """
    bc = np.array([[  1,  1],
                   [-1j, 1j]]) / np.sqrt(2)
    newbasis = np.asmatrix(np.kron(np.identity(h.shape[0]//2), bc))
    newh = newbasis * h * newbasis.H
    mat = np.bmat([[             x, y - 1j * newh],
                   [ y + 1j * newh,            -x]])
    evals = np.linalg.eigvalsh(mat)
    inv = (len(np.where(evals > 0)[0]) - len(np.where(evals <= 0)[0])) / 2
    return inv

def weak_invariant(x, y, h):
    """Compute the weak pseudospectrum invariant.

    Parameters:
    -----------
    x : numpy.ndarray.
        X position operator.
    y : numpy.ndarray.
        Y position operator.
    h : numpy.ndarray.
        Hamiltonian of the system.

    Returns:
    --------
    inv : integer.
        Weak pseudospectrum invariant.
    """
    bc = np.array([[  1,  1],
                   [-1j, 1j]]) / np.sqrt(2)
    newbasis = np.asmatrix(np.kron(np.identity(h.shape[0]//2), bc))
    newh = newbasis * h * newbasis.H
    maty = y + 1j * newh
    inv = np.linalg.slogdet(maty)[0].real
    return inv

def main():
    """Here we reproduce some of our numerical results."""

    sys = build_system(3, ['l', 'r'])
    kwant.plot(sys)
    py.title('Square patch of the Ammann-Beenker tiling.')

    # Compute the Hamiltonian and position operators.
    h = sys.hamiltonian_submatrix(args=[p])
    x = np.diag(pos_H(sys, p, 0) - 0.5)
    y = np.diag(pos_H(sys, p, 1) - 0.5)

    # Compute total wavefunction amplitude.
    e, v = np.linalg.eigh(h)
    myv = np.sum(np.abs(v[:, np.where(np.abs(e) <= 0.2)[0]]), axis=1)
    wf = myv[0::2] + myv[1::2]

    plot_amplitude(sys, wf, 1.0, 0.05)
    py.title('Total wavefunction amplitude: strong phase, |E| < 0.2')
    py.draw()

    print ('In the strong topological phase, the transmission from left ' + \
          'to right is G =', get_transmission(sys))
    print ('The strong invariant is C_ps =', strong_invariant(x, y, h))


    p.stripe_l = 0.2 # Selectively reduce hopping amplitudes

    print()
    print ('Generating a bigger tiling and computing its properties,', \
            'this may take a few minutes...')
    print()

    sys = build_system(4, ['l', 'r'])

    h = sys.hamiltonian_submatrix(args=[p])
    x = np.diag(pos_H(sys, p, 0) - 0.5)
    y = np.diag(pos_H(sys, p, 1) - 0.5)
    e, v = np.linalg.eigh(h)
    myv = np.sum(np.abs(v[:, np.where(np.abs(e) <= 0.1)[0]]), axis=1)
    wf = myv[0::2] + myv[1::2]

    plot_amplitude(sys, wf, 1.0, 0.1)
    py.title('Total wavefunction amplitude: weak phase, |E| < 0.1')
    py.draw()

    print ('In the weak topological phase, the transmission from left ' + \
          'to right is G =', get_transmission(sys))
    print ('The weak invariant is Q_y =', weak_invariant(x, y, h))

    sys = build_system(4, ['t', 'b'])
    print ('It agrees with the scattering matrix invariant, nu_y =', \
          get_detr(sys))

    a = input('Press Enter to exit...')

#if __name__ == "__main__":
#    main()
