import orbitals
import plotly.graph_objects as go
import numpy as np
from main import Ao, Mo, Gaussian1d
from integrals import factorial2


def ms(x, y, z, radius, resolution=20):
    """Return the coordinates for plotting a sphere centered at (x,y,z)"""
    u, v = np.mgrid[0:2*np.pi:resolution*2j, 0:np.pi:resolution*1j]
    X = radius * np.cos(u)*np.sin(v) + x
    Y = radius * np.sin(u)*np.sin(v) + y
    Z = radius * np.cos(v) + z
    return (X, Y, Z)

# ================================== 
#
#    Main
# 
# ==================================



def main():

    molecules = dict(
        formaldehyde=(
            ["H", "H", "C", "O"],

            [1.88973*np.array([0.000000,    0.920117,   -1.102442]),
            1.88973*np.array([-0.000000,   -0.920117,   -1.102442]),
            1.88973*np.array([0.000000,    0.000000,   -0.534011]),
            1.88973*np.array([-0.000000,  -0.000000,    0.676119])]
            ),
        water=(
            ["O", "H", "H"],

            [1.88973*np.array([0., 0., 0.116321]),
            1.88973*np.array([0., 0.751155, -0.465285]),
            1.88973*np.array([0., -0.751155, -0.465285])
            ]
        ),
        ethene=(
            ["C", "H", "H", "C", "H", "H"],

            [1.88973*np.array([0., 0., 0.660022]),
            1.88973*np.array([0., 0.910176, 1.227515]),
            1.88973*np.array([0., -0.910176, 1.227515]),
            1.88973*np.array([0., 0., -0.660022]),
            1.88973*np.array([0., -0.910176, -1.227515]),
            1.88973*np.array([0., 0.910176, -1.227515])
            ]),
        ethanol=(
            ["H", "H", "H", "C", "C", "H", "H", "O", "H"],
            [
            1.88973*np.array([1.252061,   -0.865351,    0.877772]),
            1.88973*np.array([2.081899,    0.423065,   -0.000031]),
            1.88973*np.array([1.252039,   -0.865523,   -0.877553]),
            1.88973*np.array([1.219651,   -0.233443,    0.000049]),
            1.88973*np.array([-0.075905,    0.569943,   -0.000042]),
            1.88973*np.array([-0.122215,    1.205134,   -0.878830]),
            1.88973*np.array([-0.122329,    1.205198,    0.878698]),
            1.88973*np.array([-1.149376,   -0.395839,   -0.000105]),
            1.88973*np.array([-2.008922,    0.045191,    0.000742])
            ]),        
        )

    atom_color = dict(
        H="grey",
        O="red",
        C="black"
    )

    atom_radii = dict(
        H=0.2,
        O=0.5,
        C=0.4
    )

    molecule = "water"
    basis = "STO-3G"
    filename = f"{basis}.txt"
    ao_dict = orbitals.create_ao_dict(filename)

    ao_list = []

    atoms = molecules[molecule][0]
    coordinates = molecules[molecule][1]

    for atom, coordinate in zip(atoms, coordinates):
        for atomic_orbital in ao_dict[atom]:
            new_orbital = Ao(coordinate, atomic_orbital.angular, len(atomic_orbital.coefficients))
            new_orbital.exponents = atomic_orbital.exponents.copy()
            new_orbital.coefficients = atomic_orbital.coefficients.copy()
            new_orbital.normalize()
            ao_list.append(new_orbital)

    C = np.genfromtxt(f"coefficients_{molecule}_{basis}.csv", delimiter=",")

    mo_list = []
    for i in range(len(ao_list)):
        coeffs = C[:,i]
        new_mo = Mo(ao_list,coeffs)
        mo_list.append(new_mo)

    X, Y, Z = np.mgrid[-4:4:40j, -4:4:40j, -4:4:40j]
    x=X.flatten()
    y=Y.flatten()
    z=Z.flatten()
    values=np.array([mo_list[5](x, y, z) for x, y, z in zip(x, y, z)])

    spheres = [ms(coordinate[0], coordinate[1], coordinate[2], atom_radii[atom]) for coordinate, atom in zip(coordinates, atoms)]

    data = [
        go.Isosurface(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=values,
            opacity=0.6,
            isomin=-0.1,
            isomax=0.1,
            surface_count=2,
            caps=dict(x_show=False, y_show=False, z_show=False)),
    ]
    for idx, sphere in enumerate(spheres):
        data.append(
            go.Surface(x=sphere[0], y=sphere[1], z=sphere[2],
            showscale=False,
            colorscale=[[0, atom_color[atoms[idx]]], [1, atom_color[atoms[idx]]]]),
        )

    fig= go.Figure(
        data=data
    )

    fig.show()

if __name__== "__main__":
    main()