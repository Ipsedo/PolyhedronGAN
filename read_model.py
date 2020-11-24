import voxlib.voxelize as voxel
import argparse
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Main- voxelise")

    parser.add_argument("model", type=str, help="STL/OBJ model")

    args = parser.parse_args()

    size = 64

    res = voxel.voxelize(args.model, resolution=size)

    points = [p for p in res]
    min_x, min_y, min_z = 0, 0, 0

    for p in points:
        min_x = min(min_x, p[0])
        min_y = min(min_y, p[1])
        min_z = min(min_z, p[2])

    mat_cub = np.zeros((size, size, size))

    for p in points:
        p_new = p[0] - min_x, p[1] - min_y, p[2] - min_z

        mat_cub[p_new[0], p_new[1], p_new[2]] = 1

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(mat_cub)
    plt.show()


