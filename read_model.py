import voxlib.voxelize as voxel

import torch as th

import matplotlib.pyplot as plt

import argparse


def voxelise_model(model_path: str, size: int) -> th.Tensor:
    points_generator = voxel.voxelize(model_path, resolution=size)

    # one channel
    model_mat = th.zeros(1, size, size, size)

    for p in points_generator:
        p_new = p[0] + (size // 2 - 1), \
                p[1] + (size // 2 - 1), \
                p[2] + (size // 2 - 1)

        model_mat[0, p_new[0], p_new[1], p_new[2]] = 1

    return model_mat


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Main - voxelise")
    parser.add_argument("model", type=str, help="STL/OBJ model")
    args = parser.parse_args()

    size_res = 64

    mat_cub = voxelise_model(args.model, size_res)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(mat_cub.squeeze(0))
    plt.show()
