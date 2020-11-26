import voxlib.voxelize as voxel
import stl
import numpy as np

import torch as th

import matplotlib.pyplot as plt
import os

import argparse


def gen_rand_rotation(stl_path: str, out_file: str) -> None:
    mesh = stl.mesh.Mesh.from_file(stl_path)

    rand_vec = np.random.rand(3) * 2 - 1
    rand_vec /= np.linalg.norm(rand_vec)

    mesh.rotate(rand_vec, np.random.rand() * np.pi * 2)

    mesh.save(out_file)


def voxelise_model(
        model_path: str, size: int, random_scale: bool
) -> th.Tensor:
    if random_scale:
        scaled_size = int(np.random.uniform(0.9, 1.0) * size)
        scaled_size -= scaled_size % 2
    else:
        scaled_size = size

    points_generator = voxel.voxelize(
        model_path, resolution=scaled_size
    )

    # one channel
    model_mat = th.zeros(1, size, size, size)

    min_x, min_y, min_z = scaled_size, scaled_size, scaled_size
    max_x, max_y, max_z = -scaled_size, -scaled_size, -scaled_size

    points = []
    for p in points_generator:
        min_x = min(p[0], min_x)
        min_y = min(p[1], min_y)
        min_z = min(p[2], min_z)

        max_x = max(p[0], max_x)
        max_y = max(p[1], max_y)
        max_z = max(p[2], max_z)

        points.append(p)

    for p in points:
        p_new = p[0] - min_x, \
                p[1] - min_y, \
                p[2] - min_z

        p_new_2 = p_new[0] + size // 2, \
                  p_new[1] + size // 2, \
                  p_new[2] + size // 2

        try:
            model_mat[0, p_new[0], p_new[1], p_new[2]] = 1
        except Exception as e:
            print(min_x, min_y, min_z)
            print(max_x, max_y, max_z)
            print(size)
            print(scaled_size)
            print(p)
            print(p_new)
            print(p_new_2)
            raise e

    return model_mat


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Main - voxelise")
    parser.add_argument("model", type=str, help="STL/OBJ model")

    parser.add_argument("-s", type=int, required=True, dest="size")

    sub_parser = parser.add_subparsers()
    sub_parser.required = True
    sub_parser.dest = "mode"

    read_parser = sub_parser.add_parser("read")
    gen_parser = sub_parser.add_parser("generate")

    gen_parser.add_argument("nb_example", type=int)
    gen_parser.add_argument("--stl-out-path", type=str, required=True)
    gen_parser.add_argument("--tensor-out-path", type=str, required=True)

    args = parser.parse_args()

    if args.mode == "read":
        print("read")
        mat_cub = voxelise_model(args.model, args.size, True)

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.voxels(mat_cub.squeeze(0))
        plt.show()
    elif args.mode == "generate":
        print("generate")

        res_tensor = th.empty(args.nb_example, 1, args.size, args.size,
                              args.size)

        for i in range(args.nb_example):
            print(i, "/", args.nb_example)
            gen_rand_rotation(args.model, args.stl_out_path + f"/out_{i}.stl")

            res_tensor[i, :, :, :, :] = voxelise_model(
                args.stl_out_path + f"/out_{i}.stl", args.size, True
            )

            os.remove(args.stl_out_path + f"/out_{i}.stl")

        th.save(res_tensor, args.tensor_out_path)
