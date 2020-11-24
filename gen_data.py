import stl
import numpy as np

import argparse


def gen_rand_rotation(stl_path: str, out_file: str) -> None:
    mesh = stl.mesh.Mesh.from_file(stl_path)

    rand_vec = np.random.rand(3) * 2 - 1
    rand_vec /= np.linalg.norm(rand_vec)

    mesh.rotate(rand_vec, np.random.rand() * np.pi * 2)

    mesh.save(out_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Gen STL")
    parser.add_argument("stl_file", type=str)
    parser.add_argument("out_stl_file", type=str)
    args = parser.parse_args()

    gen_rand_rotation(args.stl_file, args.out_stl_file)
