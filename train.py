import networks

import torch as th

from tqdm import tqdm

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Main - train")

    parser.add_argument("--tensor-file", type=str, required=True)

    args = parser.parse_args()

    data = th.load(args.tensor_file)

    rand_channel = 16

    mean_vec = th.randn(rand_channel)
    rand_mat = th.randn(rand_channel, rand_channel)
    cov_mat = rand_mat.t().matmul(rand_mat)
    multi_norm = th.distributions.MultivariateNormal(mean_vec, cov_mat)

    gen = networks.Generator()
    disc = networks.Disciminator()
    gen.cuda()
    disc.cuda()

    disc_lr = 3e-5
    gen_lr = 2e-5

    disc_optimizer = th.optim.Adam(disc.parameters(), lr=disc_lr)
    gen_optimizer = th.optim.Adam(gen.parameters(), lr=gen_lr)


    def __gen_rand(
            curr_batch_size: int
    ) -> th.Tensor:
        return multi_norm.sample(
            (curr_batch_size, 8, 8, 8)).permute(0, 4, 1, 2, 3)


    nb_epoch = 10
    batch_size = 4
    nb_batch = data.size(0) // batch_size

    for e in range(nb_epoch):
        tqdm_bar = tqdm(range(nb_batch))

        for b_idx in tqdm_bar:
            i_min = b_idx * batch_size
            i_max = (b_idx + 1) * batch_size

            x_real = data[i_min:i_max, :, :, :, :].cuda()
            rand_fake = __gen_rand(i_max - i_min).cuda()

            x_fake = gen(rand_fake)

            out_real = disc(x_real)
            out_fake = disc(x_fake)

            error_tp = (1. - out_real).mean().item()
            error_tn = out_fake.mean().item()

            disc_loss = networks.discriminator_loss(out_real, out_fake)

            gen_optimizer.zero_grad()
            disc_optimizer.zero_grad()

            disc_loss.backward()
            disc_optimizer.step()

            disc_loss = disc_loss.item()

            # Train generator
            rand_fake = __gen_rand(i_max - i_min).cuda()

            x_fake = gen(rand_fake)

            out_fake = disc(x_fake)

            gen_loss = networks.generator_loss(out_fake)

            disc_optimizer.zero_grad()
            gen_optimizer.zero_grad()

            gen_loss.backward()
            gen_optimizer.step()

            gen_grad_norm = th.tensor(
                [p.grad.norm() for p in gen.parameters()]
            ).mean()

            disc_grad_norm = th.tensor(
                [p.grad.norm() for p in disc.parameters()]
            ).mean()

            tqdm_bar.set_description(
                f"Epoch {e} : "
                f"disc_loss = {disc_loss:.6f}, "
                f"gen_loss = {gen_loss:.6f}, "
                f"e_tp = {error_tp:.4f}, "
                f"e_tn = {error_tn:.4f}, "
                f"gen_gr = {gen_grad_norm.item():.4f}, "
                f"disc_gr = {disc_grad_norm.item():.4f}"
            )
