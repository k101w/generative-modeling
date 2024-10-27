import os

import torch
import torch.nn.functional as F
from utils import get_args

from networks import Discriminator, Generator
from train import train_model


def compute_discriminator_loss(
    discrim_real, discrim_fake, discrim_interp, interp, lamb
):
    ##################################################################
    # TODO: 1.4: Implement LSGAN loss for discriminator.
    # Do not use discrim_interp, interp, lamb. They are placeholders
    # for Q1.5.
    ##################################################################
    ones = torch.ones_like(discrim_real)
    zeros = torch.zeros_like(discrim_fake)

    loss = torch.mean((discrim_real - ones) ** 2) + torch.mean((discrim_fake - zeros) ** 2)
    ##################################################################
    #                          END OF YOUR CODE                      #
    ##################################################################
    return loss/2


def compute_generator_loss(discrim_fake):
    ##################################################################
    # TODO: 1.4: Implement LSGAN loss for generator.
    ##################################################################
    ones = torch.ones_like(discrim_fake)
    loss = torch.mean((discrim_fake - ones) ** 2)
    ##################################################################
    #                          END OF YOUR CODE                      #
    ##################################################################
    return loss/2

if __name__ == "__main__":
    args = get_args()
    gen = Generator().cuda()
    disc = Discriminator().cuda()
    prefix = "data_ls_gan/"
    os.makedirs(prefix, exist_ok=True)

    train_model(
        gen,
        disc,
        num_iterations=int(3e4),
        batch_size=256,
        prefix=prefix,
        gen_loss_fn=compute_generator_loss,
        disc_loss_fn=compute_discriminator_loss,
        log_period=1000,
        amp_enabled=not args.disable_amp,
    )
