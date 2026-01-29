import torch
import torch.nn as nn
import torch.nn.functional as F

class network_wrapper(nn.Module):
    def __init__(self, args):
        super(network_wrapper, self).__init__()
        self.args = args
        if args.network == 'FRCRN_SE_16K':
            from models.frcrn.frcrn import FRCRN_SE_16K
            self.se_network = FRCRN_SE_16K(args).model
        elif args.network == 'FRCRNLite_SE_16K':
            from models.frcrnlite.frcrn import FRCRN_SE_16K
            self.se_network = FRCRN_SE_16K(args).model
        elif args.network == 'LCRNET_SE_16K':
            from models.lcrnet.frcrn import FRCRN_SE_16K
            self.se_network = FRCRN_SE_16K(args).model
        elif args.network == 'MossFormer2_SE_48K':
            from models.mossformer2.mossformer2_se_wrapper import MossFormer2_SE_48K
            self.se_network = MossFormer2_SE_48K(args).model
        elif args.network == 'MossFormerGAN_SE_16K':
            from models.mossformer_gan.generator import MossFormerGAN_SE_16K
            self.se_network = MossFormerGAN_SE_16K(args).model
            self.discriminator = MossFormerGAN_SE_16K(args).discriminator
        else:
            print("No network found!")
            return

    def forward(self, mixture, visual=None):
        est_source = self.se_network(mixture)
        return est_source
