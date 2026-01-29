#!/bin/bash 

# uses FRCRN_SE_16K model for enhancing 16kHz speech waveforms
#network=FRCRN_SE_16K

#use MossFormer2_SE_48K model for enhancing 48kHz speech waveforms
network=MossFormer2_SE_48K

#use MossFormerGAN_SE_16K for enhancing 16kHz speech waveforms
#network=MossFormerGAN_SE_16K

config=config/inference/${network}.yaml
CUDA_VISIBLE_DEVICES=4 python3 -u inference.py --config $config
