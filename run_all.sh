#!/bin/bash

echo "Running VAE"
python3 VAE/TestGenerateToken.py
python3 VAE/TestTraining.py

echo "Running vanillaGAN"
python3 VAE/vanillaGAN/vanillaGAN.py

echo "Running WGAN"
python3 VAE/WGAN/wgan.py

echo "All scripts executed"
