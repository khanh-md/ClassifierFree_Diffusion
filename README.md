# ClassifierFree_Diffusion

A PyTorch implementation of **Classifier-Free Diffusion Guidance** trained on the FashionMNIST dataset. This project implements a Denoising Diffusion Probabilistic Model (DDPM) with classifier-free guidance, allowing conditional image generation with controllable guidance strength.

Result:

![fashionMNIST](https://github.com/user-attachments/assets/99bcd828-8756-48cb-aaa2-c25ba304ec15)

---

## Overview

Diffusion models generate images by learning to reverse a gradual noising process. This implementation extends the standard DDPM framework with **classifier-free guidance**, which enables class-conditional generation without requiring a separate classifier at inference time. Instead, the model is trained with randomly dropped class labels, allowing it to learn both conditional and unconditional noise prediction. At inference time, the two predictions are interpolated using a guidance weight `w` to control how strongly the output adheres to the target class.

---

## How It Works

**Forward process:** Gaussian noise is gradually added to an image over `T` timesteps according to a learned noise schedule `β`.

**Reverse process:** A UNet-based model learns to predict the noise added at each timestep, conditioned on both the timestep and (optionally) a class label.

**Classifier-free guidance:** At inference, the model produces two noise estimates — one with the class condition and one without. The final noise estimate is computed as:

```
ε = (1 + w) * ε_cond - w * ε_uncond
```

where `w` controls the guidance strength. Higher `w` produces more class-faithful but less diverse samples.

---

## File Structure

```
├── Classifier_Free_Diffusion.ipynb  # Main training and sampling notebook
├── ddpm_utils.py                    # DDPM forward/reverse diffusion logic
├── UNet_utils.py                    # UNet model architecture
├── other_utils.py                   # Visualization and helper utilities
```

---

## Requirements

```
torch
torchvision
matplotlib
numpy
```

---

## Usage

Open and run `Classifier_Free_Diffusion.ipynb` to train the model and sample images. The notebook walks through:

1. Setting up the noise schedule and DDPM class
2. Training the UNet with classifier-free guidance
3. Sampling images conditioned on FashionMNIST class labels
4. Visualizing the effect of different guidance weights `w`

---

## References

- Ho et al., [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) (2020)
- Ho & Salimans, [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) (2022)

