# WaveGAN + Markov Chain Refinement for Bee Bioacoustics Synthesis (PyTorch)

PyTorch implementation of **WaveGAN** (Donahue et al. 2018) with an additional **Markov Chain Monte Carlo (MCMC) refinement step** to improve the quality of generated bee bioacoustic signals audio.  

This repository extends the WaveGAN approach by:
- Training a WaveGAN model to generate raw waveform audio (1 sec clips at 16kHz).
- Refining the generated samples with a **Metropolis–Hastings Markov Chain** to better align the synthetic distribution with real bee data.

---

## Features
- Train WaveGAN on arbitrary audio datasets (monophonic, multi-channel supported).
- Generate raw audio waveforms up to **1 second @ 16kHz**.
- Support for **GPU acceleration (CUDA 12.2, tested on NVIDIA L40S 46GB VRAM)**.
- MCMC refinement to **bridge the gap between generated and real distributions**.
- Automatic sample saving and model checkpointing.

---
## Requirements
Install Python 3.10+ and dependencies
Install other libraries : 
```bash
pip install -r requirements.txt
```
## Environment & Hardware Used
- OS: Ubuntu 22.04.5 LTS
- Python: 3.10 / 3.11
- GPU: NVIDIA L40S (46 GB VRAM)
- Driver: 535.216.01
- CUDA: 12.2

## Dataset
Datasets should be organized under a folder structure with audio files (e.g., WAV).
Examples:
> Dataset/QueenPresent_Data/
> Dataset/QueenAbsent_Data/

The params.py file points to your dataset:
```bash
target_signals_dir = 'Dataset/QueenPresent_Data'
```
## Parameters (params.py)

Key training parameters:

- n_iterations: number of training iterations (default 300000)
- lr_g / lr_d: learning rates for generator and discriminator
- beta1, beta2: Adam optimizer decay rates
- n_critic: discriminator update steps per generator update
- p_coeff: gradient penalty coefficient
- batch_size: training batch size (default 64)
- noise_latent_dim: latent noise vector size (default 100)
- model_capacity_size: controls model size (use 32 for 2–4s windows)
- window_length: output clip length [16384 (1s), 32768 (2s), 65536 (4s)]
- sampling_rate: audio sample rate (default 16384 Hz)
- backup_every_n_iters: checkpoint/save interval
- save_samples_every: sample saving interval

## Training Workflow
### Train WaveGAN
Train a WaveGAN model on your dataset
```bash
python3.11 train.py
```
This will:
- Save model checkpoints under wavegan_out/queen_absent/
- Save generated samples periodically (save_samples_every iterations)

### Refine with Markov Chain
Run MCMC refinement on WaveGAN generated samples and actual samples:
```bash
python3.11 mcmc.py \
  --beta {b} \
  --ema_alpha {a} \
  --real_dir Dataset/QueenAbsent_Data/train \
  --synth_dir wavegan_out/queen_present \
  --out_dir mc_refined/queen_present \
  --target_len 20000
```
Where:

- --beta → Controls the proposal influence in the Metropolis–Hastings algorithm.
  - Lower values make the chain rely more on the previous synthetic sample, creating smoother transitions.
  - Higher values make the chain propose candidates more aggressively from real sample differences, increasing adaptation to real data but may introduce more variance.
  - Typical values: 0.01–0.1 for smooth refinement.
- --ema_alpha → exponential moving average smoothing factor
  - Controls the Exponential Moving Average (EMA) smoothing applied to the accepted features.
  - Values closer to 1.0 give more weight to the latest accepted sample, preserving dynamic changes.
  - Values closer to 0.0 give more weight to past samples, resulting in smoother, less jittery outputs.
  - A value around 0.5 balances between smoothing and responsiveness. In practice, 0.95 works well to retain natural variability while reducing abrupt jumps.

- --real_dir → directory with real training data
- --synth_dir → directory with WaveGAN-generated samples
- --out_dir → output directory for MC-refined audio
- --target_len → number of samples to generate/refine

### Samples
- Generated audio samples and checkpoints are automatically saved in the output directory

## Contributions
This work builds upon and is inspired by the following projects:

- chrisdonahue/wavegan - the original WaveGAN implementation (Donahue et al. 2018), which introduced the architecture and methods for synthesizing raw audio waveforms using GANs.
- mostafaelaraby/wavegan-pytorch - a PyTorch port of WaveGAN with support for training on longer audio clips (up to 4 seconds), multiple channels, and various improvements/adaptations to make it more flexible.

This repository extends their work with an additional Markov Chain refinement step.

## Citation

If you use this code in research, please cite:
```bibtex
@inproceedings{donahue2019wavegan,
  title={Adversarial Audio Synthesis},
  author={Donahue, Chris and McAuley, Julian and Puckette, Miller},
  booktitle={ICLR},
  year={2019}
}


