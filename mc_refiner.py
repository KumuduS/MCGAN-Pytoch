import os
import glob
import math
import shutil
import numpy as np
import librosa
from typing import List, Tuple, Dict, Optional

# =========================
# 1) Feature extraction
# =========================
def extract_mfcc_mean(file_path: str,
                      sr: int = 16384,
                      duration: float = 1.0,
                      n_mfcc: int = 20,
                      n_fft: int = 256,
                      hop_length: int = 128) -> Optional[np.ndarray]:
    try:
        y, _ = librosa.load(file_path, sr=sr, mono=True)
    except Exception as e:
        print(f"[WARN] Failed to load {file_path}: {e}")
        return None

    seg_len = int(duration * sr)
    if len(y) < seg_len:
        if seg_len - len(y) <= sr * 0.1:
            y = np.pad(y, (0, seg_len - len(y)))
        else:
            return None

    y = y[:seg_len]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc,
                                n_fft=n_fft, hop_length=hop_length)
    return np.mean(mfcc, axis=1)


def batch_extract_mfcc_means(wav_dir: str, pattern: str = "*.wav", **kwargs) -> Tuple[np.ndarray, List[str]]:
    paths = sorted(glob.glob(os.path.join(wav_dir, pattern)))
    feats, kept_paths = [], []
    for p in paths:
        v = extract_mfcc_mean(p, **kwargs)
        if v is not None:
            feats.append(v)
            kept_paths.append(p)
    if len(feats) == 0:
        raise ValueError(f"No valid WAV features in {wav_dir}")
    return np.vstack(feats), kept_paths


# =========================
# 2) Prior over deltas
# =========================
def compute_deltas(seq: np.ndarray) -> np.ndarray:
    return seq[1:] - seq[:-1]


def gaussian_logpdf(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
    D = mean.shape[0]
    eps = 1e-6
    cov_reg = cov + eps * np.eye(D)
    try:
        L = np.linalg.cholesky(cov_reg)
    except np.linalg.LinAlgError:
        vals, vecs = np.linalg.eigh(cov_reg)
        vals = np.clip(vals, eps, None)
        inv_cov = (vecs @ np.diag(1.0 / vals) @ vecs.T)
        logdet = np.sum(np.log(vals))
        diff = (x - mean)
        return -0.5 * (D * np.log(2 * np.pi) + logdet + diff.T @ inv_cov @ diff)

    diff = (x - mean)
    y = np.linalg.solve(L, diff)
    logdet = 2.0 * np.sum(np.log(np.diag(L)))
    return -0.5 * (D * np.log(2 * np.pi) + logdet + y.T @ y)


def fit_gaussian_prior_from_real(real_feats: np.ndarray,
                                 random_state: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    deltas = compute_deltas(real_feats)
    if len(deltas) == 0:
        raise ValueError("Not enough samples for deltas.")
    mu = deltas.mean(axis=0)
    cov = np.cov(deltas.T)
    return mu, cov


# =========================
# 3) MCMC Refinement (MH algorithm)
# =========================
def ema_smooth(seq: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    print("EMA Smoothing alpha: ", alpha)
    out = np.zeros_like(seq)
    out[0] = seq[0]
    for t in range(1, len(seq)):
        out[t] = alpha * seq[t] + (1 - alpha) * out[t - 1]
    return out


def mcmc_refine_feature_path(real_feats: np.ndarray,
                             synth_feats: np.ndarray,
                             beta: float = 0.3,
                             target_len: int = 64,
                             prior_mean: Optional[np.ndarray] = None,
                             prior_cov: Optional[np.ndarray] = None,
                             ema_alpha: float = 0.2,
                             random_state: int = 42) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(random_state)
    D = synth_feats.shape[1]
    T = target_len

    if prior_mean is None or prior_cov is None:
        prior_mean, prior_cov = fit_gaussian_prior_from_real(real_feats, random_state=random_state)

    # Initialize with nearest synthetic to first real
    r0 = real_feats[0]
    dists = np.linalg.norm(synth_feats - r0[None, :], axis=1)
    s_idx = int(np.argmin(dists))
    s_curr = synth_feats[s_idx].copy()
    accepted_features = [s_curr]
    accepted_indices = [s_idx]
    theta_curr = s_curr - r0  # initial delta

    # MH loop
    for t in range(1, T):
        r_t = real_feats[t % len(real_feats)]

        # Candidate: pick a random synthetic
        cand_idx = int(rng.integers(0, len(synth_feats)))
        v = synth_feats[cand_idx]

        theta_prop = (1.0 - beta) * (v - s_curr) + beta * (v - r_t)

        log_p_prop = gaussian_logpdf(theta_prop, prior_mean, prior_cov)
        log_p_curr = gaussian_logpdf(theta_curr, prior_mean, prior_cov)
        alpha = min(1.0, np.exp(log_p_prop - log_p_curr))

        if rng.uniform() < alpha:
            s_curr = v
            s_idx = cand_idx
            theta_curr = theta_prop

        accepted_features.append(s_curr)
        accepted_indices.append(s_idx)

    accepted_features = np.vstack(accepted_features)
    smoothed_features = ema_smooth(accepted_features, alpha=ema_alpha)

    return {
        "accepted_features": accepted_features,
        "smoothed_features": smoothed_features,
        "accepted_indices": np.array(accepted_indices, dtype=int)
    }


# =========================
# 4) Nearest Neighbor Mapping
# =========================
def nearest_neighbor_indices(targets: np.ndarray, pool: np.ndarray) -> np.ndarray:
    a2 = np.sum(targets**2, axis=1, keepdims=True)
    b2 = np.sum(pool**2, axis=1, keepdims=True).T
    ab = targets @ pool.T
    d2 = a2 + b2 - 2.0 * ab
    return np.argmin(d2, axis=1)


def write_refined_wavs(selected_indices: np.ndarray,
                       synth_paths: List[str],
                       out_dir: str,
                       prefix: str = "refined") -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    out_paths = []
    for t, idx in enumerate(selected_indices):
        src = synth_paths[int(idx)]
        base = f"{prefix}_{t:04d}.wav"
        dst = os.path.join(out_dir, base)
        shutil.copy2(src, dst)
        out_paths.append(dst)
    return out_paths


# =========================
# 5) End-to-end pipeline
# =========================
def run_mcmc_refinement(real_dir: str,
                        synth_dir: str,
                        out_dir: str,
                        target_len: int = 64,
                        beta: float = 0.3,
                        ema_alpha: float = 0.2,
                        sr: int = 16384,
                        duration: float = 1.0,
                        n_mfcc: int = 20,
                        n_fft: int = 256,
                        hop_length: int = 128,
                        random_state: int = 42) -> Dict[str, object]:
    print("[1/4] Extracting real features...")
    print("Real dir : ", real_dir)
    print("wavgegan dir :", synth_dir)
    print("out dir: ", out_dir)
    print("beta: ", beta)
    print("alpha: ", ema_alpha)
    print("taget_len:" , target_len)
    real_feats, _ = batch_extract_mfcc_means(real_dir, sr=sr, duration=duration,
                                             n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)

    print("[2/4] Extracting synthetic features...")
    synth_feats, synth_paths = batch_extract_mfcc_means(synth_dir, sr=sr, duration=duration,
                                                        n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)

    print("[3/4] Running MCMC refinement...")
    prior_mean, prior_cov = fit_gaussian_prior_from_real(real_feats, random_state=random_state)
    mcmc_out = mcmc_refine_feature_path(real_feats, synth_feats,
                                        beta=beta,
                                        target_len=target_len,
                                        prior_mean=prior_mean,
                                        prior_cov=prior_cov,
                                        ema_alpha=ema_alpha,
                                        random_state=random_state)

    print("[4/4] Mapping smoothed features to nearest WAVs...")
    nn_idx = nearest_neighbor_indices(mcmc_out["smoothed_features"], synth_feats)
    out_paths = write_refined_wavs(nn_idx, synth_paths, out_dir, prefix="refined")

    return {
        "refined_paths": out_paths,
        "accepted_indices": mcmc_out["accepted_indices"],
        "accepted_features": mcmc_out["accepted_features"],
        "smoothed_features": mcmc_out["smoothed_features"],
        "prior_mean": prior_mean,
        "prior_cov": prior_cov,
    }

def clear_output_dir(path: str):
    """Delete and recreate the output directory."""
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


# =========================
# Main
# =========================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MCMC Bee Audio Refinement")
    parser.add_argument("--real_dir", type=str)
    parser.add_argument("--synth_dir", type=str)
    parser.add_argument("--out_dir", type=str)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--ema_alpha", type=float, default=0.2)
    parser.add_argument("--target_len", type=int, default=64)
    args = parser.parse_args()
    
    clear_output_dir(args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    result = run_mcmc_refinement(real_dir=args.real_dir,
                                 synth_dir=args.synth_dir,
                                 out_dir=args.out_dir,
                                 target_len=args.target_len,
                                 beta=args.beta,
                                 ema_alpha=args.ema_alpha)

    print("✅ Refinement complete.")
    print("Refined clips saved to:", args.out_dir)
    print("Total:", len(result["refined_paths"]))
