import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import sys
import json
import time
from pathlib import Path
from typing import Optional, Tuple, List
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from rdkit import Chem
import bittensor as bt

import nova_ph2
from random_sampler import run_sampler
from nova_ph2.neurons.validator.scoring import score_molecules_json
from nova_ph2.combinatorial_db.reactions import get_smiles_from_reaction

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PARENT_DIR)

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/output")

DB_PATH = str(Path(nova_ph2.__file__).resolve().parent / "combinatorial_db" / "molecules.sqlite")

# Settings
MAX_WORKERS = min(32, os.cpu_count() - 1)
WRITE_TIME_THRESHOLD_1 = 20 * 60
WRITE_TIME_THRESHOLD_2 = 25 * 60
WRITE_EARLY_INTERVAL = 2

def get_config(input_file: os.path = os.path.join(BASE_DIR, "input.json")):
    """
    Get config from input file
    """
    with open(input_file, "r") as f:
        d = json.load(f)
    config = {**d.get("config", {}), **d.get("challenge", {})}
    return config


# ----------------------------
# Caching + helper functions
# ----------------------------
@lru_cache(maxsize=10000)
def _get_smiles_from_reaction_cached(name: str) -> Optional[str]:
    try:
        return get_smiles_from_reaction(name)
    except Exception:
        return None

def _smiles_to_inchikey(smiles: Optional[str]) -> Optional[str]:
    try:
        if not smiles:
            return None
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToInchiKey(mol)
    except Exception:
        return None

def _process_name(name: str) -> Tuple[str, Optional[str], Optional[str]]:
    s = _get_smiles_from_reaction_cached(name)
    if not s:
        return (name, None, None)
    ik = _smiles_to_inchikey(s)
    return (name, s, ik)

def map_names_to_data(names: List[str], max_workers: int = MAX_WORKERS) -> List[Tuple[str, Optional[str], Optional[str]]]:
    """
    Uses threads (cheap, no pickling) to map names -> (name, smiles, inchikey)
    """
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_process_name, n): n for n in names}
        for fut in as_completed(futures):
            try:
                results.append(fut.result())
            except Exception:
                results.append((futures[fut], None, None))
    # preserve original order
    results_sorted = sorted(results, key=lambda x: names.index(x[0]))
    return results_sorted

# ----------------------------
# Score calculation
# ----------------------------
def calculate_final_scores(score_dict: dict, sampler_data: dict, config: dict, save_all_scores: bool = True, current_epoch: int = 0) -> pd.DataFrame:
    names = sampler_data.get("molecules", [])
    if not names:
        return pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])

    precomputed = sampler_data.get("_precomputed", None)
    if precomputed is not None:
        smiles = [t[1] for t in precomputed]
        inchikeys = [t[2] for t in precomputed]
    else:
        tuples = map_names_to_data(names)
        smiles = [t[1] for t in tuples]
        inchikeys = [t[2] for t in tuples]

    targets = np.array(score_dict[0]["target_scores"], dtype=np.float32)
    antitargets = np.array(score_dict[0]["antitarget_scores"], dtype=np.float32)

    final_scores = np.mean(targets, axis=0) - config.get("antitarget_weight", 1.0) * np.mean(antitargets, axis=0)

    df = pd.DataFrame({
        "name": names,
        "smiles": smiles,
        "InChIKey": inchikeys,
        "score": final_scores
    })

    return df

# ----------------------------
# Main loop
# ----------------------------
def iterative_sampling_loop(db_path: str, sampler_file_path: str, output_path: str, config: dict, save_all_scores: bool = True):
    n_samples = config.get("num_molecules", 100) * 5

    top_pool = pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])
    seen_inchikeys = set()
    iteration = 0
    mutation_prob = 0.1
    elite_frac = 0.5
    start_time = time.time()
    total_time = 0
    total_samples=0
    target_time=100.0
    min_samples=100
    max_samples=5000
    score_improvement=0.0
    prev_mean_score: Optional[float] = None

    while True:
        iter_start = time.time()
        iteration += 1

        # ---- Adaptive request sample size ----
        if iteration == 1:
            req_samples = n_samples * 2  # initial exploration burst
        else:
            req_samples = n_samples

        bt.logging.info(f"[Miner] Iteration {iteration}: sampling {n_samples} molecules")
        total_samples+=req_samples
        sampler_data = run_sampler(
            n_samples=n_samples * 4 if iteration == 1 else n_samples,
            subnet_config=config,
            output_path=sampler_file_path,
            save_to_file=True,
            db_path=db_path,
            elite_names=top_pool["name"].tolist() if not top_pool.empty else None,
            elite_frac=elite_frac,
            mutation_prob=mutation_prob,
            avoid_inchikeys=seen_inchikeys,
        )

        if not sampler_data or not sampler_data.get("molecules"):
            bt.logging.warning("[Miner] No valid molecules; continuing")
            continue

        # Precompute SMILES/InChIKeys
        precomputed = map_names_to_data(sampler_data["molecules"])

        # Filter unseen
        filtered = [(n, s, ik) for n, s, ik in precomputed if s and ik and ik not in seen_inchikeys]
        if not filtered:
            bt.logging.warning("[Miner] All molecules already seen; skipping")
            continue

        seen_inchikeys.update([ik for _, _, ik in filtered])
        filtered_names = [t[0] for t in filtered]

        sampler_payload = {"molecules": filtered_names, "_precomputed": filtered}

        with open(sampler_file_path, "w") as f:
            json.dump({"molecules": filtered_names}, f, ensure_ascii=False, separators=(",", ":"))

        # GPU scoring
        score_dict = score_molecules_json(sampler_file_path, config["target_sequences"], config["antitarget_sequences"], config)
        if not score_dict:
            bt.logging.warning("[Miner] Scoring failed; skipping iteration")
            continue

        batch_scores = calculate_final_scores(score_dict, sampler_payload, config, save_all_scores, current_epoch=iteration)

        # Merge with top pool
        if not batch_scores.empty:
            top_pool = pd.concat([top_pool, batch_scores], ignore_index=True)
            top_pool = top_pool.sort_values(by="score", ascending=False).drop_duplicates(subset=["InChIKey"], keep="first")
            top_pool = top_pool.head(config.get("num_molecules", 100))

        # Write only if needed
        elapsed = time.time() - start_time
        should_write = False
        if elapsed >= WRITE_TIME_THRESHOLD_2:
            should_write = True
        elif elapsed >= WRITE_TIME_THRESHOLD_1 and iteration % WRITE_EARLY_INTERVAL == 0:
            should_write = True

        if should_write and not top_pool.empty:
            tmp = output_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump({"molecules": top_pool["name"].tolist()}, f, ensure_ascii=False)
            os.replace(tmp, output_path)
            bt.logging.info(f"[Miner] Wrote top {len(top_pool)} molecules to {output_path}")

        current_mean = float(top_pool["score"].mean()) if not top_pool.empty else 0.0
        improvement_pct = 0.0
        if prev_mean_score is not None and prev_mean_score != 0:
            improvement_pct = (current_mean - prev_mean_score) / abs(prev_mean_score)

        if iteration > 3 and not top_pool.empty:
            score_improvement = max(0, improvement_pct)
            
            if score_improvement < 0.01:
                # Too many duplicates or stagnation → explore more
                mutation_prob = min(0.5, mutation_prob * 1.3)
                elite_frac = max(0.2, elite_frac * 0.85)
            elif score_improvement > 0.05:
                # High diversity + improving → exploit
                mutation_prob = max(0.05, mutation_prob * 0.9)
                elite_frac = min(0.8, elite_frac * 1.1)

        prev_mean_score = current_mean

        iter_end = time.time()
        ratio = (iter_end-iter_start) / target_time

        if iteration > 1:
            if ratio < 0.7:
                n_samples = int(min(n_samples * 1.5, max_samples))
            elif ratio > 1.3:
                n_samples = int(max(n_samples * 0.7, min_samples))

        total_time += iter_end - iter_start
        bt.logging.info(f"[Miner] Iter {iteration} complete | Avg score: {top_pool['score'].mean():.6f} | Iter time: {iter_end - iter_start:.2f}s | Total time: {total_time:.2f}s Total samples: {total_samples}")
        if total_time>=1000: break

def main(config: dict):
    iterative_sampling_loop(
        db_path=DB_PATH,
        sampler_file_path=os.path.join(OUTPUT_DIR, "sampler_file.json"),
        output_path=os.path.join(OUTPUT_DIR, "result.json"),
        config=config,
        save_all_scores=True,
    )
 

if __name__ == "__main__":
    config = get_config()
    
    main(config)