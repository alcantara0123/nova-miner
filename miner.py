import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import sys
import json
import traceback
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import Optional, Tuple

import bittensor as bt
import pandas as pd
import numpy as np
from rdkit import Chem
from pathlib import Path
import nova_ph2

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PARENT_DIR)

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/output")

from nova_ph2.neurons.validator.scoring import score_molecules_json
import nova_ph2.neurons.validator.scoring as scoring_module
from random_sampler import run_sampler
from nova_ph2.combinatorial_db.reactions import get_smiles_from_reaction

DB_PATH = str(Path(nova_ph2.__file__).resolve().parent / "combinatorial_db" / "molecules.sqlite")

# Cache for SMILES lookups to avoid redundant database queries
_smiles_cache = {}
_max_cache_size = 10000
_cache_lock = threading.Lock()  # Lock for thread-safe cache operations

def _get_smiles_cached(name: str) -> Optional[str]:
    """Get SMILES with caching to avoid redundant database queries."""
    # Check cache with lock to avoid race condition
    with _cache_lock:
        if name in _smiles_cache:
            return _smiles_cache[name]
    
    # Cache miss - fetch from database (outside lock to avoid blocking)
    try:
        smiles = get_smiles_from_reaction(name)
        if smiles is None:
            return None
        
        # Cache write with lock
        with _cache_lock:
            # Double-check after acquiring lock (another thread might have added it)
            if name not in _smiles_cache:
                if len(_smiles_cache) >= _max_cache_size:
                    # Clear half of the cache when it gets too large
                    # Use list slicing for efficiency
                    keys_to_remove = list(_smiles_cache.keys())[:_max_cache_size // 2]
                    for key in keys_to_remove:
                        del _smiles_cache[key]
                _smiles_cache[name] = smiles
            else:
                # Another thread added it, use cached value
                smiles = _smiles_cache[name]
        return smiles
    except Exception:
        return None

def _process_molecule_for_filtering(name: str, seen_inchikeys: set) -> Optional[Tuple[str, str, str]]:
    """Process a single molecule for filtering. Returns (name, smiles, inchikey) if valid, None otherwise.
    Returns smiles to avoid redundant lookups later."""
    try:
        s = _get_smiles_cached(name)
        if not s:
            return None
        mol = Chem.MolFromSmiles(s)
        if not mol:
            return None
        key = Chem.MolToInchiKey(mol)
        if key in seen_inchikeys:
            return None
        return (name, s, key)
    except Exception:
        return None

def _process_molecule_for_inchikey(smiles: str) -> Optional[str]:
    """Process a single SMILES to get InChIKey."""
    try:
        if not smiles:
            return None
        return Chem.MolToInchiKey(Chem.MolFromSmiles(smiles))
    except Exception:
        return None

def get_config(input_file: os.path = os.path.join(BASE_DIR, "input.json")):
    """
    Get config from input file
    """
    with open(input_file, "r") as f:
        d = json.load(f)
    config = {**d.get("config", {}), **d.get("challenge", {})}
    return config

def iterative_sampling_loop(
    db_path: str,
    sampler_file_path: str,
    output_path: str,
    config: dict,
) -> None:
    """
    Infinite loop, runs until orchestrator kills it:
      1) Sample n molecules
      2) Score them
      3) Merge with previous top x, deduplicate, sort, select top x
      4) Write top x to file only near the end of 30 minutes (to reduce I/O overhead)
    """
    n_samples = config["num_molecules"] * 5

    top_pool = pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])
    seen_inchikeys = set()

    iteration = 0
    mutation_prob = 0.1
    elite_frac = 0.5
    
    # Time tracking for periodic file writes
    start_time = time.time()
    write_interval = 5 * 60  # Write every 5 minutes as safety backup
    target_time = 30 * 60  # 30 minutes target
    next_write_time = start_time + write_interval

    while True:
        iteration += 1
        bt.logging.info(f"[Miner] Iteration {iteration}: sampling {n_samples} molecules")

        sampler_data = run_sampler(n_samples=n_samples, 
                        subnet_config=config, 
                        output_path=sampler_file_path,
                        save_to_file=False,  # Skip redundant write - we'll write filtered data below
                        db_path=db_path,
                        elite_names=top_pool["name"].tolist() if not top_pool.empty else None,
                        elite_frac=elite_frac,
                        mutation_prob=mutation_prob,
                        avoid_inchikeys=seen_inchikeys,
                        )
        
        if not sampler_data:
            bt.logging.warning("[Miner] No valid molecules produced; continuing")
            continue

        try:
            names = sampler_data["molecules"]
            # Parallelize molecule filtering using ThreadPoolExecutor
            # Store (name, smiles, inchikey) tuples to avoid redundant lookups
            filtered_data = []  # List of (name, smiles, inchikey) tuples
            new_inchikeys = set()
            
            # Use more workers to keep GPU busy while CPU processes molecules
            # Aim for ~50 molecules per worker to balance overhead vs parallelism
            max_workers = max(1, min(16, max(4, len(names) // 50)))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(_process_molecule_for_filtering, name, seen_inchikeys): name 
                          for name in names}
                
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        name, smiles, key = result
                        filtered_data.append((name, smiles, key))
                        new_inchikeys.add(key)

            if not filtered_data:
                bt.logging.warning("All sampled molecules were previously seen; continuing")
                continue
            
            # Update seen_inchikeys with new ones (only once)
            seen_inchikeys.update(new_inchikeys)

            filtered_names = [item[0] for item in filtered_data]
            if len(filtered_names) < len(names):
                bt.logging.warning(f"{len(names) - len(filtered_names)} molecules were previously seen; continuing with unseen only")

            dup_ratio = (len(names) - len(filtered_names)) / max(1, len(names))
            if dup_ratio > 0.62:
                mutation_prob = min(0.5, mutation_prob * 1.5)
                elite_frac = max(0.2, elite_frac * 0.8)
            elif dup_ratio < 0.22 and not top_pool.empty:
                mutation_prob = max(0.05, mutation_prob * 0.9)
                elite_frac = min(0.8, elite_frac * 1.1)

            # Store filtered data for later use to avoid redundant SMILES lookups
            sampler_data = {"molecules": filtered_names}
            # Store pre-computed smiles and inchikeys in a temporary structure
            # We'll use this in calculate_final_scores to avoid redundant work
            sampler_data["_filtered_data"] = filtered_data  # Internal use only
            
            # Write filtered molecules to file - required for score_molecules_json
            with open(sampler_file_path, "w") as f:
                json.dump({"molecules": filtered_names}, f, ensure_ascii=False, indent=2)
        except Exception as e:
            bt.logging.warning(f"[Miner] Pre-score deduplication failed; proceeding unfiltered: {e}")
            # Clear _filtered_data if it exists to avoid issues
            if "_filtered_data" in sampler_data:
                del sampler_data["_filtered_data"]
            # Still need to write file for score_molecules_json even if filtering failed
            with open(sampler_file_path, "w") as f:
                json.dump(sampler_data, f, ensure_ascii=False, indent=2)

        score_dict = score_molecules_json(sampler_file_path, 
                                         config["target_sequences"], 
                                         config["antitarget_sequences"], 
                                         config)
        
        if not score_dict:
            bt.logging.warning("[Miner] Scoring failed or mismatched; continuing")
            continue

        # Validate score_dict structure before accessing
        if not isinstance(score_dict, dict) or 0 not in score_dict:
            bt.logging.error(f"[Miner] Invalid score_dict structure: {type(score_dict)}")
            continue

        # Calculate final scores per molecule
        # Use iteration as current_epoch to track scoring iterations
        batch_scores = calculate_final_scores(score_dict, sampler_data, config, current_epoch=iteration)

        # Note: seen_inchikeys was already updated during filtering, so no need to update again here
        # This avoids redundant work

        # Merge, deduplicate, sort and take top x
        # Use more efficient single-pass dedup and sort
        if top_pool.empty:
            top_pool = batch_scores.copy(deep=False)
        else:
            # Concatenate without copying
            top_pool = pd.concat([top_pool, batch_scores], ignore_index=True)
        
        # Single optimized pass: filter NaN, drop duplicates, and get top N
        # This is much faster than multiple separate operations
        top_pool = (top_pool[top_pool["InChIKey"].notna()]
                    .drop_duplicates(subset=["InChIKey"], keep="first")
                    .nlargest(config["num_molecules"], "score"))
        
        # Reset index for consistency
        top_pool.reset_index(drop=True, inplace=True)

        # Check elapsed time and decide whether to write file
        current_time = time.time()
        elapsed_time = current_time - start_time
        time_remaining = target_time - elapsed_time
        
        # Write to file if:
        # 1. Approaching 30 minutes (within last 2 minutes), OR
        # 2. Periodic safety write (every 5 minutes), OR
        # 3. Past 30 minutes (shouldn't happen, but safety)
        should_write = (
            time_remaining <= 2 * 60 or  # Within 2 minutes of 30-minute target
            current_time >= next_write_time or  # Periodic safety write
            elapsed_time >= target_time  # Past target time
        )
        
        if should_write:
            # format to accepted format
            top_entries = {"molecules": top_pool["name"].tolist()}

            # write to file
            with open(output_path, "w") as f:
                json.dump(top_entries, f, ensure_ascii=False, indent=2)

            bt.logging.info(f"[Miner] Wrote {config['num_molecules']} top molecules to {output_path}")
            bt.logging.info(f"[Miner] Elapsed time: {elapsed_time/60:.1f} minutes, Time remaining: {max(0, time_remaining)/60:.1f} minutes")
            
            # Update next write time for periodic safety writes
            if current_time >= next_write_time:
                next_write_time = current_time + write_interval
        
        if not top_pool.empty:
            bt.logging.info(f"[Miner] Iteration {iteration} complete - Average score: {top_pool['score'].mean():.4f}, Pool size: {len(top_pool)}")
            if not should_write:
                bt.logging.info(f"[Miner] Skipped file write (elapsed: {elapsed_time/60:.1f} min, next write in: {max(0, next_write_time - current_time)/60:.1f} min)")

def calculate_final_scores(score_dict: dict, 
        sampler_data: dict, 
        config: dict, 
        current_epoch: int = 0) -> pd.DataFrame:
    """
    Calculate final scores per molecule.
    Optimized to use pre-computed SMILES and InChIKeys from filtering when available.
    """

    names = sampler_data["molecules"]
    
    # Check if we have pre-computed data from filtering to avoid redundant work
    if "_filtered_data" in sampler_data:
        # Use pre-computed smiles and inchikeys from filtering
        filtered_data = sampler_data["_filtered_data"]
        # Create a mapping for fast lookup
        data_map = {name: (smiles, inchikey) for name, smiles, inchikey in filtered_data}
        smiles = [data_map.get(name, (None, None))[0] for name in names]
        inchikey_list = [data_map.get(name, (None, None))[1] for name in names]
        
        # Fill in any missing values (shouldn't happen, but be safe)
        missing_indices = [i for i, s in enumerate(smiles) if s is None]
        if missing_indices:
            max_workers = max(1, min(16, max(4, len(missing_indices) // 50)))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_idx = {executor.submit(_get_smiles_cached, names[i]): i 
                                 for i in missing_indices}
                
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        smiles[idx] = future.result()
                        if smiles[idx]:
                            inchikey_list[idx] = _process_molecule_for_inchikey(smiles[idx])
                    except Exception as e:
                        bt.logging.error(f"Error fetching SMILES for {names[idx]}: {e}")
    else:
        # Fallback: parallelize SMILES lookups to keep CPU busy while GPU scores
        smiles = [None] * len(names)
        max_workers = max(1, min(16, max(4, len(names) // 50)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {executor.submit(_get_smiles_cached, name): idx 
                             for idx, name in enumerate(names)}
            
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    smiles[idx] = future.result()
                except Exception as e:
                    bt.logging.error(f"Error fetching SMILES for {names[idx]}: {e}")

        # Calculate InChIKey for each molecule in parallel
        inchikey_list = [None] * len(smiles)
        
        # Process InChIKey calculation in parallel batches
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {executor.submit(_process_molecule_for_inchikey, s): idx 
                             for idx, s in enumerate(smiles)}
            
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    inchikey_list[idx] = future.result()
                except Exception as e:
                    if smiles[idx]:
                        bt.logging.error(f"Error calculating InChIKey for {smiles[idx]}: {e}")

    # Vectorize score calculations using numpy
    # Validate score_dict structure before accessing
    if not isinstance(score_dict, dict) or 0 not in score_dict:
        bt.logging.error(f"[Miner] Invalid score_dict: missing key 0 or not a dict")
        return pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])
    
    score_data = score_dict[0]
    if not isinstance(score_data, dict):
        bt.logging.error(f"[Miner] Invalid score_dict[0]: not a dict")
        return pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])
    
    targets = score_data.get('target_scores')
    antitargets = score_data.get('antitarget_scores')
    
    # Validate that targets and antitargets have the correct structure
    num_molecules = len(names)
    if not targets or not isinstance(targets, list):
        bt.logging.error("[Miner] Invalid target_scores: empty or not a list")
        return pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])
    
    if not antitargets or not isinstance(antitargets, list):
        bt.logging.error("[Miner] Invalid antitarget_scores: empty or not a list")
        return pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])
    
    # Validate that each target/antitarget list has the same length as names
    if any(len(t) != num_molecules for t in targets):
        bt.logging.error(f"[Miner] Target scores length mismatch: expected {num_molecules} molecules")
        return pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])
    
    if any(len(a) != num_molecules for a in antitargets):
        bt.logging.error(f"[Miner] Antitarget scores length mismatch: expected {num_molecules} molecules")
        return pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])
    
    # Convert to numpy arrays for vectorized operations
    # targets is a list of lists, where each inner list has scores for all molecules
    targets_array = np.array(targets, dtype=np.float32)  # Shape: (num_targets, num_molecules)
    antitargets_array = np.array(antitargets, dtype=np.float32)  # Shape: (num_antitargets, num_molecules)
    
    # Calculate averages using numpy (mean along axis 0 = across targets/antitargets)
    avg_targets = np.mean(targets_array, axis=0)  # Shape: (num_molecules,)
    avg_antitargets = np.mean(antitargets_array, axis=0)  # Shape: (num_molecules,)
    
    # Calculate final scores vectorized - keep as numpy array for performance
    final_scores = avg_targets - config["antitarget_weight"] * avg_antitargets

    # Store final scores in dataframe - use numpy array directly to avoid list conversion
    batch_scores = pd.DataFrame({
        "name": names,
        "smiles": smiles,
        "InChIKey": inchikey_list,
        "score": final_scores
    })


    return batch_scores

def main(config: dict):
    iterative_sampling_loop(
        db_path=DB_PATH,
        sampler_file_path=os.path.join(OUTPUT_DIR, "sampler_file.json"),
        output_path=os.path.join(OUTPUT_DIR, "result.json"),
        config=config,
    )
 

if __name__ == "__main__":
    config = get_config()
    main(config)
