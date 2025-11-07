import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import sys
import json
import traceback
import time
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

def _get_smiles_cached(name: str) -> Optional[str]:
    """Get SMILES with caching to avoid redundant database queries."""
    if name in _smiles_cache:
        return _smiles_cache[name]
    try:
        smiles = get_smiles_from_reaction(name)
        if len(_smiles_cache) >= _max_cache_size:
            # Clear half of the cache when it gets too large
            keys_to_remove = list(_smiles_cache.keys())[:_max_cache_size // 2]
            for key in keys_to_remove:
                del _smiles_cache[key]
        _smiles_cache[name] = smiles
        return smiles
    except Exception:
        return None

def _process_molecule_for_filtering(name: str, seen_inchikeys: set) -> Optional[Tuple[str, str]]:
    """Process a single molecule for filtering. Returns (name, inchikey) if valid, None otherwise."""
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
        return (name, key)
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
    save_all_scores: bool = False
) -> None:
    """
    Infinite loop, runs until orchestrator kills it:
      1) Sample n molecules
      2) Score them
      3) Merge with previous top x, deduplicate, sort, select top x
      4) Write top x to file (overwrite) each iteration
    """
    n_samples = config["num_molecules"] * 5

    top_pool = pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])
    seen_inchikeys = set()

    iteration = 0
    mutation_prob = 0.1
    elite_frac = 0.5

    while True:
        iteration += 1
        bt.logging.info(f"[Miner] Iteration {iteration}: sampling {n_samples} molecules")

        sampler_data = run_sampler(n_samples=n_samples, 
                        subnet_config=config, 
                        output_path=sampler_file_path,
                        save_to_file=True,
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
            filtered_names = []
            new_inchikeys = set()
            
            # Process molecules in parallel batches
            batch_size = min(100, len(names))
            with ThreadPoolExecutor(max_workers=min(8, len(names))) as executor:
                futures = {executor.submit(_process_molecule_for_filtering, name, seen_inchikeys): name 
                          for name in names}
                
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        name, key = result
                        filtered_names.append(name)
                        new_inchikeys.add(key)

            if not filtered_names:
                bt.logging.warning("All sampled molecules were previously seen; continuing")
                continue
            
            # Update seen_inchikeys with new ones
            seen_inchikeys.update(new_inchikeys)

            if len(filtered_names) < len(names):
                bt.logging.warning(f"{len(names) - len(filtered_names)} molecules were previously seen; continuing with unseen only")

            dup_ratio = (len(names) - len(filtered_names)) / max(1, len(names))
            if dup_ratio > 0.6:
                mutation_prob = min(0.5, mutation_prob * 1.5)
                elite_frac = max(0.2, elite_frac * 0.8)
            elif dup_ratio < 0.2 and not top_pool.empty:
                mutation_prob = max(0.05, mutation_prob * 0.9)
                elite_frac = min(0.8, elite_frac * 1.1)

            sampler_data = {"molecules": filtered_names}
            with open(sampler_file_path, "w") as f:
                json.dump(sampler_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            bt.logging.warning(f"[Miner] Pre-score deduplication failed; proceeding unfiltered: {e}")

        score_dict = score_molecules_json(sampler_file_path, 
                                         config["target_sequences"], 
                                         config["antitarget_sequences"], 
                                         config)
        
        if not score_dict:
            bt.logging.warning("[Miner] Scoring failed or mismatched; continuing")
            continue

        # Calculate final scores per molecule
        batch_scores = calculate_final_scores(score_dict, sampler_data, config, save_all_scores)

        try:
            seen_inchikeys.update([k for k in batch_scores["InChIKey"].tolist() if k])
        except Exception:
            pass

        # Merge, deduplicate, sort and take top x
        # Optimize: Use list append instead of concat for better performance
        if top_pool.empty:
            top_pool = batch_scores.copy()
        else:
            top_pool = pd.concat([top_pool, batch_scores], ignore_index=True)
        
        # Drop duplicates and sort in one go, then take top
        top_pool = top_pool.drop_duplicates(subset=["InChIKey"], keep="first")
        top_pool = top_pool.nlargest(config["num_molecules"], "score",keep="first")

        # format to accepted format
        top_entries = {"molecules": top_pool["name"].tolist()}

        # write to file
        with open(output_path, "w") as f:
            json.dump(top_entries, f, ensure_ascii=False, indent=2)

        bt.logging.info(f"[Miner] Wrote {config['num_molecules']} top molecules to {output_path}")
        bt.logging.info(f"[Miner] Average score: {top_pool['score'].mean()}")

def calculate_final_scores(score_dict: dict, 
        sampler_data: dict, 
        config: dict, 
        save_all_scores: bool = True,
        current_epoch: int = 0) -> pd.DataFrame:
    """
    Calculate final scores per molecule
    """

    names = sampler_data["molecules"]
    # Use cached SMILES lookup
    smiles = [_get_smiles_cached(name) for name in names]

    # Calculate InChIKey for each molecule in parallel
    inchikey_list = [None] * len(smiles)
    
    # Process InChIKey calculation in parallel batches
    with ThreadPoolExecutor(max_workers=min(8, len(smiles))) as executor:
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
    targets = score_dict[0]['target_scores']
    antitargets = score_dict[0]['antitarget_scores']
    
    # Convert to numpy arrays for vectorized operations
    # targets is a list of lists, where each inner list has scores for all molecules
    targets_array = np.array(targets)  # Shape: (num_targets, num_molecules)
    antitargets_array = np.array(antitargets)  # Shape: (num_antitargets, num_molecules)
    
    # Calculate averages using numpy (mean along axis 0 = across targets/antitargets)
    avg_targets = np.mean(targets_array, axis=0)  # Shape: (num_molecules,)
    avg_antitargets = np.mean(antitargets_array, axis=0)  # Shape: (num_molecules,)
    
    # Calculate final scores vectorized
    final_scores = (avg_targets - config["antitarget_weight"] * avg_antitargets).tolist()

    # Store final scores in dataframe
    batch_scores = pd.DataFrame({
        "name": names,
        "smiles": smiles,
        "InChIKey": inchikey_list,
        "score": final_scores
    })

    if save_all_scores:
        all_scores = {"scored_molecules": [(mol["name"], mol["score"]) for mol in batch_scores.to_dict(orient="records")]}
        all_scores_path = os.path.join(OUTPUT_DIR, f"all_scores_{current_epoch}.json")
        if os.path.exists(all_scores_path):
            with open(all_scores_path, "r") as f:
                all_previous_scores = json.load(f)
            all_scores["scored_molecules"] = all_previous_scores["scored_molecules"] + all_scores["scored_molecules"]
        with open(all_scores_path, "w") as f:
            json.dump(all_scores, f, ensure_ascii=False, indent=2)

    return batch_scores

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
