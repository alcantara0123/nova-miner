import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple, Optional

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PARENT_DIR)

from rdkit import Chem
from rdkit.Chem import Descriptors

from nova_ph2.utils import (
    get_smiles, 
    get_heavy_atom_count, 
    compute_maccs_entropy
)

def _validate_single_molecule(molecule: str, min_heavy_atoms: int, 
                              min_rotatable_bonds: int, max_rotatable_bonds: int) -> Optional[Tuple[str, str]]:
    """
    Validate a single molecule. Returns (name, smiles) if valid, None otherwise.
    Optimized to parse SMILES only once.
    """
    if molecule is None:
        return None
    
    try:
        smiles = get_smiles(molecule)
        if not smiles:
            return None
        
        # Parse SMILES once and reuse the molecule object
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Check heavy atom count using the molecule object (more efficient)
        heavy_atom_count = mol.GetNumHeavyAtoms()
        if heavy_atom_count < min_heavy_atoms:
            return None
        
        # Check rotatable bonds using the already-parsed molecule
        num_rotatable_bonds = Descriptors.NumRotatableBonds(mol)
        if num_rotatable_bonds < min_rotatable_bonds or num_rotatable_bonds > max_rotatable_bonds:
            return None
        
        return (molecule, smiles)
    except Exception:
        return None

def validate_molecules_sampler(
    sampler_data: dict[int, dict[str, list]],
    config: dict,
) -> dict[int, dict[str, list[str]]]:
    """
    Validates molecules for all random sampler (uid=0).    
    Doesn't interrupt the process if a molecule is invalid, removes it from the list instead. 
    Doesn't check allowed reactions, chemically identical, duplicates, uniqueness (handled in random_sampler.py)
    
    Optimized with:
    - Single SMILES parsing per molecule
    - Parallel processing for large batches
    - Pre-extracted config values
    
    Args:
        sampler_data: Dictionary containing molecules to validate
        config: Configuration dictionary containing validation parameters
        
    Returns:
        Tuple of (valid_names, valid_smiles) lists
    """
    
    molecules = sampler_data["molecules"]
    
    # Pre-extract config values to avoid repeated dict lookups
    min_heavy_atoms = config['min_heavy_atoms']
    min_rotatable_bonds = config['min_rotatable_bonds']
    max_rotatable_bonds = config['max_rotatable_bonds']
    
    # Use parallelization for large batches (threshold: 50 molecules)
    # This balances overhead vs speedup
    if len(molecules) >= 50:
        valid_names = []
        valid_smiles = []
        
        # Use ThreadPoolExecutor for parallel validation
        max_workers = min(8, max(2, len(molecules) // 25))  # ~25 molecules per worker
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_validate_single_molecule, mol, min_heavy_atoms, 
                               min_rotatable_bonds, max_rotatable_bonds): mol
                for mol in molecules
            }
            
            for future in as_completed(futures):
                result = future.result()
                if result:
                    name, smiles = result
                    valid_names.append(name)
                    valid_smiles.append(smiles)
    else:
        # Sequential processing for small batches (lower overhead)
        valid_names = []
        valid_smiles = []
        
        for molecule in molecules:
            result = _validate_single_molecule(molecule, min_heavy_atoms, 
                                             min_rotatable_bonds, max_rotatable_bonds)
            if result:
                name, smiles = result
                valid_names.append(name)
                valid_smiles.append(smiles)
        
    return valid_names, valid_smiles