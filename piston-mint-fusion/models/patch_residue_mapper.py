"""
Bridge between PIsToN surface patches and MINT sequence tokens.

Parses PIsToN's resnames.npy files to identify which amino acid residues
are covered by the interface patch, then maps those residues to the
corresponding token positions in MINT's concatenated sequence input.
"""

import numpy as np


def parse_resnames(resnames_path):
    """
    Parse a resnames.npy file produced by PIsToN's map_patch_atom.py.

    Each entry has the format: "chain:resid:resname-atomid:atomname"
    Example: "A:107:HIS-1621:CD2"

    Args:
        resnames_path: path to the .npy file (per-chain or grid-level)

    Returns:
        list of tuples: [(chain_id, resid_int, resname_3letter), ...]
        Entries that could not be parsed (e.g. 'x', 0, None) are skipped.
    """
    data = np.load(resnames_path, allow_pickle=True)
    parsed = []

    # Handle both 1-D (per-chain) and 2/3-D (grid-level) arrays
    flat = data.flatten()
    for entry in flat:
        if entry is None or entry == 0 or (isinstance(entry, str) and entry == "x"):
            continue
        try:
            entry_str = str(entry)
            # Format: "chain:resid:resname-atomid:atomname"
            parts = entry_str.split(":")
            chain_id = parts[0]
            resid = int(parts[1])
            resname = parts[2].split("-")[0]
            parsed.append((chain_id, resid, resname))
        except (IndexError, ValueError):
            continue

    return parsed


def get_unique_patch_residues(parsed_resnames):
    """
    From parsed resnames, extract unique residues per chain.

    Args:
        parsed_resnames: list of (chain_id, resid, resname) tuples

    Returns:
        dict mapping chain_id -> sorted list of unique residue numbers
    """
    residues_by_chain = {}
    for chain_id, resid, _ in parsed_resnames:
        if chain_id not in residues_by_chain:
            residues_by_chain[chain_id] = set()
        residues_by_chain[chain_id].add(resid)

    # Sort for reproducibility
    return {ch: sorted(resids) for ch, resids in residues_by_chain.items()}


def map_patch_residues_to_mint_tokens(
    patch_residues,
    sequence_info,
    chain_order,
):
    """
    Map PIsToN patch residues to MINT token indices.

    MINT token layout for a two-chain complex:
        [<cls>, A1, A2, ..., An, <eos>, <cls>, B1, B2, ..., Bm, <eos>, <pad>...]
        chain_ids:
        [  0,    0,  0, ...,  0,   0,     1,    1,  1, ...,  1,    1,    ...]

    For chain 0: residue at seq_position i -> token index = 1 + i  (skip <cls>)
    For chain 1: residue at seq_position j -> token index = (len_chain0 + 2) + 1 + j

    Args:
        patch_residues: dict[chain_id -> list[resid]] from get_unique_patch_residues()
        sequence_info: dict[chain_id -> {'sequence': str, 'residue_map_simple': {resid -> seq_pos}}]
                       from extract_sequences_from_pdb()
        chain_order: list of chain_ids in the order they are concatenated, e.g. ['A', 'B']

    Returns:
        dict[chain_id -> list[token_index]] for indexing into MINT's repr tensor
        (also returns the total token count for constructing the token tensor)
    """
    token_offset = 0
    result = {}
    total_tokens = 0

    for chain_idx, chain_id in enumerate(chain_order):
        if chain_id not in sequence_info:
            raise ValueError(f"Chain '{chain_id}' not found in sequence_info")

        seq_len = len(sequence_info[chain_id]["sequence"])
        chain_token_start = token_offset + 1  # skip <cls>

        token_indices = []
        if chain_id in patch_residues:
            residue_map = sequence_info[chain_id]["residue_map_simple"]
            for resid in patch_residues[chain_id]:
                if resid in residue_map:
                    seq_pos = residue_map[resid]
                    token_idx = chain_token_start + seq_pos
                    token_indices.append(token_idx)

        result[chain_id] = sorted(token_indices)
        token_offset += seq_len + 2  # +2 for <cls> and <eos>

    total_tokens = token_offset
    return result, total_tokens
