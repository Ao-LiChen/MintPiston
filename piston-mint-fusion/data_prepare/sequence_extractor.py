"""
Extract amino acid sequences and residue-number-to-sequence-position mappings
from PDB files.  This bridges the PDB residue numbering used in PIsToN's
resnames.npy with the sequential token positions used by MINT.

Supports multi-chain sides: PIsToN uses notation like 'HL' to mean chains
H and L together form one side of the interaction.  This module extracts
each individual PDB chain separately (e.g. 'H' and 'L') so that MINT can
assign them distinct chain_ids for proper inter-chain attention.
"""

import warnings
from collections import OrderedDict

from Bio.PDB import PDBParser
from Bio import BiopythonWarning

warnings.simplefilter("ignore", BiopythonWarning)

# Standard 3-letter to 1-letter amino acid code mapping
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    # Non-standard but commonly encountered
    "MSE": "M",  # selenomethionine
    "SEC": "U",  # selenocysteine
    "PYL": "O",  # pyrrolysine
}


def parse_ppi_identifier(ppi):
    """
    Parse a PPI identifier of the form PID_side1_side2.

    PIsToN encodes multi-chain sides as concatenated single-letter chain IDs.
    For example, '1AHW_HL_C' means:
        PDB ID = 1AHW
        side 1 = chains H and L (antibody)
        side 2 = chain C (antigen)

    Returns:
        pid: str, PDB identifier
        side1: str, chain letters for side 1 (e.g. 'HL')
        side2: str, chain letters for side 2 (e.g. 'C')
        all_chains: list of individual chain letters, e.g. ['H', 'L', 'C']
    """
    parts = ppi.split("_")
    pid = parts[0]
    side1 = parts[1]
    side2 = parts[2]
    all_chains = list(side1) + list(side2)
    return pid, side1, side2, all_chains


def extract_sequences_from_pdb(pdb_path, chain_ids):
    """
    Parse a PDB file and extract the amino acid sequence for each requested chain.

    Args:
        pdb_path: path to the PDB file
        chain_ids: list of individual chain identifiers, e.g. ['H', 'L', 'C']
                   (NOT multi-letter side strings -- use parse_ppi_identifier first)

    Returns:
        dict mapping chain_id -> {
            'sequence': str  (one-letter amino acid codes),
            'residue_map': OrderedDict mapping (resid_int, insertion_code) -> seq_position (0-indexed),
            'residue_map_simple': dict mapping resid_int -> seq_position (0-indexed)
                (uses the first occurrence if multiple insertion codes exist for the same resid)
        }
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", pdb_path)
    model = structure[0]

    results = {}
    for chain_id in chain_ids:
        if chain_id not in model:
            raise ValueError(
                f"Chain '{chain_id}' not found in PDB file {pdb_path}. "
                f"Available chains: {[c.id for c in model.get_chains()]}"
            )
        chain = model[chain_id]

        sequence = []
        residue_map = OrderedDict()       # (resid, icode) -> seq_pos
        residue_map_simple = {}           # resid -> seq_pos

        seq_pos = 0
        for residue in chain.get_residues():
            het_flag = residue.id[0]
            # Skip water and heteroatoms (keep standard residues only)
            if het_flag != " ":
                continue

            resname = residue.resname.strip()
            one_letter = THREE_TO_ONE.get(resname, "X")

            resid = residue.id[1]        # residue number (int)
            icode = residue.id[2].strip()  # insertion code (e.g. 'A', '')

            sequence.append(one_letter)
            residue_map[(resid, icode)] = seq_pos
            # Only store first occurrence for the simple map
            if resid not in residue_map_simple:
                residue_map_simple[resid] = seq_pos

            seq_pos += 1

        results[chain_id] = {
            "sequence": "".join(sequence),
            "residue_map": residue_map,
            "residue_map_simple": residue_map_simple,
        }

    return results
