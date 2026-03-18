"""PPI ID encoding/decoding helpers for the SAbDab pipeline."""
import re
from pipeline_config import PDB_DIR, DECOY_DIR


def ppi_encode(pdb, kind, ch1, ch2):
    """
    Build a PIsToN-compatible PPI id with exactly 3 underscore-separated parts.
      kind: 'nat' | 'd1'..'d5'
      returns '{pdb}{kind}_{ch1}_{ch2}'  e.g. '8vtdnat_BA_C'
    PIsToN splits on '_' -> pid=8vtdnat, ch1=BA, ch2=C (exactly 3 parts).
    """
    return f"{pdb.lower()}{kind}_{ch1}_{ch2}"


def ppi_decode_pid(pid):
    """
    Reverse of the encoding:
      '8vtdnat' -> ('8vtd', 'nat')
      '8vtdd1'  -> ('8vtd', 'd1')
    Raises ValueError if pattern doesn't match.
    """
    m = re.match(r"^(.+?)(nat|d[1-9]\d*)$", pid)
    if not m:
        raise ValueError(f"Cannot decode pid {pid!r}")
    return m.group(1), m.group(2)


def source_pdb_for(pdb, kind):
    """Return the filesystem Path to the source PDB file."""
    pdb = pdb.lower()
    if kind == "nat":
        return PDB_DIR / f"{pdb}.pdb"
    return DECOY_DIR / f"{pdb}_decoy{kind[1:]}.pdb"


def get_base_pdb(ppi):
    """Extract base PDB id from a PPI id string."""
    pid = ppi.split("_")[0]
    pdb, _ = ppi_decode_pid(pid)
    return pdb
