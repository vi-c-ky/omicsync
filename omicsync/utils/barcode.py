"""TCGA barcode parsing utilities."""

from __future__ import annotations

from typing import List, Sequence

import pandas as pd

# TCGA barcode structure:
# TCGA-{TSS}-{Participant}-{Sample}{Vial}-{Portion}{Analyte}-{Plate}-{Centre}
# e.g. TCGA-02-0001-01A-01R-0177-13
#        0    1   2    3     4      5    6   (dash-split index)
# Sample type codes: 01-09 = tumour; 10-19 = normal; 20-29 = control


def parse_barcode(barcode: str) -> dict:
    """Parse a TCGA barcode into its component fields.

    Parameters
    ----------
    barcode:
        A full TCGA aliquot barcode, e.g. ``"TCGA-02-0001-01A-01R-0177-13"``.

    Returns
    -------
    dict
        Keys: ``project``, ``tss``, ``participant``, ``sample``, ``vial``,
        ``portion``, ``analyte``, ``plate``, ``centre``.  Missing trailing
        fields are ``None``.

    Raises
    ------
    ValueError
        If the barcode does not start with ``"TCGA-"``.
    """
    barcode = barcode.strip()
    if not barcode.upper().startswith("TCGA-"):
        raise ValueError(f"Not a valid TCGA barcode: {barcode!r}")

    parts = barcode.split("-")
    result: dict = {
        "project": parts[0] if len(parts) > 0 else None,
        "tss": parts[1] if len(parts) > 1 else None,
        "participant": parts[2] if len(parts) > 2 else None,
        "sample": parts[3][:2] if len(parts) > 3 else None,
        "vial": parts[3][2:] if len(parts) > 3 and len(parts[3]) > 2 else None,
        "portion": parts[4][:2] if len(parts) > 4 else None,
        "analyte": parts[4][2:] if len(parts) > 4 and len(parts[4]) > 2 else None,
        "plate": parts[5] if len(parts) > 5 else None,
        "centre": parts[6] if len(parts) > 6 else None,
    }
    return result


def truncate_to_participant(barcode: str) -> str:
    """Return the participant-level ID (first 12 characters).

    Parameters
    ----------
    barcode:
        Full or partial TCGA barcode.

    Returns
    -------
    str
        E.g. ``"TCGA-02-0001"``.
    """
    parts = barcode.strip().split("-")
    if len(parts) < 3:
        raise ValueError(
            f"Barcode {barcode!r} does not contain enough fields to extract "
            "a participant ID."
        )
    return "-".join(parts[:3])


def truncate_to_sample(barcode: str) -> str:
    """Return the sample-level ID (first 15–16 characters, through sample+vial).

    Parameters
    ----------
    barcode:
        Full or partial TCGA barcode.

    Returns
    -------
    str
        E.g. ``"TCGA-02-0001-01A"``.
    """
    parts = barcode.strip().split("-")
    if len(parts) < 4:
        raise ValueError(
            f"Barcode {barcode!r} does not contain enough fields to extract "
            "a sample ID."
        )
    return "-".join(parts[:4])


def is_tumour(barcode: str) -> bool:
    """Return ``True`` if the barcode represents a tumour sample (type 01-09).

    Parameters
    ----------
    barcode:
        Full or partial TCGA barcode.
    """
    parts = barcode.strip().split("-")
    if len(parts) < 4:
        return False
    sample_code = parts[3][:2]
    try:
        return 1 <= int(sample_code) <= 9
    except ValueError:
        return False


def is_normal(barcode: str) -> bool:
    """Return ``True`` if the barcode represents a normal sample (type 10-19).

    Parameters
    ----------
    barcode:
        Full or partial TCGA barcode.
    """
    parts = barcode.strip().split("-")
    if len(parts) < 4:
        return False
    sample_code = parts[3][:2]
    try:
        return 10 <= int(sample_code) <= 19
    except ValueError:
        return False


def batch_parse(barcodes: Sequence[str]) -> pd.DataFrame:
    """Parse a sequence of TCGA barcodes into a DataFrame.

    Parameters
    ----------
    barcodes:
        Iterable of TCGA barcode strings.

    Returns
    -------
    pandas.DataFrame
        One row per barcode; columns match the keys of :func:`parse_barcode`,
        plus ``is_tumour`` and ``is_normal`` boolean columns.
    """
    rows = []
    for bc in barcodes:
        try:
            row = parse_barcode(bc)
        except ValueError:
            row = {k: None for k in [
                "project", "tss", "participant", "sample",
                "vial", "portion", "analyte", "plate", "centre"
            ]}
        row["barcode"] = bc
        row["is_tumour"] = is_tumour(bc)
        row["is_normal"] = is_normal(bc)
        rows.append(row)
    df = pd.DataFrame(rows)
    cols = ["barcode", "project", "tss", "participant", "sample", "vial",
            "portion", "analyte", "plate", "centre", "is_tumour", "is_normal"]
    return df[[c for c in cols if c in df.columns]]
