"""Open Targets Platform GraphQL API loader."""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import requests

from omicsync.core.dataset import OmicsDataset
from omicsync.utils.logging import get_logger

logger = get_logger("loaders.open_targets")

_OT_GRAPHQL_URL = "https://api.platform.opentargets.org/api/v4/graphql"

_ASSOCIATION_QUERY = """
query targetDiseaseAssociations(
    $diseaseIds: [String!],
    $targetIds: [String!],
    $size: Int!,
    $cursor: String
) {
    associations: associatedTargets(
        diseaseIds: $diseaseIds
        size: $size
        cursor: $cursor
    ) {
        count
        cursor
        rows {
            target {
                id
                approvedSymbol
            }
            disease {
                id
                name
            }
            score
            datatypeScores {
                id
                score
            }
        }
    }
}
"""

_DATATYPE_COLUMNS = {
    "genetic_association": "genetic_association",
    "somatic_mutation": "somatic_mutation",
    "literature": "literature",
    "rna_expression": "rna_expression",
    "animal_model": "animal_model",
    "affected_pathway": "affected_pathway",
}


def _graphql_request(
    payload: Dict,
    url: str = _OT_GRAPHQL_URL,
    max_retries: int = 5,
    backoff_factor: float = 1.0,
) -> Dict:
    """Execute a GraphQL query with exponential backoff."""
    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as exc:
            if attempt == max_retries - 1:
                raise RuntimeError(
                    f"Open Targets API request failed after {max_retries} attempts: {exc}"
                ) from exc
            wait = backoff_factor * (2 ** attempt)
            logger.warning(
                "Open Targets request failed (attempt %d/%d); retrying in %.1fs.",
                attempt + 1, max_retries, wait,
            )
            time.sleep(wait)
    raise RuntimeError("Unreachable")  # pragma: no cover


def load_open_targets_targets(
    disease_ids: Optional[Sequence[str]] = None,
    target_ids: Optional[Sequence[str]] = None,
    evidence_types: Optional[Sequence[str]] = None,
    score_threshold: float = 0.0,
    page_size: int = 200,
) -> pd.DataFrame:
    """Query Open Targets Platform for target-disease associations.

    Parameters
    ----------
    disease_ids:
        EFO disease IDs to filter on, e.g. ``["EFO_0000305"]``.
        At least one of *disease_ids* or *target_ids* must be provided.
    target_ids:
        Ensembl gene IDs to filter on, e.g. ``["ENSG00000141736"]``.
    evidence_types:
        Evidence types to include in results.  ``None`` returns all.
        Valid keys: ``"genetic_association"``, ``"somatic_mutation"``,
        ``"literature"``, ``"rna_expression"``, ``"animal_model"``,
        ``"affected_pathway"``.
    score_threshold:
        Minimum overall association score (0–1).
    page_size:
        Results per API page.

    Returns
    -------
    pandas.DataFrame
        Columns: ``target_id``, ``target_name``, ``disease_id``,
        ``disease_name``, ``overall_score``, plus one column per evidence
        datatype.

    Raises
    ------
    ValueError
        If neither *disease_ids* nor *target_ids* is provided.
    """
    if disease_ids is None and target_ids is None:
        raise ValueError("Provide at least one of disease_ids or target_ids.")

    rows: List[Dict] = []
    cursor: Optional[str] = None
    total_fetched = 0

    while True:
        variables: Dict = {"size": page_size}
        if disease_ids:
            variables["diseaseIds"] = list(disease_ids)
        if cursor:
            variables["cursor"] = cursor

        result = _graphql_request({"query": _ASSOCIATION_QUERY, "variables": variables})

        data = result.get("data", {}).get("associations", {})
        page_rows = data.get("rows", [])
        cursor = data.get("cursor")

        for row in page_rows:
            target = row.get("target", {})
            disease = row.get("disease", {})
            overall_score = row.get("score", 0.0) or 0.0

            if overall_score < score_threshold:
                continue

            record: Dict = {
                "target_id": target.get("id"),
                "target_name": target.get("approvedSymbol"),
                "disease_id": disease.get("id"),
                "disease_name": disease.get("name"),
                "overall_score": overall_score,
            }

            dt_scores = {s["id"]: s["score"] for s in row.get("datatypeScores", [])}
            for col, key in _DATATYPE_COLUMNS.items():
                record[col] = dt_scores.get(key, np.nan)

            rows.append(record)

        total_fetched += len(page_rows)
        logger.info("load_open_targets_targets: fetched %d associations so far.", total_fetched)

        if not cursor or len(page_rows) < page_size:
            break

    if not rows:
        logger.warning("load_open_targets_targets: no associations returned.")
        return pd.DataFrame(columns=[
            "target_id", "target_name", "disease_id", "disease_name",
            "overall_score", *list(_DATATYPE_COLUMNS.keys()),
        ])

    df = pd.DataFrame(rows)

    if evidence_types is not None:
        keep = set(evidence_types) & set(_DATATYPE_COLUMNS.keys())
        if not keep:
            logger.warning(
                "load_open_targets_targets: none of %s are valid evidence types.", evidence_types
            )
        else:
            df = df[df[list(keep)].notna().any(axis=1)]

    logger.info(
        "load_open_targets_targets: returned %d associations.", len(df)
    )
    return df.reset_index(drop=True)


def add_open_targets_annotations(
    dataset: OmicsDataset,
    target_column: str = "gene_id",
    disease_ids: Optional[Sequence[str]] = None,
    **kwargs,
) -> OmicsDataset:
    """Annotate feature metadata in an OmicsDataset with Open Targets scores.

    Queries Open Targets for each feature in the RNA modality (or any modality
    whose feature IDs look like gene symbols or Ensembl IDs) and attaches the
    association scores as feature-level metadata.

    Parameters
    ----------
    dataset:
        An :class:`~omicsync.core.dataset.OmicsDataset`.
    target_column:
        Column in the annotation DataFrame corresponding to gene identifiers.
    disease_ids:
        Disease IDs to query.  Forwarded to :func:`load_open_targets_targets`.
    **kwargs:
        Forwarded to :func:`load_open_targets_targets`.

    Returns
    -------
    OmicsDataset
        *dataset* with ``open_targets`` key added to each modality's metadata.
    """
    ot_df = load_open_targets_targets(disease_ids=disease_ids, **kwargs)

    for name, mod in dataset._modalities.items():
        feature_ids = mod.feature_ids.tolist()
        ann = ot_df[ot_df["target_name"].isin(feature_ids)].copy()
        mod.metadata["open_targets"] = ann
        logger.info(
            "add_open_targets_annotations: %d/%d features annotated for modality %r.",
            len(ann["target_name"].unique()),
            len(feature_ids),
            name,
        )

    return dataset
