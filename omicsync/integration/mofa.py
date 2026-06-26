"""MOFA2 wrapper for multi-omics factor analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from omicsync.core.dataset import OmicsDataset
from omicsync.utils.logging import get_logger

logger = get_logger("integration.mofa")


class MOFA2Wrapper:
    """Wrapper around ``mofapy2`` for factor analysis of multi-omics data.

    Parameters
    ----------
    dataset:
        An :class:`~omicsync.core.dataset.OmicsDataset`.
    n_factors:
        Number of latent factors to learn (default 10).
    convergence_mode:
        ``"fast"``, ``"medium"``, or ``"slow"`` (default ``"fast"``).
    use_gpu:
        Whether to use GPU acceleration (requires CUDA; default ``False``).
    seed:
        Random seed for reproducibility (default 42).

    Raises
    ------
    ImportError
        If ``mofapy2`` is not installed.
    """

    def __init__(
        self,
        dataset: OmicsDataset,
        n_factors: int = 10,
        convergence_mode: str = "fast",
        use_gpu: bool = False,
        seed: int = 42,
    ) -> None:
        try:
            import mofapy2  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "mofapy2 is required for MOFA2Wrapper. "
                "Install it with: pip install mofapy2"
            ) from exc

        self.dataset = dataset
        self.n_factors = n_factors
        self.convergence_mode = convergence_mode
        self.use_gpu = use_gpu
        self.seed = seed

        self._model = None
        self._prepared_data: Optional[Dict] = None
        self._samples: Optional[List[str]] = None
        self._views: Optional[List[str]] = None

    def prepare(self) -> "MOFA2Wrapper":
        """Format the dataset for mofapy2.

        Returns
        -------
        MOFA2Wrapper
            *self*, for method chaining.
        """
        mofa_input = self.dataset.to_mofa2()
        self._prepared_data = mofa_input
        self._views = mofa_input["views"]
        self._samples = mofa_input["samples"][0]  # single group
        logger.info(
            "MOFA2Wrapper.prepare: %d views, %d samples.",
            len(self._views),
            len(self._samples),
        )
        return self

    def train(self, output_path: Optional[Union[str, Path]] = None) -> "MOFA2Wrapper":
        """Train the MOFA2 model.

        Parameters
        ----------
        output_path:
            If provided, save the trained model to this HDF5 file path.

        Returns
        -------
        MOFA2Wrapper
            *self*, for method chaining.

        Raises
        ------
        RuntimeError
            If :meth:`prepare` has not been called first.
        """
        if self._prepared_data is None:
            raise RuntimeError("Call prepare() before train().")

        from mofapy2.run.entry_point import entry_point

        ent = entry_point()

        data = self._prepared_data["data"]
        groups = self._prepared_data["groups"]
        views = self._prepared_data["views"]
        samples = self._prepared_data["samples"]

        ent.set_data_options(scale_groups=False, scale_views=False)
        ent.set_data_matrix(
            data,
            likelihoods=["gaussian"] * len(views),
            views_names=views,
            groups_names=groups,
            samples_names=samples,
            features_names=[
                self.dataset._modalities[v].feature_ids.tolist() for v in views
            ],
        )
        ent.set_model_options(
            factors=self.n_factors,
            spikeslab_weights=True,
            ard_factors=True,
            ard_weights=True,
        )
        ent.set_train_options(
            convergence_mode=self.convergence_mode,
            gpu_mode=self.use_gpu,
            seed=self.seed,
            verbose=False,
        )
        ent.build()
        ent.run()

        self._model = ent.model

        if output_path is not None:
            output_path = Path(output_path)
            ent.save(str(output_path))
            logger.info("MOFA2Wrapper: model saved to %s.", output_path)

        logger.info("MOFA2Wrapper: training complete.")
        return self

    def get_factors(self) -> pd.DataFrame:
        """Return factor scores for all samples.

        Returns
        -------
        pandas.DataFrame
            Shape (n_samples, n_factors); index = sample IDs,
            columns = ``Factor1``, ``Factor2``, ...

        Raises
        ------
        RuntimeError
            If the model has not been trained yet.
        """
        self._check_trained()
        Z = self._model.nodes["Z"].getExpectation()
        # Z has shape (n_groups, n_samples, n_factors)
        scores = Z[0]
        cols = [f"Factor{i+1}" for i in range(scores.shape[1])]
        return pd.DataFrame(scores, index=self._samples, columns=cols)

    def get_weights(
        self, modality: Optional[str] = None
    ) -> Union[Dict[str, pd.DataFrame], pd.DataFrame]:
        """Return feature weights.

        Parameters
        ----------
        modality:
            If specified, return weights for that modality only.
            Otherwise return a dict of DataFrames keyed by modality name.

        Returns
        -------
        dict[str, pandas.DataFrame] or pandas.DataFrame
        """
        self._check_trained()
        W = self._model.nodes["W"].getExpectation()
        # W: list of arrays (n_features, n_factors) per view
        result: Dict[str, pd.DataFrame] = {}
        for i, view in enumerate(self._views):
            features = self.dataset._modalities[view].feature_ids.tolist()
            cols = [f"Factor{j+1}" for j in range(W[i].shape[1])]
            result[view] = pd.DataFrame(W[i], index=features, columns=cols)

        if modality is not None:
            if modality not in result:
                raise KeyError(f"Modality {modality!r} not found; available: {list(result)}.")
            return result[modality]
        return result

    def get_variance_explained(self) -> pd.DataFrame:
        """Return R² per factor per modality.

        Returns
        -------
        pandas.DataFrame
            Shape (n_factors, n_modalities).
        """
        self._check_trained()
        r2 = self._model.calculate_variance_explained()
        # r2 is a list of arrays (n_factors, 1) per view
        data = np.concatenate([v[0] for v in r2], axis=1) if isinstance(r2, list) else r2
        cols = self._views
        idx = [f"Factor{i+1}" for i in range(data.shape[0])]
        return pd.DataFrame(data, index=idx, columns=cols)

    def plot_variance_explained(self) -> None:
        """Plot a bar chart of variance explained per factor per modality.

        Requires ``matplotlib``.

        Raises
        ------
        ImportError
            If ``matplotlib`` is not installed.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError(
                "matplotlib is required for plot_variance_explained(). "
                "Install it with: pip install matplotlib"
            ) from exc

        r2 = self.get_variance_explained()
        r2.T.plot(kind="bar", figsize=(10, 4))
        plt.ylabel("Variance Explained (R²)")
        plt.title("MOFA2 Variance Explained per View")
        plt.tight_layout()
        plt.show()

    def top_features(
        self, factor: int, modality: str, n: int = 20
    ) -> pd.DataFrame:
        """Return the top *n* features by absolute weight for a factor.

        Parameters
        ----------
        factor:
            1-based factor index.
        modality:
            Modality name.
        n:
            Number of top features to return.

        Returns
        -------
        pandas.DataFrame
            Columns: ``feature``, ``weight``, ``abs_weight``.
        """
        self._check_trained()
        weights = self.get_weights(modality)
        col = f"Factor{factor}"
        if col not in weights.columns:
            raise ValueError(
                f"Factor {factor} not found; available 1–{len(weights.columns)}."
            )
        s = weights[col].abs().sort_values(ascending=False).head(n)
        df = pd.DataFrame({
            "feature": s.index,
            "abs_weight": s.values,
            "weight": weights.loc[s.index, col].values,
        })
        return df.reset_index(drop=True)

    def _check_trained(self) -> None:
        if self._model is None:
            raise RuntimeError("Model not trained yet. Call prepare() then train().")
