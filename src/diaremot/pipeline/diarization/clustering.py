from __future__ import annotations

from typing import Any

try:
    import inspect

    from sklearn.cluster import AgglomerativeClustering

    def build_agglo(distance_threshold: float | None, **kwargs: Any) -> AgglomerativeClustering:
        init_sig = inspect.signature(AgglomerativeClustering.__init__)
        params = set(init_sig.parameters)
        wanted = {
            "n_clusters": None,
            "distance_threshold": distance_threshold,
            "linkage": kwargs.pop("linkage", "average"),
        }
        if "metric" in params:
            wanted["metric"] = kwargs.pop("metric", kwargs.pop("affinity", "cosine"))
        elif "affinity" in params:
            wanted["affinity"] = kwargs.pop("metric", kwargs.pop("affinity", "cosine"))
        for key, value in kwargs.items():
            if key in params:
                wanted[key] = value
        return AgglomerativeClustering(**wanted)

except Exception:  # pragma: no cover
    AgglomerativeClustering = None

    def build_agglo(distance_threshold: float | None, **kwargs: Any):  # type: ignore[override]
        raise RuntimeError("sklearn AgglomerativeClustering not available")


try:  # pragma: no cover
    from spectralcluster import SpectralClusterer  # type: ignore
except Exception:  # pragma: no cover
    SpectralClusterer = None  # type: ignore


__all__ = ["build_agglo", "SpectralClusterer"]
