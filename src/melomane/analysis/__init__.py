"""Analysis library (PLAN §6 S7-S9, §7): pure, seeded numpy-in / dict-or-DataFrame-out functions.

Modules: stats, weat, procrustes, snr, compression, attribution, agreement, temporal, direction_test, multiverse, battery."""
from .stats import (paired_cluster_bootstrap, holm, permutation_test, bootstrap_ci, spearman_ci, cohens_kappa,
                    weighted_kappa)
from .weat import weat, weat_region_energy
from .procrustes import procrustes_align, procrustes_matrix, shared_rows
from .snr import orthogonal_residual_snr, snr_ranking
from .compression import (CompressedEmbedding, compress, retention_curve, smallest_config, quantise_int8,
                          dequantise_int8, storage_size_bytes)
from .attribution import shapley_2player, shapley_attribution
from .agreement import (knn_indices, knn_jaccard, neighbour_recall, linear_cka, rbf_cka, agreement_report)
from .temporal import procrustes_disparity, mode_persistence
from .direction_test import (supervised_pole, sample_seed_pole_pairs, theta_sweep_metrics, monotonicity,
                             increasing_share, multi_pole_composition)
from .multiverse import run_grid, summarise, cell_seed
from .battery import probe_battery, stratum_means, geometry_summary, knn_battery

__all__ = [
    "paired_cluster_bootstrap", "holm", "permutation_test", "bootstrap_ci", "spearman_ci", "cohens_kappa", "weighted_kappa",
    "weat", "weat_region_energy", "procrustes_align", "procrustes_matrix", "shared_rows",
    "orthogonal_residual_snr", "snr_ranking",
    "CompressedEmbedding", "compress", "retention_curve", "smallest_config", "quantise_int8", "dequantise_int8",
    "storage_size_bytes", "shapley_2player", "shapley_attribution",
    "knn_indices", "knn_jaccard", "neighbour_recall", "linear_cka", "rbf_cka", "agreement_report",
    "procrustes_disparity", "mode_persistence",
    "supervised_pole", "sample_seed_pole_pairs", "theta_sweep_metrics", "monotonicity", "increasing_share",
    "multi_pole_composition", "run_grid", "summarise", "cell_seed",
    "probe_battery", "stratum_means", "geometry_summary", "knn_battery",
]
