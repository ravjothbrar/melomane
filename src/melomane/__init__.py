"""Melomane: emergent geometry of music embeddings."""
__version__ = "0.1.0"

# Use system certificate store on Windows / native environments (prevents SSL CERTIFICATE_VERIFY_FAILED)
try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

# Compatibility shim: umap-learn (>=0.5.8) passes `ensure_all_finite` introduced in sklearn 1.6,
# but on Windows we pin scikit-learn 1.5.2 to satisfy Windows Smart App Control code integrity.
try:
    import sklearn.utils.validation as _sk_val
    import sklearn.utils as _sk_utils

    _orig_check_array = _sk_val.check_array

    def _compat_check_array(*args, **kwargs):
        if "ensure_all_finite" in kwargs and "force_all_finite" not in kwargs:
            kwargs["force_all_finite"] = kwargs.pop("ensure_all_finite")
        return _orig_check_array(*args, **kwargs)

    _sk_val.check_array = _compat_check_array
    _sk_utils.check_array = _compat_check_array
except Exception:
    pass
