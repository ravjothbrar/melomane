"""Release products: model bundles (`export`), the static explorer payload (`explorer`) and the human-QC logic (`review`).

Submodules are imported lazily by callers (``from melomane.release.export import export_bundle``) so that importing the
package never pulls in UMAP or pandas-heavy code paths.
"""
