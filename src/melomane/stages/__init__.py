"""Importing this package registers every stage."""
import importlib

_MODULES = ["s0_doctor", "s0_ingest", "s1_curation", "s2_annotation", "s3_behaviour", "s4_compounds", "s5_training", "s6_modes", "s7_eval",
            "s8_slerp", "s9_extensions", "s10_trackb", "s11_trackc", "s12_products", "s13_paper", "s14_qa"]
for _m in _MODULES:
    try:
        importlib.import_module(f"{__name__}.{_m}")
    except ModuleNotFoundError as e:                 # stage module not present yet (development) - never hide real import errors
        if e.name != f"{__name__}.{_m}":
            raise
