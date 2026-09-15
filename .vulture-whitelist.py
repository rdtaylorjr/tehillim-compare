"""Symbols vulture cannot see used: entry points the Snakefile calls."""

from tehillim_compare import driver

#: Called from the Snakefile, which vulture does not scan.
driver.cell_provenance
driver.roots_from_config
