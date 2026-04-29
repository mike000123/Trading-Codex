"""
Data package.

Intentionally side-effect free so importing a submodule such as
`data.fair_value` does not also force heavy sibling imports during app boot.
"""

__all__: list[str] = []
