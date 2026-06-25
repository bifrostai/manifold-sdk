"""Pure helper math that adapters compose from.

The functions here take and return plain numbers and arrays. They do not reference
domain model types (embodiment, benchmark, or spec) and may use only the convention
enums, so an adapter can build a conversion out of them without pulling in the
rest of the package.
"""
