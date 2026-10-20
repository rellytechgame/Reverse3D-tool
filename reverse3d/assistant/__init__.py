"""Studio's assistant: a local language model understands the request and picks
actions; Voxel Enterprise's own model (Optiverse) and the Core carry them out.

The language model never edits anything itself: it only chooses among fixed,
validated actions (see :mod:`reverse3d.assistant.actions`), and every reply is
built from what those actions really did.
"""
