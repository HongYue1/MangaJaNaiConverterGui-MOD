"""The out-of-process upscaling pipeline.

Run as a script (``python -m janai.worker.worker --job job.json``) rather than
imported by the interface: torch, numpy and pyvips must stay out of the GUI
process.
"""
