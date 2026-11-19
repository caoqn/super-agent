"""Compatibility import for the method-local adapter."""
import importlib, sys
sys.modules[__name__] = importlib.import_module("methods.Evoteam.macbench.adapter")
