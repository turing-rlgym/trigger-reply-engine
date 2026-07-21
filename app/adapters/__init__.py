"""Gym adapters: turn a validated action + params into a concrete HTTP request."""

from app.adapters.base import BuiltRequest, GymAdapter, ManifestAdapter

__all__ = ["BuiltRequest", "GymAdapter", "ManifestAdapter"]
