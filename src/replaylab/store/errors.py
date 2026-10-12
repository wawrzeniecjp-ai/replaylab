"""Store exceptions. All inherit from :class:`StoreError`."""

from __future__ import annotations


class StoreError(Exception):
    """Base class for every error raised by the campaign store."""


class SchemaError(StoreError):
    """The database cannot be used with this version of the code."""


class UnknownCampaign(StoreError):
    """The campaign id does not exist."""


class UnknownDesign(StoreError):
    """A design id is not part of the campaign."""


class InvalidTransition(StoreError):
    """A batch status change that is not allowed."""


class OutcomeConflict(StoreError):
    """A delivered outcome differs from the one already stored for that design."""
