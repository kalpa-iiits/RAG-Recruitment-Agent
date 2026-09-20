"""One shape for every paginated list endpoint.

Offset paging rather than cursors: these lists are a single user's own
resumes, they are re-sorted by the client's choice of key, and the UI needs
a page count, all of which a cursor makes awkward and none of which is worth
the extra API surface at this size.
"""

from typing import Generic, TypeVar

from pydantic import BaseModel

T = TypeVar("T")

# What a client gets when it asks for a list without saying how much of it.
DEFAULT_LIMIT = 20
MAX_LIMIT = 100


class Page(BaseModel, Generic[T]):
    """One slice of a longer list, and enough context to page through it."""

    items: list[T]
    # Rows matching the request's filters, across every page — not len(items).
    total: int
    limit: int
    offset: int
