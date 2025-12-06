from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class Action:
    """
    A single user action in a search session.

    Represents either a query issuance or a click on a result item.
    Only `session_id`, `query`, and `is_click` are required; all other
    fields are optional to accommodate different dataset formats.

    Attributes
    ----------
    session_id : str
        Unique identifier for the session this action belongs to.
    query : str
        The query text issued by the user.
    is_click : bool
        True if this action represents a click, False for a query submission.
    topic_id : str, optional
        Topic/task identifier (e.g., TREC topic number).
    doc_id : str, optional
        Document identifier that was clicked.
    rank : int, optional
        Rank position of the clicked document in the SERP.
    timestamp : float, optional
        Unix timestamp or relative time of the action.
    title : str, optional
        Title of the clicked document.
    snippet : str, optional
        Snippet/summary of the clicked document shown in SERP.
    url : str, optional
        URL of the clicked document.
    dwell_time : float, optional
        Time spent on the clicked document (in seconds).
    relevance_label : int, optional
        Ground-truth relevance label for the document.
    """

    session_id: str
    query: str
    is_click: bool
    # Optional fields for different dataset types
    topic_id: Optional[str] = None
    doc_id: Optional[str] = None
    rank: Optional[int] = None
    timestamp: Optional[float] = None
    title: Optional[str] = None
    snippet: Optional[str] = None
    url: Optional[str] = None
    dwell_time: Optional[float] = None
    relevance_label: Optional[int] = None


@dataclass(frozen=True)
class Session:
    """
    A full search session containing a sequence of user actions.

    Attributes
    ----------
    session_id : str
        Unique identifier for this session.
    actions : List[Action]
        Ordered list of actions (queries and clicks) in this session.
    topic_id : str, optional
        Topic/task identifier for the session.
    """

    session_id: str
    actions: List[Action]
    topic_id: Optional[str] = None
