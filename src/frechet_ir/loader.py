"""
Generic session loaders for various dataset formats.

This module provides utilities to load search session data from common formats
including JSON, CSV, and Python dictionaries.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from .model import Action, Session


def _parse_action(data: Dict[str, Any], session_id: str) -> Action:
    """Parse a single action from a dictionary."""
    return Action(
        session_id=session_id,
        query=str(data.get("query", "")),
        is_click=bool(data.get("is_click", False)),
        topic_id=data.get("topic_id"),
        doc_id=data.get("doc_id"),
        rank=int(data["rank"]) if data.get("rank") is not None else None,
        timestamp=float(data["timestamp"]) if data.get("timestamp") is not None else None,
        title=data.get("title"),
        snippet=data.get("snippet"),
        url=data.get("url"),
        dwell_time=float(data["dwell_time"]) if data.get("dwell_time") is not None else None,
        relevance_label=int(data["relevance_label"]) if data.get("relevance_label") is not None else None,
    )


def load_sessions_from_dict(
    data: Sequence[Dict[str, Any]],
) -> List[Session]:
    """
    Load sessions from a list of dictionaries.

    Each dictionary should represent a session with the following structure:
    
    ```python
    {
        "session_id": "123",
        "topic_id": "topic_1",  # optional
        "actions": [
            {
                "query": "python tutorial",
                "is_click": False,
            },
            {
                "query": "python tutorial",
                "is_click": True,
                "doc_id": "doc_001",
                "rank": 1,
                "title": "Python Tutorial",  # optional
                "snippet": "Learn Python...",  # optional
                "url": "https://...",  # optional
                "dwell_time": 30.5,  # optional
                "relevance_label": 2,  # optional
            },
        ]
    }
    ```

    Parameters
    ----------
    data : Sequence[Dict[str, Any]]
        List of session dictionaries.

    Returns
    -------
    List[Session]
        Parsed session objects.
    """
    sessions: List[Session] = []

    for session_data in data:
        session_id = str(session_data.get("session_id", ""))
        topic_id = session_data.get("topic_id")
        
        actions_data = session_data.get("actions", [])
        actions = [_parse_action(a, session_id) for a in actions_data]
        
        sessions.append(Session(
            session_id=session_id,
            actions=actions,
            topic_id=topic_id,
        ))

    return sessions


def load_sessions_from_json(
    path: Union[str, Path],
    encoding: str = "utf-8",
) -> List[Session]:
    """
    Load sessions from a JSON file.

    The JSON file should contain a list of session objects. See
    `load_sessions_from_dict` for the expected structure.

    Parameters
    ----------
    path : str or Path
        Path to the JSON file.
    encoding : str
        File encoding, default "utf-8".

    Returns
    -------
    List[Session]
        Parsed session objects.

    Examples
    --------
    >>> sessions = load_sessions_from_json("sessions.json")
    >>> print(len(sessions))
    100
    """
    path = Path(path)
    with open(path, "r", encoding=encoding) as f:
        data = json.load(f)

    if isinstance(data, dict) and "sessions" in data:
        # Support {"sessions": [...]} wrapper format
        data = data["sessions"]

    return load_sessions_from_dict(data)


def load_sessions_from_csv(
    path: Union[str, Path],
    encoding: str = "utf-8",
    delimiter: str = ",",
) -> List[Session]:
    """
    Load sessions from a CSV file.

    The CSV file should have one row per action with at minimum the columns:
    `session_id`, `query`, `is_click`. Additional optional columns include:
    `topic_id`, `doc_id`, `rank`, `timestamp`, `title`, `snippet`, `url`,
    `dwell_time`, `relevance_label`.

    Actions are grouped by `session_id` and ordered by their appearance
    in the file (or by `timestamp` if available).

    Parameters
    ----------
    path : str or Path
        Path to the CSV file.
    encoding : str
        File encoding, default "utf-8".
    delimiter : str
        CSV delimiter, default ",".

    Returns
    -------
    List[Session]
        Parsed session objects.

    Examples
    --------
    >>> sessions = load_sessions_from_csv("query_logs.csv")
    >>> for s in sessions[:3]:
    ...     print(s.session_id, len(s.actions))
    """
    path = Path(path)
    
    sessions_dict: Dict[str, Dict[str, Any]] = {}

    with open(path, "r", encoding=encoding, newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        
        for row in reader:
            session_id = row.get("session_id", "")
            
            if session_id not in sessions_dict:
                sessions_dict[session_id] = {
                    "session_id": session_id,
                    "topic_id": row.get("topic_id"),
                    "actions": [],
                }
            
            # Parse is_click - handle various boolean representations
            is_click_raw = row.get("is_click", "false").lower()
            is_click = is_click_raw in ("true", "1", "yes", "click")
            
            action_data = {
                "query": row.get("query", ""),
                "is_click": is_click,
                "topic_id": row.get("topic_id"),
                "doc_id": row.get("doc_id"),
                "rank": row.get("rank") if row.get("rank") else None,
                "timestamp": row.get("timestamp") if row.get("timestamp") else None,
                "title": row.get("title"),
                "snippet": row.get("snippet"),
                "url": row.get("url"),
                "dwell_time": row.get("dwell_time") if row.get("dwell_time") else None,
                "relevance_label": row.get("relevance_label") if row.get("relevance_label") else None,
            }
            
            sessions_dict[session_id]["actions"].append(action_data)

    # Convert to session objects
    sessions_list = list(sessions_dict.values())
    
    # Sort actions by timestamp if available
    for session_data in sessions_list:
        actions = session_data["actions"]
        if actions and any(a.get("timestamp") for a in actions):
            actions.sort(key=lambda a: float(a["timestamp"]) if a.get("timestamp") else 0.0)

    return load_sessions_from_dict(sessions_list)
