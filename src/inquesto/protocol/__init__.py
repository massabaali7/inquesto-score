"""Inquesto Protocol: a versioned definition that turns conversations into one number.

    IS = 100 x (clean successes / calls)

A call is a clean success when the caller's goal was achieved and no failure event of
severity S3 or above occurred. The population of calls (scenarios x conditions x speaker
groups x seeds), the failure taxonomy, the severity table and the judge are fixed by the
protocol version, so a score is reproducible and comparable across papers.

See docs/inquesto-score-v0.1.md for the specification this package implements.
"""

from .score import CallResult, Event, citation_line, passes, record, views, wilson
from .spec import PROTOCOL, Protocol

__all__ = ["PROTOCOL", "CallResult", "Event", "Protocol", "citation_line", "passes", "record", "views", "wilson"]
