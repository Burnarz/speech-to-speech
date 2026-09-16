"""Server event models that extend the stock OpenAI Realtime protocol.

These events are emitted in addition to the standard GA events so clients can
observe work that is not part of the public protocol (for example a tool call
whose arguments are still streaming and cannot be parsed yet).  They are
unknown to stock clients, which simply ignore the extra event type.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class ResponseFunctionCallArgumentsProgressEvent(BaseModel):
    """Incremental raw content of a function call while its block is open.

    Mirrors ``response.function_call_arguments.done`` but streams.  ``delta``
    carries the raw in-block text since the previous progress event (the
    delimiter markers are not included); it is not a JSON fragment of the
    final arguments.  ``item_id`` and ``call_id`` match the function-call
    item that is emitted when the call completes, so clients can correlate.
    """

    type: str = "response.function_call_arguments.progress"
    event_id: str
    item_id: str
    call_id: str
    name: Optional[str] = None
    output_index: int = 0
    response_id: str
    delta: str = ""
