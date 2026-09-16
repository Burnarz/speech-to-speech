"""Contract tests for the streaming tool-call progress side channel.

The server extends the OpenAI Realtime protocol with
``response.function_call_arguments.progress`` so document-style tools can
render before a tool call is complete and parseable.  These tests drive that
side channel through the service and check:

* the progress events carry the same ``item_id`` / ``call_id`` /
  ``output_index`` as the final ``output_item.added`` +
  ``function_call_arguments.done`` pair, so clients can correlate them;
* standard events still satisfy the OpenAI schema and the response
  lifecycle contract;
* the early side channel (tool call ready) and the ordered copy still dedupe
  when a progress reservation is in place.
"""

from typing import Any

from openai.types.realtime.realtime_response_create_params import RealtimeResponseCreateParams

from speech_to_speech.pipeline.events import (
    AssistantOutputEvent,
    AssistantToolCallProgressEvent,
    AssistantToolCallReadyEvent,
)
from speech_to_speech.pipeline.messages import (
    AssistantTextPart,
    AssistantToolCallPart,
)

from .realtime_contract import (
    assert_openai_schema,
    assert_response_lifecycle_contract,
)

_PROGRESS_TYPE = "response.function_call_arguments.progress"

_RESPONSE_KEY = "response_1"


def _progress(delta: str, *, name: str | None = None) -> AssistantToolCallProgressEvent:
    return AssistantToolCallProgressEvent(
        response_key=_RESPONSE_KEY,
        name=name,
        delta=delta,
    )


def _tool(call_id: str) -> AssistantToolCallPart:
    return AssistantToolCallPart(
        tool={
            "type": "function_call",
            "call_id": call_id,
            "name": "show_document",
            "arguments": '{"title": "T", "content": "C"}',
        }
    )


def _open_response(service: Any, conn_id: str) -> None:
    service._state(conn_id).current_response_params = RealtimeResponseCreateParams(
        output_modalities=["text"],
    )
    service.response._ensure_response(conn_id)


def _split(events: list[Any]) -> tuple[list[Any], list[Any]]:
    progress = [event for event in events if event.type == _PROGRESS_TYPE]
    standard = [event for event in events if event.type != _PROGRESS_TYPE]
    return progress, standard


def test_progress_streams_and_final_item_reuses_reserved_identity(service, conn_id):
    _open_response(service, conn_id)

    events: list[Any] = []
    events += service.dispatch_pipeline_event(conn_id, _progress("show_document(ti"))
    events += service.dispatch_pipeline_event(conn_id, _progress("tle='T', content='C')", name="show_document"))
    events += service.dispatch_pipeline_event(
        conn_id,
        AssistantOutputEvent(
            response_key=_RESPONSE_KEY,
            output_sequence=0,
            parts=[_tool("c1")],
        ),
    )
    events += service.dispatch_pipeline_event(
        conn_id,
        AssistantOutputEvent(
            response_key=_RESPONSE_KEY,
            output_sequence=1,
            parts=[AssistantTextPart(text="after")],
        ),
    )
    events += service.finish_response(conn_id, status="completed")

    progress, standard = _split(events)
    assert [event.delta for event in progress] == [
        "show_document(ti",
        "tle='T', content='C')",
    ]
    # The first delta arrives before the opening parenthesis, so the name is
    # only known from the second delta on; the client adopts the first
    # non-null name it receives.
    assert [event.name for event in progress] == [None, "show_document"]

    added = next(event for event in standard if event.type == "response.output_item.added")
    assert added.item.type == "function_call"
    assert added.item.name == "show_document"
    for event in progress:
        assert event.item_id == added.item.id
        assert event.call_id == added.item.call_id
        assert event.output_index == added.output_index

    args_done = next(event for event in standard if event.type == "response.function_call_arguments.done")
    assert args_done.item_id == added.item.id
    assert args_done.call_id == added.item.call_id
    assert args_done.output_index == added.output_index

    done = next(event for event in standard if event.type == "response.done")
    assert all(event.response_id == done.response.id for event in progress)

    assert_openai_schema(standard)
    assert_response_lifecycle_contract(standard, wants_audio=False, expected_status="completed")


def test_progress_with_early_side_channel_does_not_duplicate_items(service, conn_id):
    """Progress, then the ready side channel, then the ordered copy.

    The ready flush emits added/done and consumes the reservation; the ordered
    copy must dedupe instead of emitting a second item.
    """
    _open_response(service, conn_id)

    events: list[Any] = []
    events += service.dispatch_pipeline_event(conn_id, _progress("show_document(", name="show_document"))
    events += service.dispatch_pipeline_event(
        conn_id,
        AssistantToolCallReadyEvent(
            response_key=_RESPONSE_KEY,
            output_sequence=0,
            part=_tool("c1"),
        ),
    )
    events += service.dispatch_pipeline_event(
        conn_id,
        AssistantOutputEvent(
            response_key=_RESPONSE_KEY,
            output_sequence=0,
            parts=[_tool("c1")],
        ),
    )
    events += service.finish_response(conn_id, status="completed")

    progress, standard = _split(events)
    assert len(progress) == 1

    added = [event for event in standard if event.type == "response.output_item.added"]
    args_done = [event for event in standard if event.type == "response.function_call_arguments.done"]
    item_done = [event for event in standard if event.type == "response.output_item.done"]
    function_items = [
        event for event in added if event.item.type == "function_call"
    ]
    assert len(function_items) == 1, "the tool call must be exposed exactly once"
    assert len(args_done) == 1
    function_item_dones = [
        event for event in item_done if event.item.id == function_items[0].item.id
    ]
    assert len(function_item_dones) == 1

    assert progress[0].item_id == function_items[0].item.id
    assert progress[0].call_id == function_items[0].item.call_id
    assert progress[0].output_index == function_items[0].output_index

    done = next(event for event in standard if event.type == "response.done")
    assert done.response.output[0].type == "function_call"

    assert_openai_schema(standard)
    assert_response_lifecycle_contract(standard, wants_audio=False, expected_status="completed")
