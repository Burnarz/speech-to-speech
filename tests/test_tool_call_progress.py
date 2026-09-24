"""Tests for streaming tool-call progress while the code block is open.

Covers the layers that produce, route, and commit progress:

* ``BaseLanguageModelHandler._process_printable_text`` emits
  ``AssistantToolCallProgressPart`` chunks while a ``<code>`` block is still
  incomplete, throttled to bounded deltas.
* ``LMOutputProcessor`` routes those parts to the side channel only (no TTS,
  no ordered assistant output, no chat history).
* ``LanguageModelHandler._commit_ordered_output`` skips side-channel progress
  parts so chat history only ever records text and tool calls.
"""

import json
from queue import Queue

from openai.types.responses import ResponseFunctionToolCall

from speech_to_speech.LLM.chat import Chat, make_user_message
from speech_to_speech.LLM.language_model import (
    TOOL_CALL_PROGRESS_MIN_CHARS,
    LanguageModelHandler,
    StreamContext,
)
from speech_to_speech.LLM.lm_output_processor import LMOutputProcessor
from speech_to_speech.LLM.tool_call.function_tool import FunctionTool
from speech_to_speech.LLM.tool_call.tool_prompt import END_CODE, ENTER_CODE, build_block_regex
from speech_to_speech.pipeline.events import AssistantToolCallProgressEvent
from speech_to_speech.pipeline.messages import (
    AssistantTextPart,
    AssistantToolCallPart,
    AssistantToolCallProgressPart,
    LLMResponseChunk,
)


def _ctx() -> StreamContext:
    return StreamContext(
        function_tools=[
            FunctionTool(
                type="function",
                name="show_document",
                description="Show a document to the user.",
                parameters={
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["content"],
                },
            )
        ],
        block_regex=build_block_regex(),
        enter_code=ENTER_CODE,
        end_code=END_CODE,
    )


def _handler() -> LanguageModelHandler:
    return object.__new__(LanguageModelHandler)


def _progress_parts(chunks: list[LLMResponseChunk]) -> list[AssistantToolCallProgressPart]:
    return [part for chunk in chunks for part in chunk.parts if isinstance(part, AssistantToolCallProgressPart)]


def test_open_block_streams_progress_with_name_and_delta():
    handler = _handler()
    ctx = _ctx()
    text = f"{ENTER_CODE}show_document(title='Recette'"

    chunks, tools, remaining = handler._process_printable_text(text, None, [], ctx)

    progress = _progress_parts(chunks)
    assert len(progress) == 1
    assert progress[0].name == "show_document"
    assert progress[0].delta == "show_document(title='Recette'"
    assert tools == []
    assert remaining == text
    assert ctx.tool_call_progress_sent == len(text) - len(ENTER_CODE)


def test_progress_is_throttled_to_min_chars():
    handler = _handler()
    ctx = _ctx()
    first = f"{ENTER_CODE}show_document(title='A'"

    chunks, _, _ = handler._process_printable_text(first, None, [], ctx)
    assert len(_progress_parts(chunks)) == 1

    small = first + "b"
    chunks, _, remaining = handler._process_printable_text(small, None, [], ctx)
    assert _progress_parts(chunks) == []
    assert remaining == small

    big = small + "c" * (TOOL_CALL_PROGRESS_MIN_CHARS + 1)
    chunks, _, _ = handler._process_printable_text(big, None, [], ctx)
    progress = _progress_parts(chunks)
    assert len(progress) == 1
    assert progress[0].delta == "b" + "c" * (TOOL_CALL_PROGRESS_MIN_CHARS + 1)
    assert ctx.tool_call_progress_sent == len(big) - len(ENTER_CODE)


def test_completed_block_resets_progress_and_yields_tool_call():
    handler = _handler()
    ctx = _ctx()
    head = f"{ENTER_CODE}show_document(title='Recette', content='une ligne')"
    full = head + f"\n{END_CODE}"

    chunks, tools, _ = handler._process_printable_text(head, None, [], ctx)
    assert _progress_parts(chunks)

    chunks, tools, remaining = handler._process_printable_text(full, None, tools, ctx)

    assert _progress_parts(chunks) == []
    assert remaining == ""
    assert ctx.tool_call_progress_sent == 0
    assert len(tools) == 1
    assert tools[-1].name == "show_document"
    assert json.loads(tools[-1].arguments) == {"title": "Recette", "content": "une ligne"}


def test_triple_quoted_multiline_content_streams_and_parses():
    handler = _handler()
    ctx = _ctx()
    doc = "line one\nline two\n  indented"
    block = f'{ENTER_CODE}show_document(title="Doc", content="""{doc}""")\n{END_CODE}'

    cut = block.index(END_CODE)
    chunks, tools, _ = handler._process_printable_text(block[:cut], None, [], ctx)
    progress = _progress_parts(chunks)
    assert progress
    assert progress[0].name == "show_document"
    assert '"""' in progress[0].delta

    chunks, tools, _ = handler._process_printable_text(block, None, tools, ctx)
    assert tools
    assert json.loads(tools[-1].arguments) == {"title": "Doc", "content": doc}
    assert ctx.tool_call_progress_sent == 0


def test_progress_name_is_none_until_callable():
    handler = _handler()
    ctx = _ctx()
    text = f"{ENTER_CODE}show"

    chunks, _, _ = handler._process_printable_text(text, None, [], ctx)

    progress = _progress_parts(chunks)
    assert len(progress) == 1
    assert progress[0].name is None
    assert progress[0].delta == "show"


def test_lm_processor_routes_progress_to_side_channel_only():
    processor = LMOutputProcessor.__new__(LMOutputProcessor)
    side: Queue = Queue()
    processor.setup(text_output_queue=side)

    chunk = LLMResponseChunk(
        parts=[AssistantToolCallProgressPart(name="show_document", delta="partial content")],
        response_key="rk1",
        turn_id="turn_1",
        turn_revision=2,
    )

    out = list(processor.process(chunk))

    assert out == []
    event = side.get_nowait()
    assert isinstance(event, AssistantToolCallProgressEvent)
    assert event.name == "show_document"
    assert event.delta == "partial content"
    assert event.response_key == "rk1"
    assert event.turn_id == "turn_1"
    assert event.turn_revision == 2
    assert side.empty()


def test_commit_ordered_output_skips_side_channel_progress_parts():
    chat = Chat(5)
    chat.add_item(make_user_message("go"))
    tool_call = ResponseFunctionToolCall(
        type="function_call",
        id="fc_doc",
        call_id="call_doc",
        name="show_document",
        arguments="{}",
    )
    parts = [
        AssistantTextPart(text="before"),
        AssistantToolCallProgressPart(name="show_document", delta="partial"),
        AssistantTextPart(text="lead in"),
        AssistantToolCallPart(tool=tool_call),
    ]

    committed = LanguageModelHandler._commit_ordered_output(chat, parts, wants_audio=True)

    assert committed
    output = chat.buffer[1:]
    assert [item.type for item in output] == ["message", "function_call"]
    assert output[0].content[0].text == "before lead in"
    assert output[1].name == "show_document"
    assert output[1].call_id == "call_doc"
