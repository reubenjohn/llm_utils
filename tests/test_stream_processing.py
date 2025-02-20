from typing import Iterable, List
import pytest  # type: ignore[import]
from llm.stream_processing import (
    MessageStream,
    ToolStream,
    FinalResponse,
    StreamElement,
    parse_streaming_response,
)


def assert_stream(stream: Iterable[StreamElement], expected_stream: List[StreamElement]):
    stream = list(stream)
    for actual in stream:
        if isinstance(actual, MessageStream):
            actual.cache()

    assert stream == expected_stream


def message_stream(sender, role, content_stream):
    stream = MessageStream(sender=sender, role=role, content_stream=content_stream)
    stream._content = "".join(content_stream)
    return stream


JUNK = {"foo": "bar", "tool_calls": None}


def test_message_stream():
    content_iter = iter(["Hello! ", "How are you?"])
    stream = MessageStream(sender="user1", role="user", content_stream=content_iter)
    assert stream.sender == "user1"
    assert stream.role == "user"
    assert stream.is_cached is False
    assert stream.content_stream is content_iter
    stream.cache()
    assert stream.content_stream == ["Hello! ", "How are you?"]
    assert stream.is_cached is True
    assert stream.content == "Hello! How are you?"

    stream = MessageStream(
        sender="user1", role="user", content_stream=iter(["Hello! ", "How are you?"])
    )
    assert stream.is_cached is False
    assert stream.content == "Hello! How are you?"
    assert stream.content_stream == ["Hello! ", "How are you?"]
    assert stream.is_cached is True


def test_parse_streaming_response_message_stream():
    response = [
        {"sender": "user1", "role": "user", "content": "Hello! "},
        {"content": "How are you?"},
        {"delim": "end"},
    ]
    expected = [message_stream("user1", "user", ["Hello! ", "How are you?"])]
    assert_stream(parse_streaming_response(response), expected)

    response = [
        JUNK,
        {"sender": "user1", "role": "user", "content": "Hello! "},
        JUNK,
        {"content": "How are you?"},
        JUNK,
        {"delim": "end"},
        JUNK,
    ]
    assert_stream(parse_streaming_response(response), expected)


def test_parse_streaming_response_tool_stream():
    response = [
        {"sender": "user2", "tool_calls": [{"function": {"name": "tool1"}}]},
        {"sender": "user2", "tool_calls": [{"function": {"name": ""}}]},
        {"delim": "end"},
    ]
    expected = [ToolStream(sender="user2", tool_name="tool1")]
    assert_stream(parse_streaming_response(response), expected)
    response = [
        JUNK,
        {"sender": "user2", "tool_calls": [{"function": {"name": "tool1"}}]},
        JUNK,
        {"delim": "end"},
        JUNK,
    ]
    assert_stream(parse_streaming_response(response), expected)


def test_parse_streaming_response_final_response():
    response = [
        {"response": {"status": "complete"}},
        {"delim": "end"},
    ]
    expected = [FinalResponse(response={"status": "complete"})]
    assert_stream(parse_streaming_response(response), expected)

    response = [
        JUNK,
        {"response": {"status": "complete"}},
        JUNK,
        {"delim": "end"},
        JUNK,
    ]
    assert_stream(parse_streaming_response(response), expected)


def test_parse_streaming_response_mixed():
    response = [
        {"sender": "user", "role": "user", "content": "Hello"},
        {"content": "?"},
        {"delim": "end"},
        {"sender": "agent", "tool_calls": [{"function": {"name": "tool1"}}]},
        {"response": {"status": "complete"}},
    ]
    expected = [
        message_stream("user", "user", ["Hello", "?"]),
        ToolStream(sender="agent", tool_name="tool1"),
        FinalResponse(response={"status": "complete"}),
    ]
    assert_stream(parse_streaming_response(response), expected)

    response = [
        JUNK,
        {"sender": "user", "role": "user", "content": "Hello"},
        JUNK,
        {"content": "?"},
        JUNK,
        {"delim": "end"},
        JUNK,
        {"sender": "agent", "tool_calls": [{"function": {"name": "tool1"}}]},
        JUNK,
        {"response": {"status": "complete"}},
        JUNK,
    ]
    assert_stream(parse_streaming_response(response), expected)


def test_parse_streaming_response_empty():
    response = []
    result = list(parse_streaming_response(response))
    assert len(result) == 0


def test_parse_streaming_response_no_delim():
    response = [{"sender": "user1", "content": "Hello!"}, {"response": {"status": "complete"}}]
    with pytest.raises(ValueError, match="Response should not be present before 'delim'"):
        list(parse_streaming_response(response))


def test_parse_streaming_response_tool_calls():
    response = [
        {"sender": "user1", "content": "Hello!"},
        {"sender": "user1", "tool_calls": [{"function": {"name": "tool1"}}]},
        {"delim": "end"},
    ]
    with pytest.raises(
        ValueError, match="Tool calls should not be present in message content stream"
    ):
        list(parse_streaming_response(response))
