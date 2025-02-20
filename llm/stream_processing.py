from dataclasses import dataclass
import itertools
from typing import Iterable, Union


@dataclass
class MessageStream:
    sender: str
    role: str
    content_stream: Iterable[str]
    _content: str = ""

    def cache(self):
        if self.is_cached:
            return
        self.content_stream = list(self.content_stream)
        self._content = "".join(self.content_stream)

    @property
    def is_cached(self):
        return isinstance(self.content_stream, list)

    @property
    def content(self):
        self.cache()
        return self._content


@dataclass
class ToolStream:
    sender: str
    tool_name: str


@dataclass
class FinalResponse:
    response: dict


StreamElement = Union[MessageStream, ToolStream, FinalResponse]


def stream_message_content(response):
    for chunk in response:
        if "content" in chunk and chunk["content"] is not None:
            yield chunk["content"]

        if "delim" in chunk and chunk["delim"] == "end":
            return

        if "tool_calls" in chunk and chunk["tool_calls"] is not None:
            raise ValueError("Tool calls should not be present in message content stream")

        if "response" in chunk:
            raise ValueError(
                "Response should not be present before 'delim' in message content stream"
            )


def parse_streaming_response(response: Iterable[dict]) -> Iterable[StreamElement]:
    last_sender: str = ""
    last_role: str = ""

    response = iter(response)
    for chunk in response:
        if "sender" in chunk:
            last_sender = chunk["sender"]
            last_role = chunk.get("role", "")

        if "content" in chunk and chunk["content"] is not None:
            unconsumed_response = itertools.chain([chunk], response)
            message_stream = MessageStream(
                last_sender, last_role, stream_message_content(unconsumed_response)
            )
            yield message_stream
            message_stream.cache()

        if "tool_calls" in chunk and chunk["tool_calls"] is not None:
            for tool_call in chunk["tool_calls"]:
                f = tool_call["function"]
                name = f["name"]
                if not name:
                    continue
                yield ToolStream(last_sender, name)

        if "response" in chunk:
            yield FinalResponse(chunk["response"])
