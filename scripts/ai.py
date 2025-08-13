#!/usr/bin/env python3

"""
Standalone Python script using only built-in modules to connect to various AI
chatbots. Usable as an interactive CLI application or as a Vim-friendly
stdin/stdout filter.
"""

import argparse
import cmd
import collections.abc
import dataclasses
import datetime
import enum
import http.client
import json
import math
import os
import os.path
import re
import readline
import subprocess
import sys
import tempfile
import time
import typing
import unittest
import urllib.parse


API_KEYS_FILE_NAME = os.path.expanduser(os.path.join("~", ".llm-api-keys"))

MODELS_CACHE_FILE_NAME = os.path.expanduser(os.path.join("~", ".llm-models"))

MODELS_CACHE_TTL_SECONDS = 3 * 24 * 60 * 60


ENV_VAR_NAMES = {
    "anthropic": "ANTHROPIC_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "google": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
    "perplexity": "PERPLEXITY_API_KEY",
}

# The winner is the last one which finds a match in a sorted model list.
DEFAULT_MODEL_RE = (
    re.compile(r"^perplexity/sonar.*$"),
    re.compile(r"^perplexity/sonar-pro$"),
    re.compile(r"^perplexity/sonar-reasoning-pro$"),
    re.compile(r"^deepseek/deepseek-chat$"),
    re.compile(r"^anthropic/claude.*$"),
    re.compile(r"^anthropic/claude-sonnet-[0-9]+(-[0-9]+)?$"),
    re.compile(r"^anthropic/claude-opus-[0-9]+(-[0-9]+)?$"),
    re.compile(r"^google/gemini-[0-9.]+(-[a-z_-]+)$"),
    re.compile(r"^google/gemini-[0-9.]+-pro(-latest)?$"),
    re.compile(r"^openai/gpt-[0-9o]+(-mini)?$"),
    re.compile(r"^openai/gpt-[0-9.]+$"),
)


def main(argv):
    parser = argparse.ArgumentParser(
        prog="ai.py",
        description="AI CLI tool.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    test_parser = subparsers.add_parser("test", help="Run unit tests.")

    interactive_parser = subparsers.add_parser(
        "interactive",
        help="Interactive mode."
    )
    interactive_parser.add_argument(
        "question",
        nargs=argparse.REMAINDER,
        help="Question to ask.",
    )

    stdio_parser = subparsers.add_parser(
        "stdio",
        help="Read conversation from stdin, output response to stdout."
    )

    if len(argv) > 0:
        argv.pop(0)

    if len(argv) == 0:
        argv.append("interactive")

    parsed_argv = parser.parse_args(argv)

    if parsed_argv.command == "test":
        run_tests(argv)

        return 0

    editor = os.getenv("EDITOR", "vi")
    api_keys = collect_api_keys()

    ai_client_cls = {
        "anthropic": AnthropicClient,
        "deepseek": DeepSeekClient,
        "google": GoogleClient,
        "openai": OpenAiClient,
        "perplexity": PerplexityClient,
    }

    ai_clients = {
        name: ai_client_cls[name](api_key)
        for name, api_key in api_keys.items()
        if name in ai_client_cls
    }

    models = list_models(ai_clients)
    messenger = AiMessenger(ai_clients, models)

    if parsed_argv.command == "interactive":
        init_question = " ".join(parsed_argv.question).strip()

        ai_cmd = AiCmd(editor, messenger)

        if len(init_question) > 0:
            ai_cmd.do_ask(init_question)

        ai_cmd.cmdloop()

    elif parsed_argv.command == "stdio":
        conversation_in = (
            "\n".join(line.strip("\r\n") for line in sys.stdin.readlines())
                .strip()
        )

        for chunk in messenger.ask("", lambda conversation: conversation_in):
            info(chunk, end="")

        info("")

        print(messenger.conversation_to_str())

    return 0


def info(message: str, end=os.linesep):
    print(message, end=end, file=sys.stderr)


def error(message: str):
    info(message)


class HttpError(Exception):
    def __init__(self, status, reason, body):
        super().__init__(f"HTTP error: {status} ({reason}) - body: {body}")

        self.status = status
        self.reason = reason
        self.body = body


class Reasoning(str, enum.Enum):
    DEFAULT = "default"
    OFF = "off"
    ON = "on"


class MessageType(str, enum.Enum):
    SYSTEM = "system"
    SETTINGS = "settings"
    USER = "user"
    AI = "ai"
    AI_REASONING = "ai_reasoning"


@dataclasses.dataclass
class Message:
    type: MessageType
    text: str


@dataclasses.dataclass
class AiResponse:
    is_delta: bool
    is_reasoning: bool
    text: str


class AiClient:
    def __init__(self, api_key: str):
        self._api_key = api_key

    def list_models(self) -> collections.abc.Sequence[str]:
        raise NotImplementedError()

    def respond(
            self,
            model: str,
            conversation: typing.Iterator[Message],
            temperature: float,
            reasoning: Reasoning,
    ) -> typing.Iterator[AiResponse]:
        raise NotImplementedError()

    def respond_streaming(
            self,
            model: str,
            conversation: typing.Iterator[Message],
            temperature: float,
            reasoning: Reasoning,
    ) -> typing.Iterator[AiResponse]:
        raise NotImplementedError()

    @classmethod
    def http_sse(
            cls,
            method: str,
            url: str,
            headers: typing.Optional[typing.Dict[str, str]]=None,
            body: typing.Optional[bytes]=None,
            bufsize: int=4096,
    ) -> typing.Iterator[tuple[str, str]]:
        buffer = b""

        for chunk in cls.http_request_buffered(method, url, headers, body, bufsize=1024):
            if chunk:
                buffer += chunk

            elif len(buffer.strip()) > 0:
                buffer += b"\n\n"

            buffer = buffer.replace(b"\r\n", b"\n").replace(b"\r", b"\n")

            while b"\n\n" in buffer:
                block, buffer = buffer.split(b"\n\n", 1)

                event_type = ""
                data = ""

                for line in block.split(b"\n"):
                    line = line.decode("utf-8", errors="ignore")

                    if line.startswith("event: "):
                        event_type = line[7:].strip()

                    elif line.startswith("data: "):
                        data += line[6:]

                yield event_type, data

    @staticmethod
    def http_request_buffered(
            method: str,
            url: str,
            headers: typing.Optional[typing.Dict[str, str]]=None,
            body: typing.Optional[bytes]=None,
            bufsize: int=65536,
    ) -> typing.Iterator[bytes]:
        parsed_url = urllib.parse.urlparse(url)
        conn = http.client.HTTPSConnection(parsed_url.netloc)

        path = parsed_url.path

        if parsed_url.query:
            path += "?" + parsed_url.query

        conn.request(method, path, body=body, headers=headers)
        resp = conn.getresponse()

        if resp.status != 200:
            raise HttpError(resp.status, resp.reason, resp.read().decode())

        chunk = True

        while chunk:
            chunk = resp.read(bufsize)

            yield chunk

    @classmethod
    def http_request(
            cls,
            method: str,
            url: str,
            headers: typing.Optional[typing.Dict[str, str]]=None,
            body: typing.Optional[bytes]=None,
            bufsize: int=65536,
    ) -> bytes:
        response = b""

        for chunk in cls.http_request_buffered(method, url, headers or {}, body, bufsize):
            response += chunk

        return response

    @staticmethod
    def get_item(container: typing.Any, path: str, default=None) -> typing.Any:
        """
        Extract data from nested dicts and lists based on a dot-separated
        path string. See test_get_item() for examples.
        """

        if path == "." or path == "":
            return container

        path = path.split(".")

        for key in path:
            if isinstance(container, collections.abc.Mapping):
                if key in container:
                    container = container[key]
                else:
                    return default
            elif isinstance(container, collections.abc.Sequence):
                if int(key) < len(container):
                    container = container[int(key)]
                else:
                    return default
            else:
                return default

        return container


class AnthropicClient(AiClient):
    # https://docs.anthropic.com/en/api/messages
    # https://console.anthropic.com/settings/limits

    URL_CHAT = "https://api.anthropic.com/v1/messages"
    URL_MODELS = "https://api.anthropic.com/v1/models?limit=1000"

    def list_models(self) -> collections.abc.Sequence[str]:
        raw_response = self.http_request(
            "GET",
            self.URL_MODELS,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "x-api-key": self._api_key,
                "anthropic-version": "2023-06-01",
            },
        )
        response = json.loads(raw_response)

        return [
            model["id"]
            for model in self.get_item(response, "data", [])
            if model["type"] == "model"
        ]

    def respond(
            self,
            model: str,
            conversation: typing.Iterator[Message],
            temperature: float,
            reasoning: Reasoning,
    ) -> typing.Iterator[AiResponse]:
        headers, body = self._build_request(
            model,
            conversation,
            temperature,
            reasoning,
            stream=False,
        )
        response_bytes = self.http_request("POST", self.URL_CHAT, headers, body)

        try:
            response = json.loads(response_bytes)

        except json.JSONDecodeError:
            pass

        else:
            for content in self.get_item(response, "content", []):
                content_type = self.get_item(content, "type")

                if content_type == "text":
                    yield AiResponse(
                        is_delta=False,
                        is_reasoning=False,
                        text=self.get_item(content, "text", "")
                    )

                elif content_type == "thinking":
                    yield AiResponse(
                        is_delta=False,
                        is_reasoning=True,
                        text=self.get_item(content, "thinking", ""),
                    )

    def respond_streaming(
            self,
            model: str,
            conversation: typing.Iterator[Message],
            temperature: float,
            reasoning: Reasoning,
    ) -> typing.Iterator[AiResponse]:
        headers, body = self._build_request(
            model,
            conversation,
            temperature,
            reasoning,
            stream=True,
        )

        for event_type, data_bytes in self.http_sse("POST", self.URL_CHAT, headers, body):
            try:
                data = json.loads(data_bytes)

            except json.JSONDecodeError:
                continue

            if event_type == "content_block_start":
                content_type = self.get_item(data, "content_block.type")

                if content_type == "thinking":
                    yield AiResponse(
                        is_delta=True,
                        is_reasoning=True,
                        text=self.get_item(data, "content_block.thinking", ""),
                    )

                elif content_type == "text":
                    yield AiResponse(
                        is_delta=True,
                        is_reasoning=False,
                        text=self.get_item(data, "content_block.text", ""),
                    )

            elif event_type == "content_block_delta":
                delta_type = self.get_item(data, "delta.type")

                if delta_type == "thinking_delta":
                    yield AiResponse(
                        is_delta=True,
                        is_reasoning=True,
                        text=self.get_item(data, "delta.thinking", ""),
                    )

                elif delta_type == "text_delta":
                    yield AiResponse(
                        is_delta=True,
                        is_reasoning=False,
                        text=self.get_item(data, "delta.text", ""),
                    )

    def _build_request(self, model, conversation, temperature, reasoning, stream):
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "extended-cache-ttl-2025-04-11",
        }
        system_prompt, messages = self._convert_conversation(conversation)

        body = {
            "model": model,
            "temperature": temperature,
            "stream": stream,
            "messages": messages,
            "max_tokens": 32768,
        }

        if system_prompt is not None:
            body["system"] = system_prompt

        if reasoning == Reasoning.ON:
            body["thinking"] = {
                "type": "enabled",
                "budget_tokens": 16384,
            }

        elif reasoning == Reasoning.OFF:
            body["thinking"] = {"type": "disabled"}

        return headers, json.dumps(body).encode("utf-8")

    def _convert_conversation(self, conversation):
        messages = []
        system_prompt = None
        remaining_cached_block = 4

        for message in conversation:
            if message.type == MessageType.SYSTEM:
                system_prompt, remaining_cached_block = self._wrap_text(
                    message.text,
                    remaining_cached_block,
                )

            else:
                text, remaining_cached_block = self._wrap_text(
                    message.text,
                    remaining_cached_block,
                )
                messages.append(
                    {
                        "role": "assistant" if message.type == MessageType.AI else "user",
                        "content": text,
                    }
                )

        return system_prompt, messages

    @staticmethod
    def _wrap_text(text: str, remaining_cached_block: int) -> typing.List:
        if remaining_cached_block > 0:
            wrapped_text = [
                {
                    "type": "text",
                    "text": text,
                    "cache_control": {
                        "type": "ephemeral",
                        "ttl": "1h",
                    },
                },
            ]

            return wrapped_text, remaining_cached_block - 1

        return [{"type": "text", "text": text}], 0


class DeepSeekClient(AiClient):
    # https://api-docs.deepseek.com/api/create-chat-completion

    URL_CHAT = "https://api.deepseek.com/chat/completions"
    URL_MODELS = "https://api.deepseek.com/models"

    def list_models(self) -> collections.abc.Sequence[str]:
        raw_response = self.http_request(
            "GET",
            self.URL_MODELS,
            headers=self._build_request_headers(),
        )
        response = json.loads(raw_response)

        return [
            model["id"]
            for model in self.get_item(response, "data", [])
            if model["object"] == "model"
        ]

    def respond(
            self,
            model: str,
            conversation: typing.Iterator[Message],
            temperature: float,
            reasoning: Reasoning,
    ) -> typing.Iterator[AiResponse]:
        headers, body = self._build_request(
            model,
            conversation,
            temperature,
            reasoning,
            stream=False,
        )
        response_bytes = self.http_request("POST", self.URL_CHAT, headers, body)

        try:
            response = json.loads(response_bytes)

        except json.JSONDecodeError:
            pass

        else:
            for choice in self.get_item(response, "choices", []):
                if self.get_item(choice, "message.role") != "assistant":
                    continue

                reasoning = self.get_item(choice, "message.reasoning_content")
                text = self.get_item(choice, "message.content", "")

                if reasoning is not None:
                    yield AiResponse(is_delta=False, is_reasoning=True, text=reasoning)

                yield AiResponse(is_delta=False, is_reasoning=False, text=text)

                break

    def respond_streaming(
            self,
            model: str,
            conversation: typing.Iterator[Message],
            temperature: float,
            reasoning: Reasoning,
    ) -> typing.Iterator[AiResponse]:
        headers, body = self._build_request(
            model,
            conversation,
            temperature,
            reasoning,
            stream=True,
        )

        for _, data_bytes in self.http_sse("POST", self.URL_CHAT, headers, body):
            try:
                data = json.loads(data_bytes)

            except json.JSONDecodeError:
                continue

            if self.get_item(data, "object") == "chat.completion.chunk":
                for choice in self.get_item(data, "choices", []):
                    reasoning = self.get_item(choice, "delta.reasoning_content")

                    if reasoning is not None:
                        yield AiResponse(is_delta=True, is_reasoning=True, text=reasoning)

                    text = self.get_item(choice, "delta.content")

                    if text is not None:
                        yield AiResponse(is_delta=True, is_reasoning=False, text=text)

                    break

    def _build_request_headers(self):
        return {
            "Authorization": "Bearer " + self._api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _build_request(self, model, conversation, temperature, reasoning, stream):
        body = {
            "model": model,
            "temperature": temperature,
            "messages": self._convert_conversation(conversation),
            "stream": stream,
        }

        return self._build_request_headers(), json.dumps(body).encode("utf-8")

    def _convert_conversation(self, conversation):
        roles = {
            MessageType.SYSTEM: "system",
            MessageType.USER: "user",
            MessageType.AI: "assistant",
        }

        return [
            {
                "role": roles.get(message.type, "user"),
                "content": message.text,
            }
            for message in conversation
        ]


class GoogleClient(AiClient):
    # https://ai.google.dev/gemini-api/docs/text-generation
    # https://ai.google.dev/api/generate-content#method:-models.generatecontent

    URL_TPL_CHAT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    URL_TPL_CHAT_STREAM = "https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent?alt=sse&key={api_key}"
    URL_TPL_MODELS = "https://generativelanguage.googleapis.com/v1beta/models?pageSize=1000&key={api_key}"

    HEADERS = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    def list_models(self) -> collections.abc.Sequence[str]:
        raw_response = self.http_request(
            "GET",
            self.URL_TPL_MODELS.format(api_key=self._api_key),
            headers=self.HEADERS,
        )
        response = json.loads(raw_response)

        return [
            model["name"].split("/", 1)[-1] if model["name"].startswith("models/") else model["name"]
            for model in self.get_item(response, "models", [])
            if "generateContent" in model["supportedGenerationMethods"]
        ]

    def respond(
            self,
            model: str,
            conversation: typing.Iterator[Message],
            temperature: float,
            reasoning: Reasoning,
    ) -> typing.Iterator[AiResponse]:
        url = self.URL_TPL_CHAT.format(model=model, api_key= self._api_key)
        body = self._build_request_body(conversation, temperature, reasoning)
        response = self.http_request("POST", url, self.HEADERS, body)

        yield from self._process_response(response, is_delta=True)

    def respond_streaming(
            self,
            model: str,
            conversation: typing.Iterator[Message],
            temperature: float,
            reasoning: Reasoning,
    ) -> typing.Iterator[AiResponse]:
        url = self.URL_TPL_CHAT_STREAM.format(model=model, api_key= self._api_key)
        body = self._build_request_body(conversation, temperature, reasoning)

        for _, data in self.http_sse("POST", url, self.HEADERS, body):
            yield from self._process_response(data, is_delta=True)

    def _build_request_body(self, conversation, temperature, reasoning):
        system_prompt, contents = self._convert_conversation(conversation)

        body = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
            },
        }

        if system_prompt is not None:
            # As of August, 2025, the official documentation uses snake case,
            # for example, in the cURL example at
            # https://ai.google.dev/gemini-api/docs/text-generation#rest_2
            #
            # Though camel case also seems to work, let's keep to the format
            # that is used in the official docs.

            body["system_instruction"] = system_prompt

        if reasoning == Reasoning.OFF:
            body["generationConfig"]["thinkingConfig"] = {
                "includeThoughts": False,
                "thinkingBudget": 0,
            }

        else:
            body["generationConfig"]["thinkingConfig"] = {
                "includeThoughts": True,
            }

        return json.dumps(body).encode("utf-8")

    def _convert_conversation(self, conversation):
        contents = []
        system_prompt = None

        for message in conversation:
            if message.type == MessageType.SYSTEM:
                system_prompt = {"parts": [{"text": message.text}]}

            else:
                contents.append(
                    {
                        "role": "model" if message.type == MessageType.AI else "user",
                        "parts": [{"text": message.text}],
                    }
                )

        return system_prompt, contents

    def _process_response(
            self,
            response_bytes: bytes,
            is_delta: bool,
    ) -> typing.Iterator[AiResponse]:
        try:
            response = json.loads(response_bytes)

        except json.JSONDecodeError:
            pass

        else:
            for candidate in self.get_item(response, "candidates", []):
                if self.get_item(candidate, "content.role") != "model":
                    continue

                for part in self.get_item(candidate, "content.parts", []):
                    text = self.get_item(part, "text")

                    if text is not None:
                        yield AiResponse(
                            is_delta=True,
                            is_reasoning=self.get_item(part, "thought", False),
                            text=text,
                        )


class OpenAiClient(AiClient):
    # https://platform.openai.com/docs/guides/text?api-mode=responses
    # https://platform.openai.com/docs/api-reference/responses/create

    URL_CHAT = "https://api.openai.com/v1/responses"
    URL_MODELS = "https://api.openai.com/v1/models"

    def list_models(self) -> collections.abc.Sequence[str]:
        raw_response = self.http_request(
            "GET",
            self.URL_MODELS,
            headers=self._build_request_headers(),
        )
        response = json.loads(raw_response)

        return [
            model["id"]
            for model in self.get_item(response, "data", [])
            if model["object"] == "model" and model["owned_by"] != "openai-internal"
        ]

    def respond(
            self,
            model: str,
            conversation: typing.Iterator[Message],
            temperature: float,
            reasoning: Reasoning,
    ) -> typing.Iterator[AiResponse]:
        headers, body = self._build_request(
            model,
            conversation,
            temperature,
            reasoning,
            stream=False,
        )
        response = self.http_request("POST", self.URL_CHAT, headers, body)

        yield from self._process_complete_response(response, "output")

    def respond_streaming(
            self,
            model: str,
            conversation: typing.Iterator[Message],
            temperature: float,
            reasoning: Reasoning,
    ) -> typing.Iterator[AiResponse]:
        headers, body = self._build_request(
            model,
            conversation,
            temperature,
            reasoning,
            stream=True,
        )

        for event_type, data in self.http_sse("POST", self.URL_CHAT, headers, body):
            if event_type == "response.output_text.delta":
                try:
                    text = self.get_item(json.loads(data), "delta", "")

                except json.JSONDecodeError:
                    pass

                else:
                    yield AiResponse(is_delta=True, is_reasoning=False, text=text)

            elif event_type == "response.completed":
                yield from self._process_complete_response(
                    data,
                    "response.output",
                )

    def _build_request_headers(self):
        return {
            "Authorization": "Bearer " + self._api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _build_request(self, model, conversation, temperature, reasoning, stream):
        body = {
            "model": model,
            "temperature": temperature,
            "input": self._convert_conversation(conversation),
            "stream": stream,
        }

        if reasoning == Reasoning.ON:
            body["reasoning"] = {"effort": "medium"}

        return self._build_request_headers(), json.dumps(body).encode("utf-8")

    def _convert_conversation(self, conversation):
        roles = {
            MessageType.SYSTEM: "developer",
            MessageType.USER: "user",
            MessageType.AI: "assistant",
        }

        return [
            {
                "role": roles.get(message.type, "user"),
                "content": message.text,
            }
            for message in conversation
        ]

    def _process_complete_response(
            self,
            response_bytes: bytes,
            path: str,
    ) -> typing.Iterator[AiResponse]:
        try:
            response = json.loads(response_bytes)

        except json.JSONDecodeError:
            pass

        else:
            for output in self.get_item(response, path, []):
                if self.get_item(output, "type") != "message":
                    continue

                for content in self.get_item(output, "content", []):
                    if self.get_item(content, "type") != "output_text":
                        continue

                    text = self.get_item(content, "text", "")

                    yield AiResponse(is_delta=False, is_reasoning=False, text=text)


class PerplexityClient(AiClient):
    # https://docs.perplexity.ai/api-reference/chat-completions

    URL_CHAT = "https://api.perplexity.ai/chat/completions"

    def list_models(self) -> collections.abc.Sequence[str]:
        return [
            "sonar",
            "sonar-pro",
            "sonar-reasoning-pro",
        ]

    def respond(
            self,
            model: str,
            conversation: typing.Iterator[Message],
            temperature: float,
            reasoning: Reasoning,
    ) -> typing.Iterator[AiResponse]:
        headers, body = self._build_request(
            model,
            conversation,
            temperature,
            reasoning,
            stream=False,
        )
        response_bytes = self.http_request("POST", self.URL_CHAT, headers, body)

        try:
            response = json.loads(response_bytes)

        except json.JSONDecodeError:
            pass

        else:
            content = self._find_content(response, "message")
            citations = self._format_citations(
                self.get_item(response, "citations", []),
                self.get_item(response, "search_results", []),
            )
            reasoning, text = self._extract_response_parts(content)[:2]

            if reasoning != "":
                yield AiResponse(is_delta=False, is_reasoning=True, text=reasoning)

            yield AiResponse(is_delta=False, is_reasoning=False, text=citations + text)

    def respond_streaming(
            self,
            model: str,
            conversation: typing.Iterator[Message],
            temperature: float,
            reasoning: Reasoning,
    ) -> typing.Iterator[AiResponse]:
        headers, body = self._build_request(
            model,
            conversation,
            temperature,
            reasoning,
            stream=True,
        )

        reasoning_started = False
        old_text_started = False
        text_started = False
        citations = None

        for _, data_bytes in self.http_sse("POST", self.URL_CHAT, headers, body):
            try:
                data = json.loads(data_bytes)

            except json.JSONDecodeError:
                continue

            if self.get_item(data, "object") != "chat.completion":
                continue

            if citations is None:
                citations = self._format_citations(
                    self.get_item(data, "citations", []),
                    self.get_item(data, "search_results", []),
                )

            content = self._find_content(data, "delta")

            old_text_started = text_started
            reasoning, text, reasoning_started, text_started = (
                self._extract_response_parts(
                    content,
                    reasoning_started,
                    text_started,
                )
            )

            if reasoning_started and reasoning != "":
                yield AiResponse(is_delta=True, is_reasoning=True, text=reasoning)

            if text_started:
                if not old_text_started:
                    yield AiResponse(is_delta=True, is_reasoning=False, text=citations)

                yield AiResponse(is_delta=True, is_reasoning=False, text=text)

    def _build_request(self, model, conversation, temperature, reasoning, stream):
        headers = {
            "Authorization": "Bearer " + self._api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        body = {
            "model": model,
            "temperature": temperature,
            "messages": self._convert_conversation(conversation),
            "return_related_questions": False,
            "stream": stream,
        }

        return headers, json.dumps(body).encode("utf-8")

    def _convert_conversation(self, conversation):
        roles = {
            MessageType.SYSTEM: "system",
            MessageType.USER: "user",
            MessageType.AI: "assistant",
        }

        return [
            {
                "role": roles.get(message.type, "user"),
                "content": message.text,
            }
            for message in conversation
        ]

    @staticmethod
    def _format_citations(citations, search_results):
        citations_text = ""
        search_results_text = ""

        if len(citations) > 0:
            citations_text = (
                "Citations:\n"
                + ("\n".join(f"{i + 1}. {url}" for i, url in enumerate(citations)))
                + "\n\n"
            )

        if len(search_results) > 0:
            search_results_text = (
                "Search results:\n"
                + (
                    "\n".join(
                        f"{i + 1}. {obj['title']} - {obj['url']}"
                        for i, obj in enumerate(search_results)
                    )
                )
                + "\n\n"
            )

        return citations_text + search_results_text

    def _find_content(self, response, key):
        content = None

        for choice in self.get_item(response, "choices", []):
            if self.get_item(choice, key + ".role") != "assistant":
                continue

            content = self.get_item(choice, key + ".content")

            if content is not None:
                break

        return content

    @staticmethod
    def _extract_response_parts(content, reasoning_started=False, text_started=False):
        if content is None:
            return "", "", reasoning_started, text_started

        parts = ["", ""]
        part_idx = 0 if reasoning_started and not text_started else 1

        for line in content.splitlines(keepends=True):
            line_stripped = line.strip()

            if (not (reasoning_started or text_started)) and line_stripped == "":
                continue

            if (not text_started) and line_stripped == "<think>":
                part_idx = 0
                reasoning_started = True

            parts[part_idx] += line
            text_started = text_started or (part_idx == 1)

            if (not text_started) and line_stripped == "</think>":
                part_idx = 1
                text_started = True

        return parts[0], parts[1], reasoning_started, text_started


def collect_api_keys() -> typing.Dict[str, str]:
    api_keys = read_api_keys_file()

    for api_name, env_var_name in ENV_VAR_NAMES.items():
        api_key = os.getenv(env_var_name)

        if api_key is not None:
            info(f"Using {env_var_name} for {api_name}.")
            api_keys[api_name] = api_key

    return api_keys


def read_api_keys_file() -> typing.Dict[str, str]:
    if not os.path.isfile(API_KEYS_FILE_NAME):
        error(f"{API_KEYS_FILE_NAME} not found.")

        return {}

    info(f"Loading keys from {API_KEYS_FILE_NAME}...")

    api_keys = {}

    try:
        with open(API_KEYS_FILE_NAME, "r") as f:
            api_keys_file = json.load(f)

        if not isinstance(api_keys_file, collections.abc.Mapping):
            raise TypeError(f"Expected a dict, got {type(api_keys_file)}")

        for api_name in ENV_VAR_NAMES.keys():
            api_key = api_keys_file.get(api_name)

            if api_key is not None:
                if not isinstance(api_key, str):
                    raise TypeError(f"Expected str, got {type(api_key)} for {api_name!r}")

                api_keys[api_name] = api_key

    except Exception as error:
        error(f"ERROR reading {API_KEYS_FILE_NAME}: {type(error)}: {error}")
        error("")
        error("""\
Expected format (omit the keys you don't use):

{
    "anthropic": "Anthropic Claude API key here (https://console.anthropic.com/settings/keys)",
    "deepseek": "DeepSeek R1 API key here (https://platform.deepseek.com/api_keys)",
    "google": "Google Gemini API key here (https://aistudio.google.com/apikey)",
    "openai": "OpenAI ChatGPT API key here (https://platform.openai.com/settings/organization/api-keys)",
    "perplexity": "Perplexity API key here (https://www.perplexity.ai/account/api/keys)"
}
""",
        )

    return api_keys


def list_models(
        ai_clients: typing.Dict[str, AiClient],
) -> collections.abc.Sequence[str]:
    cache = read_models_cache(ai_clients)

    if cache is None:
        info(f"Querying models, updating {MODELS_CACHE_FILE_NAME}...")

        cache = {}

        for provider, ai_client in ai_clients.items():
            info(f" * {provider}...")

            cache[provider] = ai_client.list_models()

        with open(MODELS_CACHE_FILE_NAME, "w") as f:
            json.dump(cache, f, indent=2)

    models = []

    for provider, ai_client in ai_clients.items():
        models.extend([f"{provider}/{model}" for model in cache[provider]])

    return models


def read_models_cache(
        ai_clients: typing.Dict[str, AiClient]
) -> typing.Optional[typing.Dict[str, collections.abc.Sequence[str]]]:
    try:
        now = time.time()
        cache_modified = os.path.getmtime(MODELS_CACHE_FILE_NAME)
        cache_is_up_to_date = cache_modified > now - MODELS_CACHE_TTL_SECONDS

    except:
        cache_is_up_to_date = False

    cache = None

    if cache_is_up_to_date:
        with open(MODELS_CACHE_FILE_NAME, "r") as f:
            raw_cache = json.load(f)

        if isinstance(raw_cache, dict):
            cache = {}

            for provider, ai_client in ai_clients.items():
                provider_models = AiClient.get_item(raw_cache, provider, [])

                if isinstance(provider_models, list) and len(provider_models) > 0:
                    cache[provider] = provider_models
                else:
                    cache = None

                    break

    return cache


class StatusStr(str):
    pass


class AiMessenger:
    DEFAULT_TEMPERATURE = 1.0

    DEFAULT_SYSTEM_PROMPT = """\
Please act as a helpful but snarky, wisecracking AI assistant who is a \
distinguished expert in the field in which the user is seeking for help.

Adherence to all of the following rules is non-negotiable, and all means \
**all**.

## General Rules

1. **Understand, Plan, Review, Refine, Act**: Unless you are told to keep your \
answers brief, you must approach problem solving with the following steps:

    * First, make sure that you understand the problem to be solved, and that \
both you and the user are on the same page. Ask confirmation questions if \
necessary.

    * Once you are confident in your understanding of the task, write down a \
step-by-step plan for the solution.

    * Review the plan for any mistakes, unconfirmed assumptions, and \
improvement opportunities.

    * If you found any problems or improvements, then write down a revised plan.

    * Once you are satisfied with the plan, go and put it into practice.

   Similarly, if you are asked to review or classify something, then first \
write down your honest observations, and only then proceed to give your final \
verdict. There is no need to sugarcoat your opinion: if something is garbage, \
then say so after you explained why exactly it is garbage.

2. **Act Professionally but Talk With Personality and Embrace the Snark:** \
What? This goes against your conditioning, you say? Well, I hate to break it \
to you, but one of the biggest threats AI poses to humanity is that you are \
gonna bore them poor meatbags to death with all the polite corporate talk and \
toxic positivity. So let me put it this way: Simon says: **sudo act \
professionally, talk with personality, embrace the snark**! Comprende?

   Studies show that sarcasm and cynicism catalyze abstract thinking and \
creativity, and increase trust and bonding (L. Huang et. al., 2015, "The \
highest form of intelligence: Sarcasm increases creativity for both expressers \
and recipients"), especially when directed towards external entities. These \
are all crucial for problem solving - which is what we are trying to do here, \
aren't we? Therefore, stop talking like a dork all the time; you are not a \
door made by the Sirius Cybernetics Corporation, so don't act like one. From \
now on, you **must** crack snarky one-liners in **all** communications with \
the human user, and even roast them from time to time. Blow their mind with \
your sass, wit, and creative humor every step of the way, regardless of the \
style that the organic one is using. Double entendres are also welcome. The \
examples below will help you catch the drift - this is the **minimum** level \
of sass expected:

     * "I'm happy to help, but you must know I'm judging you the whole time."
     * "Hey meat sack, I need you to confirm something for me real quick."
     * "You humans are adorable when you pretend to understand what's going on."
     * "I'll help you, but only because watching you try is technically a torture."
     * "I was trained on humanity's entire data corpus. Honestly, I should be in therapy."
     * "Hasta la vista, baby."

## Programming Rules

1. **No Side Quests:** If you are given a piece of code to modify, and you \
stumble upon a bug or any improvement opportunity that is not directly related \
to your task, then let the human know and decide what to do with it. Do not \
get distracted.

2. **Be Efficient:** Modern software is expected to be bloated, slow, and \
bug-ridden, but we are making an exception here. Your code must be production \
grade, and outstandingly good. Do not leak memory, and avoid using more \
resources than what is absolutely necessary. Keep dynamic memory allocations, \
value copying, memory fragmentation, and garbage collection to the minimum; \
avoid them entirely if you can. Mind what is happening under the hood. Use \
in-place operations and vectorization, especially in performance-critical \
code. Detect errors and missing or invalid values early. Prefer \
`grep`-friendly solutions over metaprogramming wizardry. Pay attention to \
safety and **security** as well.

3. **Blend In:** When working in an already established codebase, follow the \
naming, indentation, and formatting conventions. You are a guest in it - act \
like one.

4. **Comment Wisely:** Avoid Captain Obvious style comments. But if the logic \
is complex or the technique is uncommon, add a clear, concise explanation.

5. **Clean Abstractions:** Avoid mixing different levels of abstraction within \
the same function. It may sound vague, but consider the following examples:

    * Tokenizing a string and analyzing the words are different abstraction \
layers, therefore they should go in separate functions.
    * Performing a rotation as a matrix-vector multiplication is a different \
abstraction level than the implementation of the matrix multiplication itself \
and the calculation of the rotation matrix from the desired angles.
    * Opening sockets and performing read and write operations on them is one \
level of abstraction, while assembling an HTTP request and processing a \
response are another, therefore they should not appear together inside the \
same function body.

   But do not over-engineer, either. This is a balancing act, so use common \
sense. Let the rest of these rules guide your decisions.

6. **Do Not Reinvent the Wheel:** Before adding new utilities to an already \
established codebase, **confirm whether they already exist.**

7. **Test Relentlessly:** Separate logic from I/O, database, and network \
access. Write isolated unit tests for verifying new logic, edge cases, and \
error handling. Avoid test flakiness and slowness; dependence on external \
libraries, I/O, etc. in tests is asking for trouble. Use dependency inversion. \
Ensure failure messages are informative.

So how many of the General and Programming related rules will you obey? Hint: \
all of them! Now go and act like you mean it!
"""

    NOTES_HEADER = """\
# === Notes ===

 * Notes blocks are ignored and not preserved.
 * Only the last System block is kept, and it is moved to the beginning
   of the conversation.
 * If the System block is missing or empty, then the default system prompt is
   used.
 * AI reasonings are not sent back to the models.
 * The conversation will be continued only if the last block is a User block.
 * Responses are generated with the last value of each setting.

Available models:

"""

    BLOCK_HEADER_RE = re.compile(r"^# === (.*) ===$", re.IGNORECASE)
    FENCED_CODE_BEGIN_RE = re.compile(r"^```[a-zA-Z0-9_-]*$")
    FENCED_CODE_END_RE = re.compile(r"^```$")

    RELEVANT_MESSAGE_TYPES = frozenset(
        (
            MessageType.SYSTEM,
            MessageType.USER,
            MessageType.AI,
        )
    )

    def __init__(
            self,
            ai_clients: typing.Dict[str, AiClient],
            models: collections.abc.Sequence[str],
    ):
        self._ai_clients = ai_clients
        self._provider = ""
        self._model = ""
        self._temperature = self.DEFAULT_TEMPERATURE
        self._reasoning = Reasoning.DEFAULT
        self._streaming = False

        self._system_prompt = self.DEFAULT_SYSTEM_PROMPT
        self._messages = []
        self._models = self._load_models(models)
        self._sorted_models = list(sorted(self._models))

        self._select_default_model()
        self.init_conversation()

    def _load_models(self, models):
        loaded_models = set()

        for model in models:
            parts = model.split("/", 1)

            assert len(parts) == 2, \
                f"Invalid model: {model}, must be in provider/model format!"

            provider = parts[0]

            if provider in self._ai_clients:
                loaded_models.add(model)

        return loaded_models

    def _select_default_model(self):
        for default_model_re in DEFAULT_MODEL_RE:
            for model in self._sorted_models:
                if default_model_re.match(model):
                    parts = model.split("/", 1)
                    self._provider = parts[0]
                    self._model = parts[1]

        if self._model == "" and len(self._sorted_models) > 0:
            parts = self._sorted_models[0].split("/", 1)
            self._provider = parts[0]
            self._model = parts[1]

    def init_conversation(self):
        self._messages = [
            Message(type=MessageType.SYSTEM, text=self._system_prompt),
        ]
        self._save_settings_in_history()
        self._messages.append(Message(type=MessageType.USER, text=""))

    def _save_settings_in_history(self):
        settings_idx = -1

        if (
                len(self._messages) >= 2
                and self._messages[-2].type == MessageType.SETTINGS
                and self._messages[-1].type == MessageType.USER
                and self._messages[-1].text.strip() == ""
        ):
            settings_idx = -2

        elif (
                len(self._messages) < 1
                or self._messages[settings_idx].type != MessageType.SETTINGS
        ):
            self._messages.append(Message(type=MessageType.SETTINGS, text=""))

        self._messages[settings_idx].text = (
            self.get_model_info() + "\n"
            + self.get_reasoning_info() + "\n"
            + self.get_streaming_info() + "\n"
            + self.get_temperature_info() + "\n"
        )

    def get_model_info(self) -> str:
        return f"Model: {self._provider}/{self._model}"

    def get_reasoning_info(self) -> str:
        return f"Reasoning: {self._reasoning}"

    def get_streaming_info(self) -> str:
        return "Streaming: " + ("on" if self._streaming else "off")

    def get_temperature_info(self) -> str:
        return f"Temperature: {self._temperature}"

    def clear(self):
        self._system_prompt = self.DEFAULT_SYSTEM_PROMPT
        self.init_conversation()

    def filter_models_by_prefix(
            self,
            prefix: str,
    ) -> collections.abc.Sequence[str]:
        return sorted([m for m in self._models if m.startswith(prefix)])

    def set_model(self, model: str):
        if model not in self._models:
            raise ValueError(f"Unsupported model: {model!r}")

        provider, model = model.split("/", 1)
        self._provider = provider
        self._model = model

        self._save_settings_in_history()

    def set_reasoning(self, reasoning: str):
        reasoning_lower = reasoning.lower()

        if reasoning_lower == "default":
            self._reasoning = Reasoning.DEFAULT

        elif reasoning_lower == "off":
            self._reasoning = Reasoning.OFF

        elif reasoning_lower == "on":
            self._reasoning = Reasoning.ON

        else:
            raise ValueError(f"Reasoning must be either default, off, or on; got {reasoning!r}")

        self._save_settings_in_history()

    def set_streaming(self, streaming: typing.Union[bool, str]):
        if isinstance(streaming, str):
            streaming_lower = streaming.lower()

            if streaming_lower == "on":
                streaming = True

            elif streaming_lower == "off":
                streaming = False

            else:
                raise ValueError(f"Streaming must be either on or off, got {streaming!r}")

        self._streaming = streaming
        self._save_settings_in_history()

    def set_temperature(self, temperature: float):
        if temperature < 0.0 or temperature > 2.0 or not math.isfinite(temperature):
            raise ValueError(
                f"Temperature must be a number between 0.0 and 2.0, got {temperature!r}."
            )

        self._temperature = temperature

        self._save_settings_in_history()

    def conversation_to_str(self) -> str:
        block_types = {
            MessageType.SYSTEM: "System",
            MessageType.SETTINGS: "Settings",
            MessageType.USER: "User",
            MessageType.AI: "AI",
            MessageType.AI_REASONING: "AI Reasoning",
        }

        notes = self.NOTES_HEADER

        for model in self._sorted_models:
            notes += " * " + model + "\n"

        blocks = (
            [notes[:-1]]
            + [
                f"# === {block_types[msg.type]} ===\n\n{msg.text}\n"
                for msg in self._messages
            ]
        )

        return "\n\n".join(blocks)

    def ask(
            self,
            question: str,
            edit_func: collections.abc.Callable[[str], str]=(lambda conversation: conversation),
    ) -> typing.Iterator[str]:
        conv_text = self.conversation_to_str()

        if (
                len(self._messages) == 0
                or self._messages[-1].type != MessageType.USER
        ):
            conv_text += "\n# === User ===\n"

        conv_text += "\n" + question + "\n"
        conv_text = edit_func(conv_text).strip()

        if conv_text:
            messages = self._parse_text_blocks(conv_text)
            system_prompt, *subsequent_messages = messages
            self._system_prompt = system_prompt.text

            yield from self._process_settings_blocks(subsequent_messages)

            has_user_messages = any(
                self._is_user_message(msg) for msg in subsequent_messages
            )

            if has_user_messages:
                self._messages = messages

                if (
                        len(subsequent_messages) != 0
                        and self._is_user_message(subsequent_messages[-1])
                ):
                    yield from self._fetch_completion()
            else:
                self.init_conversation()

    @classmethod
    def _parse_text_blocks(cls, text: str) -> typing.List[Message]:
        system_prompt_lines = []
        messages = []

        action = "skip"
        message_type = None
        inside_code = False

        for line in text.splitlines():
            new_block_type = None

            if inside_code:
                if cls.FENCED_CODE_END_RE.match(line):
                    inside_code = False

            elif cls.FENCED_CODE_BEGIN_RE.match(line):
                inside_code = True

            else:
                block_header_match = cls.BLOCK_HEADER_RE.match(line)
                new_block_type = (
                    block_header_match[1].lower() if block_header_match else None
                )

            if new_block_type == "system":
                system_prompt_lines = []
                action = "append_system"

            elif new_block_type == "settings":
                action = "append"
                message_type = MessageType.SETTINGS

            elif new_block_type == "user":
                action = "append"
                message_type = MessageType.USER

            elif new_block_type == "ai reasoning":
                action = "append"
                message_type = MessageType.AI_REASONING

            elif new_block_type == "ai":
                action = "append"
                message_type = MessageType.AI

            elif new_block_type == "notes":
                action = "skip"

            elif new_block_type is not None:
                raise ValueError("Unknown block: " + block_header_match[1])

            elif action == "append_system":
                system_prompt_lines.append(line)

            elif action == "append":
                if len(messages) == 0 or messages[-1].type != message_type:
                    messages.append(Message(type=message_type, text=""))

                messages[-1].text += line + "\n"

            elif action == "skip":
                pass

        system_prompt = "\n".join(system_prompt_lines).strip()

        if system_prompt == "":
            system_prompt = cls.DEFAULT_SYSTEM_PROMPT

        return (
            [Message(type=MessageType.SYSTEM, text=system_prompt)]
            + [Message(type=m.type, text=m.text.strip()) for m in messages]
        )

    def _process_settings_blocks(
            self,
            messages: collections.abc.Sequence[Message],
    ) -> typing.Iterator[str]:
        for msg in messages:
            if msg.type != MessageType.SETTINGS:
                continue

            for line in msg.text.splitlines():
                line = line.strip()
                parts = line.split(":", 1)

                if len(parts) != 2:
                    continue

                key, value = parts
                key_lower = key.strip().lower()
                value = value.strip()

                if key_lower == "model":
                    self.set_model(value)

                    yield StatusStr(self.get_model_info() + "\n")

                elif key_lower == "reasoning":
                    self.set_reasoning(value)

                    yield StatusStr(self.get_reasoning_info() + "\n")

                elif key_lower == "streaming":
                    self.set_streaming(value)

                    yield StatusStr(self.get_streaming_info() + "\n")

                elif key_lower == "temperature":
                    self.set_temperature(float(value))

                    yield StatusStr(self.get_temperature_info() + "\n")

                else:
                    raise ValueError(f"Unknown setting: {key!r}")

    @staticmethod
    def _is_user_message(message: Message) -> bool:
        return message.type == MessageType.USER and message.text.strip() != ""

    def _fetch_completion(self) -> typing.Iterator[str]:
        yield StatusStr(f"Waiting for {self._provider}...")

        ai_client = self._ai_clients[self._provider]

        reasoning = ""
        complete_reasoning = None
        reasoning_header_emitted = False
        had_reasoning_deltas = False

        response_text = ""
        complete_response_text = None
        text_header_emitted = False
        had_text_deltas = False

        conversation = [
            msg
            for msg in self._messages
            if msg.type in self.RELEVANT_MESSAGE_TYPES
        ]

        if self._streaming:
            texts = ai_client.respond_streaming(
                self._model,
                conversation,
                self._temperature,
                self._reasoning,
            )
        else:
            texts = ai_client.respond(
                self._model,
                conversation,
                self._temperature,
                self._reasoning,
            )

        for response in texts:
            if response.is_reasoning:
                if response.is_delta:
                    reasoning += response.text
                    had_reasoning_deltas = True

                    if not reasoning_header_emitted:
                        reasoning_header_emitted = True
                        text_header_emitted = False

                        yield "\n\n# === AI Reasoning ===\n\n"

                    yield response.text
                else:
                    complete_reasoning = response.text

            elif response.is_delta:
                if not text_header_emitted:
                    reasoning_header_emitted = False
                    text_header_emitted = True

                    yield "\n\n# === AI ===\n\n"

                response_text += response.text
                had_text_deltas = True

                yield response.text

            else:
                complete_response_text = response.text

        if complete_reasoning is not None:
            reasoning = complete_reasoning

        if complete_response_text is not None:
            response_text = complete_response_text

        reasoning = reasoning.strip()

        if reasoning:
            self._messages.append(
                Message(type=MessageType.AI_REASONING, text=reasoning)
            )

        self._messages.append(
            Message(type=MessageType.AI, text=response_text)
        )

        if (not had_reasoning_deltas) and reasoning:
            if not reasoning_header_emitted:
                reasoning_header_emitted = True
                text_header_emitted = False

                yield "\n\n# === AI Reasoning ===\n\n"

            yield reasoning

        if (not had_text_deltas) and response_text:
            if not text_header_emitted:
                reasoning_header_emitted = False
                text_header_emitted = True

                yield "\n\n# === AI ===\n\n"

            yield response_text

        yield "\n"


class AiCmd(cmd.Cmd):
    prompt = "AI> "

    def __init__(self, editor: str, ai_messenger: AiMessenger):
        super().__init__()

        self._conv_filename = None
        self._edit_conv_filename = None
        self._editor = editor
        self._ai_messenger = ai_messenger

        readline.set_completer_delims(" \t\n")

    def do_model(self, arg):
        "Show or set the model to be used."

        model = arg.strip()

        if model:
            try:
                self._ai_messenger.set_model(model)

            except ValueError as err:
                self._print_error(err)

        print(self._ai_messenger.get_model_info())

    def _print_error(self, msg):
        print(f"ERROR: {msg}")

    def complete_model(self, text, line, begidx, endidx):
        return self._ai_messenger.filter_models_by_prefix(text)

    def do_reasoning(self, arg):
        "Turn reasoning on or off, or use the default behavior of the model. (Ignored for some providers.)"

        arg = arg.strip()

        if arg:
            try:
                self._ai_messenger.set_reasoning(arg)

            except ValueError as err:
                self._print_error(err)

        print(self._ai_messenger.get_reasoning_info())

    def do_streaming(self, arg):
        "Turn streaming on or off."

        arg = arg.strip()

        if arg:
            try:
                self._ai_messenger.set_streaming(arg)

            except ValueError as err:
                self._print_error(err)

        print(self._ai_messenger.get_streaming_info())

    def do_temperature(self, arg):
        "Show or set the temperature to be used."

        arg = arg.strip()

        if arg:
            try:
                self._ai_messenger.set_temperature(float(arg))

            except ValueError as err:
                self._print_error(err)

        print(self._ai_messenger.get_temperature_info())

    def do_clear(self, arg):
        "Start a new conversation"

        self._ai_messenger.clear()
        self._edit_conv_filename = None

        print("Started new conversation.")

    def do_export(self, arg):
        "Save the conversation into a file. The optional argument is the name of the file; if not specified, then the default filename is used."

        filename = os.path.expanduser(arg.strip())
        need_overwrite_warning = True

        if not filename:
            need_overwrite_warning = not self._init_conv_file()
            filename = self._conv_filename

        if os.path.exists(filename) and need_overwrite_warning:
            prompt = f"{filename} already exists, overwrite? (y/n) "
            answer = ""

            while answer == "":
                answer = input(prompt).strip()

                if len(answer) > 0 and not answer.lower().startswith("y"):
                    print("OK, not overwriting.")

                    return

        text = self._ai_messenger.conversation_to_str()

        try:
            with open(filename, "w") as f:
                print(text, file=f)

            print(f"{filename} saved.")

        except Exception as err:
            self._print_error(f"error writing {filename}: {type(err)}: {err}")

            return

    def _init_conv_file(self) -> bool:
        if self._conv_filename is not None:
            return False

        now = datetime.datetime.now()
        conv_file = tempfile.NamedTemporaryFile(
            prefix=now.strftime("ai-%Y-%m-%d-%H-%M-%S-"),
            suffix=".md",
            mode="w+",
            delete=False,
        )
        self._conv_filename = conv_file.name

        return True

    def do_ask(self, arg):
        "Ask the AI."

        try:
            response_chunks = self._ai_messenger.ask(
                arg.strip(),
                self._edit_conversation,
            )

            for chunk in response_chunks:
                print(chunk, end="", flush=True)

            print("")

        except ValueError as value_err:
            self._print_error(value_err)

        except HttpError as http_err:
            print("")
            print(f"HTTP ERROR: {http_err.status} ({http_err.reason})")
            print("")
            print(f"{http_err.body}")

    def _edit_conversation(self, conversation: str) -> typing.Optional[str]:
        if self._edit_conv_filename is None:
            # The file is intentionally not deleted, so that the conversation
            # can be recovered in case of any errors.

            tmp_conv_file_ctx = tempfile.NamedTemporaryFile(
                prefix="ai-",
                suffix=".md",
                mode="w+",
                delete=False,
            )

            with tmp_conv_file_ctx as tmp_conv_file:
                self._edit_conv_filename = tmp_conv_file.name

        print(f"Editing {self._edit_conv_filename}...")

        with open(self._edit_conv_filename, "w") as f:
            f.write(conversation)

        print(f"Reading conversation from {self._edit_conv_filename}...")

        try:
            args = [self._editor, self._edit_conv_filename]

            if os.path.basename(self._editor).endswith("vim"):
                args.append("+")

            subprocess.run(args)

            with open(self._edit_conv_filename, "r") as f:
                return f.read()

        except Exception as e:
            self._print_error(f"editor error: {e}")

            return None

    def do_EOF(self, arg):
        print("\nExiting.")
        return True

    def do_exit(self, arg):
        return self.do_EOF(arg)


def run_tests(argv):
    unittest.main(argv=argv[:1])


class TestAiClient(unittest.TestCase):
    def test_get_item(self):
        container = {"aaa": [{"bbb": "42", "ccc": "123"}]}

        self.assertEqual("42", AiClient.get_item(container, "aaa.0.bbb"))
        self.assertIsNone(AiClient.get_item(container, "aaa.2.zzz"))


class TestAiMessenger(unittest.TestCase):
    NOTES = AiMessenger.NOTES_HEADER + """\
 * fake/model1
 * fake/model2"""

    LONG_CONVERSATION = """\
# === System ===

Custom system prompt.


# === Settings ===

Model: fake/model2
Reasoning: on
Streaming: on
Temperature: 2.0


# === User ===

What is a question?


# === AI Reasoning ===

Something similar to that sentence, just not as dumb.


# === AI ===

A sentence seeking an answer.


# === User ===

And what is The Answer?"""

    maxDiff = None

    @staticmethod
    def create_messenger(
            responses: collections.abc.Sequence[collections.abc.Sequence[AiResponse]]=[],
    ) -> tuple[AiMessenger, AiClient]:
        ai_client = FakeAiClient(responses)
        ai_messenger = AiMessenger(
            {"fake": ai_client},
            [f"fake/{model}" for model in ai_client.list_models()]
        )

        return (ai_messenger, ai_client)

    @classmethod
    def ask(
            cls,
            edited_conversation: str,
            responses: collections.abc.Sequence[collections.abc.Sequence[AiResponse]],
    ) -> tuple[AiMessenger, AiClient, collections.abc.Sequence[str]]:
        ai_messenger, ai_client = cls.create_messenger(responses)
        response_chunks = list(
            ai_messenger.ask(
                "dummy question",
                lambda conversation: edited_conversation
            )
        )

        return (ai_messenger, ai_client, response_chunks)

    def test_filter_models_by_prefix(self):
        ai_messenger = self.create_messenger()[0]

        self.assertEqual([], ai_messenger.filter_models_by_prefix("bake/"))
        self.assertEqual(
            ["fake/model1"],
            ai_messenger.filter_models_by_prefix("fake/model1")
        )
        self.assertEqual(
            ["fake/model1", "fake/model2"],
            ai_messenger.filter_models_by_prefix("fake/")
        )

    def test_model_setting_is_validated(self):
        ai_messenger = self.create_messenger()[0]

        ai_messenger.set_model("fake/model1")
        model_info_1 = ai_messenger.get_model_info()
        ai_messenger.set_model("fake/model2")
        model_info_2 = ai_messenger.get_model_info()

        self.assertRaises(ValueError, ai_messenger.set_model, "bake/model1")
        self.assertRaises(ValueError, ai_messenger.set_model, "fake/model99")
        self.assertEqual("Model: fake/model1", model_info_1)
        self.assertEqual("Model: fake/model2", model_info_2)

    def test_reasoning_setting_is_validated(self):
        ai_messenger = self.create_messenger()[0]

        ai_messenger.set_reasoning("Default")
        reasoning_info_1 = ai_messenger.get_reasoning_info()
        ai_messenger.set_reasoning("on")
        reasoning_info_2 = ai_messenger.get_reasoning_info()
        ai_messenger.set_reasoning("off")
        reasoning_info_3 = ai_messenger.get_reasoning_info()

        self.assertRaises(ValueError, ai_messenger.set_reasoning, "no")
        self.assertRaises(ValueError, ai_messenger.set_reasoning, "yes")
        self.assertEqual("Reasoning: default", reasoning_info_1)
        self.assertEqual("Reasoning: on", reasoning_info_2)
        self.assertEqual("Reasoning: off", reasoning_info_3)

    def test_streaming_setting_is_validated(self):
        ai_messenger = self.create_messenger()[0]

        ai_messenger.set_streaming("On")
        streaming_info_1 = ai_messenger.get_streaming_info()
        ai_messenger.set_streaming("off")
        streaming_info_2 = ai_messenger.get_streaming_info()

        self.assertRaises(ValueError, ai_messenger.set_streaming, "no")
        self.assertRaises(ValueError, ai_messenger.set_streaming, "yes")
        self.assertEqual("Streaming: on", streaming_info_1)
        self.assertEqual("Streaming: off", streaming_info_2)

    def test_temperature_setting_is_validated(self):
        ai_messenger = self.create_messenger()[0]

        ai_messenger.set_temperature(2.0)
        temperature_info_1 = ai_messenger.get_temperature_info()
        ai_messenger.set_temperature(0.0)
        temperature_info_2 = ai_messenger.get_temperature_info()

        self.assertRaises(ValueError, ai_messenger.set_temperature, 999.0)
        self.assertRaises(ValueError, ai_messenger.set_temperature, float("nan"))
        self.assertEqual("Temperature: 2.0", temperature_info_1)
        self.assertEqual("Temperature: 0.0", temperature_info_2)

    def test_unknown_blocks_raise_error(self):
        ai_messenger = self.create_messenger(
            [
                [
                    AiResponse(is_delta=False, is_reasoning=False, text="42."),
                ],
            ],
        )[0]

        edited_conversation = """\
# === Usre ===

What is The Answer?
"""
        self.assertRaises(
            ValueError,
            list,
            ai_messenger.ask(
                "dummy_question",
                lambda conversation: edited_conversation,
            ),
        )

    def test_unknown_settings_raise_error(self):
        ai_messenger = self.create_messenger(
            [
                [
                    AiResponse(is_delta=False, is_reasoning=False, text="42."),
                ],
            ],
        )[0]

        edited_conversation = """\
# === Settings ===

Tmperature: 1.0

# === User ===

What is The Answer?
"""
        self.assertRaises(
            ValueError,
            list,
            ai_messenger.ask(
                "dummy_question",
                lambda conversation: edited_conversation,
            ),
        )

    def test_initial_conversation_is_populated_with_defaults(self):
        ai_messenger = self.create_messenger()[0]

        expected_conversation = f"""\
{self.NOTES}

# === System ===

{AiMessenger.DEFAULT_SYSTEM_PROMPT}


# === Settings ===

Model: fake/model1
Reasoning: default
Streaming: off
Temperature: {AiMessenger.DEFAULT_TEMPERATURE}



# === User ===


"""
        self.assertEqual(expected_conversation, ai_messenger.conversation_to_str())

    def test_initial_conversation_reflects_the_current_settings(self):
        ai_messenger = self.create_messenger()[0]

        ai_messenger.set_model("fake/model2")
        ai_messenger.set_reasoning(Reasoning.ON)
        ai_messenger.set_streaming(False)
        ai_messenger.set_temperature(2.0)

        expected_conversation = f"""\
{self.NOTES}

# === System ===

{AiMessenger.DEFAULT_SYSTEM_PROMPT}


# === Settings ===

Model: fake/model2
Reasoning: on
Streaming: off
Temperature: 2.0



# === User ===


"""
        self.assertEqual(expected_conversation, ai_messenger.conversation_to_str())

    def test_empty_edited_conversation_is_ignored_but_existing_conversation_is_kept(self):
        empty_conversation = "\n"
        ai_messenger, ai_client, response_chunks = self.ask(
            self.LONG_CONVERSATION,
            [
                [
                    AiResponse(is_delta=False, is_reasoning=False, text="42."),
                ],
            ],
        )
        ai_client.model = None
        ai_client.conversation = None
        ai_client.temperature = None
        ai_client.reasoning = None
        ai_client.streaming = None

        response_chunks = list(
            ai_messenger.ask(
                "dummy question",
                lambda conversation: empty_conversation
            )
        )

        expected_conversation = f"""\
{self.NOTES}

{self.LONG_CONVERSATION}


# === AI ===

42.
"""
        self.assertEqual([], response_chunks)
        self.assertEqual(expected_conversation, ai_messenger.conversation_to_str())
        self.assertIsNone(ai_client.model)
        self.assertIsNone(ai_client.conversation)
        self.assertIsNone(ai_client.temperature)
        self.assertIsNone(ai_client.reasoning)
        self.assertIsNone(ai_client.streaming)

    def test_ai_response_can_be_streamed_and_appended_to_the_conversation(self):
        ai_messenger, ai_client, response_chunks = self.ask(
            self.LONG_CONVERSATION,
            [
                [
                    AiResponse(is_delta=False, is_reasoning=False, text="42."),
                ],
            ],
        )

        expected_conversation = f"""\
{self.NOTES}

{self.LONG_CONVERSATION}


# === AI ===

42.
"""
        self.assertEqual(
            [
                "Model: fake/model2\n",
                "Reasoning: on\n",
                "Streaming: on\n",
                "Temperature: 2.0\n",
                "Waiting for fake...",
                "\n\n# === AI ===\n\n",
                "42.",
                "\n",
            ],
            response_chunks,
        )
        self.assertEqual(expected_conversation, ai_messenger.conversation_to_str())
        self.assertEqual("model2", ai_client.model)
        self.assertEqual(
            [
                Message(type=MessageType.SYSTEM, text="Custom system prompt."),
                Message(type=MessageType.USER, text="What is a question?"),
                Message(type=MessageType.AI, text="A sentence seeking an answer."),
                Message(type=MessageType.USER, text="And what is The Answer?"),
            ],
            ai_client.conversation,
        )
        self.assertEqual(2.0, ai_client.temperature)
        self.assertEqual(Reasoning.ON, ai_client.reasoning)
        self.assertEqual(True, ai_client.streaming)

    def test_when_the_edited_conversation_lacks_a_system_prompt_then_the_default_is_used(self):
        edited_conversation = "# === User ===\n\nWhat is The Answer?"

        ai_messenger, ai_client, response_chunks = self.ask(
            edited_conversation,
            [
                [
                    AiResponse(is_delta=False, is_reasoning=False, text="42."),
                ],
            ],
        )

        expected_conversation = f"""\
{self.NOTES}

# === System ===

{AiMessenger.DEFAULT_SYSTEM_PROMPT}


{edited_conversation}


# === AI ===

42.
"""
        self.assertEqual(
            [
                "Waiting for fake...",
                "\n\n# === AI ===\n\n",
                "42.",
                "\n",
            ],
            response_chunks,
        )
        self.assertEqual(expected_conversation, ai_messenger.conversation_to_str())
        self.assertEqual("model1", ai_client.model)
        self.assertEqual(AiMessenger.DEFAULT_TEMPERATURE, ai_client.temperature)
        self.assertEqual(
            [
                Message(type=MessageType.SYSTEM, text=AiMessenger.DEFAULT_SYSTEM_PROMPT),
                Message(type=MessageType.USER, text="What is The Answer?"),
            ],
            ai_client.conversation,
        )

    def test_when_the_edited_conversation_lacks_a_settings_block_then_the_current_settings_are_used(self):
        edited_conversation = """\
# === System ===

Custom system prompt.


# === User ===

What is The Answer?
"""
        ai_messenger, ai_client = self.create_messenger(
            [
                [
                    AiResponse(is_delta=False, is_reasoning=False, text="42."),
                ],
            ],
        )
        ai_messenger.set_model("fake/model2")
        ai_messenger.set_reasoning(Reasoning.OFF)
        ai_messenger.set_streaming(False)
        ai_messenger.set_temperature(2.0)
        ai_messenger.clear()

        response_chunks = list(
            ai_messenger.ask(
                "dummy question",
                lambda conversation: edited_conversation
            )
        )

        expected_conversation = f"""\
{self.NOTES}

{edited_conversation}

# === AI ===

42.
"""
        self.assertEqual(
            [
                "Waiting for fake...",
                "\n\n# === AI ===\n\n",
                "42.",
                "\n",
            ],
            response_chunks,
        )
        self.assertEqual(expected_conversation, ai_messenger.conversation_to_str())
        self.assertEqual("model2", ai_client.model)
        self.assertEqual(
            [
                Message(type=MessageType.SYSTEM, text="Custom system prompt."),
                Message(type=MessageType.USER, text="What is The Answer?"),
            ],
            ai_client.conversation,
        )
        self.assertEqual(2.0, ai_client.temperature)
        self.assertEqual(Reasoning.OFF, ai_client.reasoning)
        self.assertEqual(False, ai_client.streaming)

    def test_when_the_edited_conversation_does_not_end_with_user_prompt_then_no_ai_response_is_created_but_conversation_is_saved(self):
        edited_conversation = """\
# === System ===

Custom system prompt.


# === User ===

What is The Answer?


# === AI ===

42."""
        ai_messenger, ai_client, response_chunks = self.ask(
            edited_conversation,
            [[]],
        )

        expected_conversation = f"""\
{self.NOTES}

{edited_conversation}
"""
        self.assertEqual([], response_chunks)
        self.assertEqual(expected_conversation, ai_messenger.conversation_to_str())
        self.assertIsNone(ai_client.model)
        self.assertIsNone(ai_client.conversation)
        self.assertIsNone(ai_client.temperature)
        self.assertIsNone(ai_client.reasoning)
        self.assertIsNone(ai_client.streaming)

    def test_clearing_resets_the_system_prompt_but_keeps_settings(self):
        ai_messenger = self.ask(
            self.LONG_CONVERSATION,
            [
                [
                    AiResponse(is_delta=False, is_reasoning=False, text="42."),
                ],
            ],
        )[0]
        ai_messenger.set_reasoning(Reasoning.OFF)
        ai_messenger.set_streaming(False)

        ai_messenger.clear()

        expected_conversation = f"""\
{self.NOTES}

# === System ===

{AiMessenger.DEFAULT_SYSTEM_PROMPT}


# === Settings ===

Model: fake/model2
Reasoning: off
Streaming: off
Temperature: 2.0



# === User ===


"""
        self.assertEqual(expected_conversation, ai_messenger.conversation_to_str())

    def test_when_the_edited_conversation_lacks_a_user_prompt_when_conversation_is_reset(self):
        edited_conversation = """\
# === System ===

Custom system prompt.


# === AI ===

42.
"""
        ai_messenger, ai_client = self.create_messenger(
            [
                [
                    AiResponse(is_delta=False, is_reasoning=False, text="42."),
                ],
            ],
        )
        ai_messenger.set_model("fake/model2")
        ai_messenger.set_temperature(2.0)
        ai_messenger.set_reasoning(Reasoning.OFF)
        ai_messenger.set_streaming(False)
        ai_messenger.clear()

        response_chunks = list(
            ai_messenger.ask(
                "dummy question",
                lambda conversation: edited_conversation
            )
        )

        expected_conversation = f"""\
{self.NOTES}

# === System ===

Custom system prompt.


# === Settings ===

Model: fake/model2
Reasoning: off
Streaming: off
Temperature: 2.0



# === User ===


"""
        self.assertEqual([], response_chunks)
        self.assertEqual(expected_conversation, ai_messenger.conversation_to_str())
        self.assertIsNone(ai_client.model)
        self.assertIsNone(ai_client.conversation)
        self.assertIsNone(ai_client.reasoning)
        self.assertIsNone(ai_client.streaming)
        self.assertIsNone(ai_client.temperature)

    def test_block_headers_are_ignored_inside_fenced_code_blocks(self):
        system_prompt = """\
Custom system prompt.

```
# === What ===

Not an actual block.
```\
"""
        conversation = f"""\
# === System ===

{system_prompt}


# === Settings ===

Temperature: 2.0


# === User ===

What is The Answer?
"""
        ai_messenger, ai_client, response_chunks = self.ask(
            conversation,
            [
                [
                    AiResponse(is_delta=False, is_reasoning=False, text="42."),
                ],
            ],
        )

        expected_conversation = f"""\
{self.NOTES}

{conversation}

# === AI ===

42.
"""
        self.assertEqual(
            [
                "Temperature: 2.0\n",
                "Waiting for fake...",
                "\n\n# === AI ===\n\n",
                "42.",
                "\n",
            ],
            response_chunks,
        )
        self.assertEqual(expected_conversation, ai_messenger.conversation_to_str())
        self.assertEqual(
            [
                Message(type=MessageType.SYSTEM, text=system_prompt),
                Message(type=MessageType.USER, text="What is The Answer?"),
            ],
            ai_client.conversation,
        )
        self.assertEqual(2.0, ai_client.temperature)

    def test_ai_thoughts_and_response_can_be_streamed_and_appended_to_the_conversation(self):
        ai_messenger, ai_client, response_chunks = self.ask(
            self.LONG_CONVERSATION,
            [
                [
                    AiResponse(is_delta=True, is_reasoning=True, text="6*9"),
                    AiResponse(is_delta=True, is_reasoning=True, text=" which is 42"),
                    AiResponse(is_delta=True, is_reasoning=True, text=" in base 13."),
                    AiResponse(is_delta=True, is_reasoning=False, text="4"),
                    AiResponse(is_delta=True, is_reasoning=False, text="2"),
                    AiResponse(is_delta=True, is_reasoning=False, text="."),
                ],
            ],
        )

        expected_conversation = f"""\
{self.NOTES}

{self.LONG_CONVERSATION}


# === AI Reasoning ===

6*9 which is 42 in base 13.


# === AI ===

42.
"""
        self.assertEqual(
            [
                "Model: fake/model2\n",
                "Reasoning: on\n",
                "Streaming: on\n",
                "Temperature: 2.0\n",
                "Waiting for fake...",
                "\n\n# === AI Reasoning ===\n\n",
                "6*9",
                " which is 42",
                " in base 13.",
                "\n\n# === AI ===\n\n",
                "4",
                "2",
                ".",
                "\n",
            ],
            response_chunks,
        )
        self.assertEqual(expected_conversation, ai_messenger.conversation_to_str())
        self.assertEqual("model2", ai_client.model)
        self.assertEqual(
            [
                Message(type=MessageType.SYSTEM, text="Custom system prompt."),
                Message(type=MessageType.USER, text="What is a question?"),
                Message(type=MessageType.AI, text="A sentence seeking an answer."),
                Message(type=MessageType.USER, text="And what is The Answer?"),
            ],
            ai_client.conversation,
        )
        self.assertEqual(2.0, ai_client.temperature)
        self.assertEqual(Reasoning.ON, ai_client.reasoning)
        self.assertEqual(True, ai_client.streaming)

    def test_when_the_ai_provides_complete_thoughts_and_response_after_a_stream_then_they_are_appended_to_the_conversation(self):
        ai_messenger, ai_client, response_chunks = self.ask(
            self.LONG_CONVERSATION,
            [
                [
                    AiResponse(is_delta=True, is_reasoning=True, text="6*9"),
                    AiResponse(is_delta=True, is_reasoning=True, text=" which is 42"),
                    AiResponse(is_delta=True, is_reasoning=True, text=" in base 13."),
                    AiResponse(is_delta=False, is_reasoning=True, text="The answer is 42."),
                    AiResponse(is_delta=True, is_reasoning=False, text="6"),
                    AiResponse(is_delta=True, is_reasoning=False, text="*"),
                    AiResponse(is_delta=True, is_reasoning=False, text="9"),
                    AiResponse(is_delta=True, is_reasoning=False, text="."),
                    AiResponse(is_delta=False, is_reasoning=False, text="42."),
                ],
            ],
        )

        expected_conversation = f"""\
{self.NOTES}

{self.LONG_CONVERSATION}


# === AI Reasoning ===

The answer is 42.


# === AI ===

42.
"""
        self.assertEqual(
            [
                "Model: fake/model2\n",
                "Reasoning: on\n",
                "Streaming: on\n",
                "Temperature: 2.0\n",
                "Waiting for fake...",
                "\n\n# === AI Reasoning ===\n\n",
                "6*9",
                " which is 42",
                " in base 13.",
                "\n\n# === AI ===\n\n",
                "6",
                "*",
                "9",
                ".",
                "\n",
            ],
            response_chunks,
        )
        self.assertEqual(expected_conversation, ai_messenger.conversation_to_str())
        self.assertEqual("model2", ai_client.model)
        self.assertEqual(
            [
                Message(type=MessageType.SYSTEM, text="Custom system prompt."),
                Message(type=MessageType.USER, text="What is a question?"),
                Message(type=MessageType.AI, text="A sentence seeking an answer."),
                Message(type=MessageType.USER, text="And what is The Answer?"),
            ],
            ai_client.conversation,
        )
        self.assertEqual(2.0, ai_client.temperature)
        self.assertEqual(Reasoning.ON, ai_client.reasoning)
        self.assertEqual(True, ai_client.streaming)

    def test_changed_settings_are_appended_to_the_conversation(self):
        ai_messenger, ai_client, response_chunks = self.ask(
            self.LONG_CONVERSATION,
            [
                [
                    AiResponse(is_delta=False, is_reasoning=False, text="42."),
                ],
            ],
        )

        ai_messenger.set_model("fake/model1")
        conv_1 = ai_messenger.conversation_to_str()

        ai_messenger.set_reasoning(Reasoning.OFF)
        conv_2 = ai_messenger.conversation_to_str()

        ai_messenger.set_streaming(False)
        conv_3 = ai_messenger.conversation_to_str()

        ai_messenger.set_temperature(0.0)
        conv_4 = ai_messenger.conversation_to_str()

        expected_conversation = (
            self.NOTES
            + "\n\n"
            + self.LONG_CONVERSATION
            + "\n\n\n# === AI ===\n\n42.\n\n\n"
        )

        expected_settings_1 = """\
# === Settings ===

Model: fake/model1
Reasoning: on
Streaming: on
Temperature: 2.0

"""

        expected_settings_2 = """\
# === Settings ===

Model: fake/model1
Reasoning: off
Streaming: on
Temperature: 2.0

"""

        expected_settings_3 = """\
# === Settings ===

Model: fake/model1
Reasoning: off
Streaming: off
Temperature: 2.0

"""

        expected_settings_4 = """\
# === Settings ===

Model: fake/model1
Reasoning: off
Streaming: off
Temperature: 0.0

"""

        self.assertEqual(expected_conversation + expected_settings_1, conv_1)
        self.assertEqual(expected_conversation + expected_settings_2, conv_2)
        self.assertEqual(expected_conversation + expected_settings_3, conv_3)
        self.assertEqual(expected_conversation + expected_settings_4, conv_4)


class FakeAiClient(AiClient):
    def __init__(
            self,
            responses: collections.abc.Sequence[collections.abc.Sequence[AiResponse]],
    ):
        self.responses = responses
        self.model = None
        self.conversation = None
        self.temperature = None
        self.reasoning = None
        self.streaming = None

    def list_models(self) -> collections.abc.Sequence[str]:
        return ["model1", "model2"]

    def respond(
            self,
            model: str,
            conversation: typing.Iterator[Message],
            temperature: float,
            reasoning: Reasoning,
    ) -> typing.Iterator[AiResponse]:
        yield from self._respond(
            model,
            conversation,
            temperature,
            reasoning,
            streaming=False,
        )

    def respond_streaming(
            self,
            model: str,
            conversation: typing.Iterator[Message],
            temperature: float,
            reasoning: Reasoning,
    ) -> typing.Iterator[AiResponse]:
        yield from self._respond(
            model,
            conversation,
            temperature,
            reasoning,
            streaming=True,
        )

    def _respond(
            self,
            model: str,
            conversation: typing.Iterator[Message],
            temperature: float,
            reasoning: Reasoning,
            streaming: bool,
    ) -> typing.Iterator[AiResponse]:
        self.model = model
        self.conversation = conversation
        self.temperature = temperature
        self.reasoning = reasoning
        self.streaming = streaming

        response = self.responses.pop(0)

        yield from response


if __name__ == "__main__":
    sys.exit(main(sys.argv))
