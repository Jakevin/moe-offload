#!/usr/bin/env python3
"""Unit tests for tool_calls.py (pure Python, no model, no mlx import)."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import unittest

import gen_limits
import tool_calls

try:
    import jinja2
    from jinja2.sandbox import ImmutableSandboxedEnvironment
except ImportError:
    jinja2 = None  # type: ignore[assignment]
    ImmutableSandboxedEnvironment = None  # type: ignore[assignment]

# Model dir: same QWEN38_MODEL env var as loader.py (template test is skipped if absent).
CHAT_TEMPLATE_PATH = (
    Path(os.environ.get("QWEN38_MODEL", "models/Qwen3.8-Flash-Next-GSQ-RCO-Q2-mlx"))
    / "chat_template.jinja"
)


SAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "days": {"type": "integer", "description": "Forecast days"},
                    "threshold": {"type": "number"},
                    "include_hourly": {"type": "boolean"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "meta": {"type": "object"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_query",
            "description": "Execute SQL query",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["sql"],
            },
        },
    },
]


class TestToolCalls(unittest.TestCase):
    def test_no_mlx_imported(self) -> None:
        self.assertNotIn("mlx", sys.modules)
        self.assertNotIn("mlx.core", sys.modules)

    def test_single_call(self) -> None:
        raw = (
            "<tool_call>\n"
            "<function=get_weather>\n"
            "<parameter=city>\n"
            "San Francisco\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        content, calls = tool_calls.parse_tool_calls(raw, SAMPLE_TOOLS)
        self.assertIsNone(content)
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0]["id"].startswith("call_"))
        self.assertEqual(calls[0]["type"], "function")
        self.assertEqual(calls[0]["function"]["name"], "get_weather")
        self.assertEqual(
            json.loads(calls[0]["function"]["arguments"]),
            {"city": "San Francisco"},
        )

    def test_multiple_calls(self) -> None:
        raw = (
            "<tool_call>\n"
            "<function=get_weather>\n"
            "<parameter=city>\n"
            "Tokyo\n"
            "</parameter>\n"
            "<parameter=days>\n"
            "3\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>\n"
            "<tool_call>\n"
            "<function=run_query>\n"
            "<parameter=sql>\n"
            "SELECT *\nFROM weather_logs\nWHERE city = 'Tokyo';\n"
            "</parameter>\n"
            "<parameter=limit>\n"
            "10\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        content, calls = tool_calls.parse_tool_calls(raw, SAMPLE_TOOLS)
        self.assertIsNone(content)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["function"]["name"], "get_weather")
        self.assertEqual(
            json.loads(calls[0]["function"]["arguments"]),
            {"city": "Tokyo", "days": 3},
        )
        self.assertEqual(calls[1]["function"]["name"], "run_query")
        self.assertEqual(
            json.loads(calls[1]["function"]["arguments"]),
            {
                "sql": "SELECT *\nFROM weather_logs\nWHERE city = 'Tokyo';",
                "limit": 10,
            },
        )
        self.assertNotEqual(calls[0]["id"], calls[1]["id"])

    def test_typed_params_and_unknown_params(self) -> None:
        raw = (
            "<tool_call>\n"
            "<function=get_weather>\n"
            "<parameter=city>\n"
            "94107\n"
            "</parameter>\n"
            "<parameter=days>\n"
            "5\n"
            "</parameter>\n"
            "<parameter=threshold>\n"
            "19.75\n"
            "</parameter>\n"
            "<parameter=include_hourly>\n"
            "true\n"
            "</parameter>\n"
            "<parameter=tags>\n"
            '["urgent", "coastal"]\n'
            "</parameter>\n"
            "<parameter=meta>\n"
            '{"source": "noaa", "retries": 2}\n'
            "</parameter>\n"
            "<parameter=unknown_num>\n"
            "42\n"
            "</parameter>\n"
            "<parameter=unknown_str>\n"
            "raw_value\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        content, calls = tool_calls.parse_tool_calls(raw, SAMPLE_TOOLS)
        self.assertIsNone(content)
        self.assertEqual(len(calls), 1)
        args = json.loads(calls[0]["function"]["arguments"])
        # 'city' has schema type 'string', so "94107" stays a string
        self.assertEqual(args["city"], "94107")
        self.assertIsInstance(args["city"], str)
        self.assertEqual(args["days"], 5)
        self.assertIsInstance(args["days"], int)
        self.assertAlmostEqual(args["threshold"], 19.75)
        self.assertIsInstance(args["threshold"], float)
        self.assertIs(args["include_hourly"], True)
        self.assertEqual(args["tags"], ["urgent", "coastal"])
        self.assertEqual(args["meta"], {"source": "noaa", "retries": 2})
        self.assertEqual(args["unknown_num"], 42)
        self.assertEqual(args["unknown_str"], "raw_value")

        # Fallback when integer param has non-JSON string
        raw_bad_int = (
            "<tool_call>\n"
            "<function=get_weather>\n"
            "<parameter=city>\nParis\n</parameter>\n"
            "<parameter=days>\nthree\n</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        _, calls_bad = tool_calls.parse_tool_calls(raw_bad_int, SAMPLE_TOOLS)
        args_bad = json.loads(calls_bad[0]["function"]["arguments"])
        self.assertEqual(args_bad["days"], "three")

    def test_missing_close_tags_and_whitespace_variations(self) -> None:
        # Missing </tool_call>
        raw1 = (
            "<tool_call>\n"
            "<function=get_weather>\n"
            "<parameter=city>\nBerlin\n</parameter>\n"
            "</function>"
        )
        content1, calls1 = tool_calls.parse_tool_calls(raw1, SAMPLE_TOOLS)
        self.assertIsNone(content1)
        self.assertEqual(len(calls1), 1)
        self.assertEqual(json.loads(calls1[0]["function"]["arguments"]), {"city": "Berlin"})

        # Missing both </function> and </tool_call>
        raw2 = (
            "<tool_call>\n"
            "<function=get_weather>\n"
            "<parameter=city>\nMadrid\n</parameter>"
        )
        content2, calls2 = tool_calls.parse_tool_calls(raw2, SAMPLE_TOOLS)
        self.assertIsNone(content2)
        self.assertEqual(len(calls2), 1)
        self.assertEqual(json.loads(calls2[0]["function"]["arguments"]), {"city": "Madrid"})

        # Single-line compact XML with extra spaces in tag attributes
        raw3 = "<tool_call>  <function= get_weather > <parameter= city >Rome</parameter> </function> </tool_call>"
        content3, calls3 = tool_calls.parse_tool_calls(raw3, SAMPLE_TOOLS)
        self.assertIsNone(content3)
        self.assertEqual(len(calls3), 1)
        self.assertEqual(calls3[0]["function"]["name"], "get_weather")
        self.assertEqual(json.loads(calls3[0]["function"]["arguments"]), {"city": "Rome"})

    def test_json_fallback(self) -> None:
        raw = (
            'Let me look that up.\n\n'
            '<tool_call>{"name": "get_weather", "arguments": {"city": "London", "days": 2}}</tool_call>'
        )
        content, calls = tool_calls.parse_tool_calls(raw, SAMPLE_TOOLS)
        self.assertEqual(content, "Let me look that up.")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "get_weather")
        self.assertEqual(
            json.loads(calls[0]["function"]["arguments"]),
            {"city": "London", "days": 2},
        )

    def test_text_before_call_and_think_stripping(self) -> None:
        raw = (
            "<think>\nI should call get_weather for Seoul.\n</think>\n\n"
            "I will check the forecast for Seoul right away.\n\n"
            "<tool_call>\n"
            "<function=get_weather>\n"
            "<parameter=city>\nSeoul\n</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        content, calls = tool_calls.parse_tool_calls(raw, SAMPLE_TOOLS)
        self.assertEqual(content, "I will check the forecast for Seoul right away.")
        self.assertEqual(len(calls), 1)
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"city": "Seoul"})

    def test_no_call(self) -> None:
        raw = "<think>\n\n</think>\n\nThe capital of France is Paris, and 2 < 3."
        content, calls = tool_calls.parse_tool_calls(raw, SAMPLE_TOOLS)
        self.assertEqual(content, "The capital of France is Paris, and 2 < 3.")
        self.assertEqual(calls, [])

    def test_stream_parser_chunked_at_every_byte_boundary(self) -> None:
        samples = [
            # 1. Text before single tool call + UTF-8 characters
            (
                "Checking weather in Zürich & 東京 (<10km radius)...\n\n"
                "<tool_call>\n"
                "<function=get_weather>\n"
                "<parameter=city>\nZürich\n</parameter>\n"
                "<parameter=days>\n3\n</parameter>\n"
                "</function>\n"
                "</tool_call>"
            ),
            # 2. Only tool call (with <think> block and no content text)
            (
                "<think>\nThinking about calling tool...\n</think>\n\n"
                "<tool_call>\n"
                "<function=get_weather>\n"
                "<parameter=city>\nParis\n</parameter>\n"
                "</function>\n"
                "</tool_call>"
            ),
            # 3. Multiple tool calls + missing closing tag
            (
                "Running two queries:\n"
                "<tool_call>\n<function=get_weather>\n<parameter=city>\nOslo\n</parameter>\n</function>\n</tool_call>\n"
                "<tool_call>\n<function=run_query>\n<parameter=sql>\nSELECT 1;\n</parameter>\n</function>"
            ),
            # 4. Plain text with '<' and partial '<tool_' prefix but no tool call
            "Here is a comparison: a < b and <tool_not_a_call> works fine!",
        ]

        for sample in samples:
            expected_content, expected_calls = tool_calls.parse_tool_calls(sample, SAMPLE_TOOLS)
            expected_text = expected_content or ""
            expected_fn_specs = [
                (c["function"]["name"], json.loads(c["function"]["arguments"]))
                for c in expected_calls
            ]

            raw_bytes = sample.encode("utf-8")

            # A. Feed 1 byte at a time across every byte boundary
            parser_1b = tool_calls.ToolCallStreamParser(SAMPLE_TOOLS)
            emitted_parts_1b: list[str] = []
            for b_idx in range(len(raw_bytes)):
                events = parser_1b.feed(raw_bytes[b_idx : b_idx + 1])
                for kind, piece in events:
                    self.assertEqual(kind, "text")
                    emitted_parts_1b.append(piece)
            rem_1b, calls_1b = parser_1b.finish()
            if rem_1b:
                emitted_parts_1b.append(rem_1b)
            self.assertEqual("".join(emitted_parts_1b), expected_text)
            self.assertEqual(
                [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in calls_1b],
                expected_fn_specs,
            )

            # B. Split into 2 chunks at every possible byte boundary (0..len(raw_bytes))
            for split_at in range(len(raw_bytes) + 1):
                parser_split = tool_calls.ToolCallStreamParser(SAMPLE_TOOLS)
                parts: list[str] = []
                for chunk in (raw_bytes[:split_at], raw_bytes[split_at:]):
                    for kind, piece in parser_split.feed(chunk):
                        self.assertEqual(kind, "text")
                        parts.append(piece)
                rem, calls = parser_split.finish()
                if rem:
                    parts.append(rem)
                self.assertEqual("".join(parts), expected_text)
                self.assertEqual(
                    [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in calls],
                    expected_fn_specs,
                )

            # C. Feed 1 character (str) at a time
            parser_1c = tool_calls.ToolCallStreamParser(SAMPLE_TOOLS)
            parts_1c: list[str] = []
            for ch in sample:
                for kind, piece in parser_1c.feed(ch):
                    self.assertEqual(kind, "text")
                    parts_1c.append(piece)
            rem_1c, calls_1c = parser_1c.finish()
            if rem_1c:
                parts_1c.append(rem_1c)
            self.assertEqual("".join(parts_1c), expected_text)
            self.assertEqual(
                [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in calls_1c],
                expected_fn_specs,
            )

    def test_normalize_tools_and_tool_choice(self) -> None:
        mixed_tools = [
            {"type": "web_search"},
            {
                "type": "function",
                "name": "flat_fn",
                "description": "A flat Responses API tool",
                "parameters": {"type": "object", "properties": {"x": {"type": "integer"}}},
            },
            SAMPLE_TOOLS[0],
        ]
        norm = tool_calls.normalize_tools(mixed_tools)
        self.assertEqual(len(norm), 2)
        self.assertEqual(norm[0]["function"]["name"], "flat_fn")
        self.assertEqual(norm[1]["function"]["name"], "get_weather")

        base_msgs = [{"role": "user", "content": "Hi"}]

        # auto
        eff_auto, msgs_auto = tool_calls.apply_tool_choice(norm, "auto", base_msgs)
        self.assertEqual(len(eff_auto), 2)
        self.assertEqual(msgs_auto, base_msgs)

        # none
        eff_none, msgs_none = tool_calls.apply_tool_choice(norm, "none", base_msgs)
        self.assertEqual(eff_none, [])
        self.assertEqual(msgs_none, base_msgs)

        # required
        eff_req, msgs_req = tool_calls.apply_tool_choice(norm, "required", base_msgs)
        self.assertEqual(len(eff_req), 2)
        self.assertEqual(msgs_req[0]["role"], "system")
        self.assertIn("MUST call", msgs_req[0]["content"])

        # specific function (Chat Completions form)
        eff_fn1, msgs_fn1 = tool_calls.apply_tool_choice(
            norm,
            {"type": "function", "function": {"name": "flat_fn"}},
            base_msgs,
        )
        self.assertEqual(len(eff_fn1), 1)
        self.assertEqual(eff_fn1[0]["function"]["name"], "flat_fn")
        self.assertEqual(msgs_fn1[0]["role"], "system")
        self.assertIn("flat_fn", msgs_fn1[0]["content"])

        # specific function (Responses flat form)
        eff_fn2, msgs_fn2 = tool_calls.apply_tool_choice(
            norm,
            {"type": "function", "name": "get_weather"},
            [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Weather?"}],
        )
        self.assertEqual(len(eff_fn2), 1)
        self.assertEqual(eff_fn2[0]["function"]["name"], "get_weather")
        self.assertEqual(msgs_fn2[0]["role"], "system")
        self.assertIn("Be brief.", msgs_fn2[0]["content"])
        self.assertIn("get_weather", msgs_fn2[0]["content"])

        # required with only non-function tools -> raises ValueError
        with self.assertRaises(ValueError):
            tool_calls.apply_tool_choice([{"type": "web_search"}], "required", base_msgs)

        # unknown function name in tool_choice -> raises ValueError
        with self.assertRaises(ValueError):
            tool_calls.apply_tool_choice(
                norm,
                {"type": "function", "name": "nonexistent"},
                base_msgs,
            )

    def test_chat_message_conversion(self) -> None:
        raw_messages = [
            {"role": "developer", "content": "System rule 1."},
            {"role": "user", "content": [{"type": "text", "text": "What is the weather?"}]},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_abc",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Berlin", "days": 2}',
                        },
                    },
                    {
                        "id": "call_bad",
                        "type": "function",
                        "function": {
                            "name": "run_query",
                            "arguments": "not valid json",
                        },
                    },
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_abc",
                "content": '{"temp_c": 18}',
            },
        ]
        converted = tool_calls.convert_chat_messages(raw_messages)
        self.assertEqual(converted[0], {"role": "system", "content": "System rule 1."})
        self.assertEqual(converted[1], {"role": "user", "content": "What is the weather?"})
        self.assertEqual(converted[2]["role"], "assistant")
        self.assertEqual(converted[2]["content"], "")
        tcs = converted[2]["tool_calls"]
        self.assertEqual(tcs[0]["function"]["arguments"], {"city": "Berlin", "days": 2})
        self.assertEqual(tcs[1]["function"]["arguments"], {"raw": "not valid json"})
        self.assertEqual(converted[3]["role"], "tool")
        self.assertEqual(converted[3]["content"], '{"temp_c": 18}')

    def test_responses_input_conversion(self) -> None:
        raw_input = [
            {"type": "message", "role": "developer", "content": "Use tools when needed."},
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Compare Paris and London weather."}],
            },
            {"type": "reasoning", "id": "rs_1", "summary": []},
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Checking both cities."}],
            },
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "get_weather",
                "arguments": '{"city": "Paris"}',
            },
            {
                "type": "function_call",
                "call_id": "call_2",
                "name": "get_weather",
                "arguments": '{"city": "London"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "Paris: 20C Sunny",
            },
            {
                "type": "function_call_output",
                "call_id": "call_2",
                "output": [{"type": "output_text", "text": "London: 15C Rain"}],
            },
        ]
        msgs = tool_calls.convert_responses_input(raw_input, instructions="Top-level instruction.")
        # System + developer consolidated at index 0
        self.assertEqual(msgs[0]["role"], "system")
        self.assertIn("Top-level instruction.", msgs[0]["content"])
        self.assertIn("Use tools when needed.", msgs[0]["content"])
        # User message at index 1
        self.assertEqual(msgs[1], {"role": "user", "content": "Compare Paris and London weather."})
        # Assistant message + 2 consecutive function_calls merged into one assistant message at index 2
        self.assertEqual(msgs[2]["role"], "assistant")
        self.assertEqual(msgs[2]["content"], "Checking both cities.")
        self.assertEqual(len(msgs[2]["tool_calls"]), 2)
        self.assertEqual(msgs[2]["tool_calls"][0]["function"]["arguments"], {"city": "Paris"})
        self.assertEqual(msgs[2]["tool_calls"][1]["function"]["arguments"], {"city": "London"})
        # Two tool messages at indices 3 and 4
        self.assertEqual(msgs[3]["role"], "tool")
        self.assertEqual(msgs[3]["content"], "Paris: 20C Sunny")
        self.assertEqual(msgs[4]["role"], "tool")
        self.assertEqual(msgs[4]["content"], "London: 15C Rain")

    @unittest.skipIf(
        jinja2 is None or not CHAT_TEMPLATE_PATH.exists(),
        "jinja2 or model chat_template.jinja not available",
    )
    def test_jinja_template_rendering_round_trip(self) -> None:
        env = ImmutableSandboxedEnvironment(
            trim_blocks=True,
            lstrip_blocks=True,
            extensions=["jinja2.ext.loopcontrols"],
        )
        template_str = CHAT_TEMPLATE_PATH.read_text(encoding="utf-8")
        template = env.from_string(template_str)

        tools = tool_calls.normalize_tools(SAMPLE_TOOLS)
        chat_msgs = tool_calls.convert_chat_messages(
            [
                {"role": "system", "content": "You are a weather bot."},
                {"role": "user", "content": "Weather in Paris?"},
                {
                    "role": "assistant",
                    "content": "Let me check.",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city": "Paris", "days": 1}',
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": '{"temp_c": 21}'},
            ]
        )

        rendered = template.render(
            messages=chat_msgs,
            tools=tools,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        self.assertIn("<tools>", rendered)
        self.assertIn('"name": "get_weather"', rendered)
        self.assertIn("You are a weather bot.", rendered)
        self.assertIn(
            "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n"
            "<parameter=days>\n1\n</parameter>\n</function>\n</tool_call>",
            rendered,
        )
        self.assertIn("<|im_start|>user\n<tool_response>\n{\"temp_c\": 21}\n</tool_response><|im_end|>", rendered)
        self.assertTrue(rendered.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"))



CHAT_MAX_KEYS = ("max_tokens", "max_completion_tokens")
RESPONSES_MAX_KEYS = ("max_output_tokens", "max_tokens", "max_completion_tokens")


class TestGenLimits(unittest.TestCase):
    """Default output limit: unlimited (-1), capped at the remaining context."""

    def test_parse_limit(self) -> None:
        self.assertIsNone(gen_limits.parse_limit(-1))
        self.assertIsNone(gen_limits.parse_limit("-1"))
        self.assertEqual(gen_limits.parse_limit(0), 0)
        self.assertEqual(gen_limits.parse_limit("64"), 64)
        for bad in (-2, "abc", None, True, 1.5j):
            with self.assertRaises(ValueError):
                gen_limits.parse_limit(bad)

    def test_max_tokens_arg(self) -> None:
        import argparse

        self.assertEqual(gen_limits.max_tokens_arg("-1"), -1)
        self.assertEqual(gen_limits.max_tokens_arg("4096"), 4096)
        for bad in ("-5", "x"):
            with self.assertRaises(argparse.ArgumentTypeError):
                gen_limits.max_tokens_arg(bad)

    def test_requested_limit_default_unlimited(self) -> None:
        self.assertIsNone(gen_limits.requested_limit({}, CHAT_MAX_KEYS, None))
        self.assertIsNone(gen_limits.requested_limit({"max_tokens": None}, CHAT_MAX_KEYS, None))
        self.assertEqual(gen_limits.requested_limit({}, CHAT_MAX_KEYS, 4096), 4096)

    def test_request_overrides_default(self) -> None:
        self.assertEqual(gen_limits.requested_limit({"max_tokens": 64}, CHAT_MAX_KEYS, None), 64)
        self.assertEqual(gen_limits.requested_limit({"max_tokens": 64}, CHAT_MAX_KEYS, 4096), 64)
        self.assertEqual(gen_limits.requested_limit({"max_tokens": 0}, CHAT_MAX_KEYS, 4096), 0)
        self.assertIsNone(gen_limits.requested_limit({"max_tokens": -1}, CHAT_MAX_KEYS, 4096))
        with self.assertRaises(ValueError):
            gen_limits.requested_limit({"max_tokens": -3}, CHAT_MAX_KEYS, None)

    def test_field_order(self) -> None:
        both = {"max_tokens": 10, "max_completion_tokens": 20, "max_output_tokens": 30}
        self.assertEqual(gen_limits.requested_limit(both, CHAT_MAX_KEYS, None), 10)
        self.assertEqual(gen_limits.requested_limit(both, RESPONSES_MAX_KEYS, None), 30)
        self.assertEqual(
            gen_limits.requested_limit({"max_tokens": None, "max_completion_tokens": 20}, CHAT_MAX_KEYS, None),
            20,
        )
        self.assertEqual(
            gen_limits.requested_limit({"max_tokens": 10, "max_completion_tokens": 20}, RESPONSES_MAX_KEYS, None),
            10,
        )

    def test_effective_caps_at_remaining_context(self) -> None:
        self.assertEqual(gen_limits.effective_max_tokens(None, 1000, 262144), 261144)
        self.assertEqual(gen_limits.effective_max_tokens(64, 1000, 262144), 64)
        self.assertEqual(gen_limits.effective_max_tokens(500, 900, 1000), 100)
        self.assertEqual(gen_limits.effective_max_tokens(0, 10, 1000), 0)
        with self.assertRaises(ValueError):
            gen_limits.effective_max_tokens(None, 1000, 1000)
        with self.assertRaises(ValueError):
            gen_limits.effective_max_tokens(64, 2000, 1000)

    def test_context_length_from_config(self) -> None:
        cfg = {"text_config": {"max_position_embeddings": 262144}}
        self.assertEqual(gen_limits.context_length_from_config(cfg), 262144)
        self.assertEqual(gen_limits.context_length_from_config({"max_position_embeddings": 8192}), 8192)
        self.assertEqual(gen_limits.context_length_from_config({}, 131072), 131072)
        # HF tokenizers use a huge sentinel when no limit is set
        self.assertEqual(
            gen_limits.context_length_from_config(None, int(1e30)), gen_limits.FALLBACK_CONTEXT_LENGTH
        )

    def test_server_default_is_unlimited(self) -> None:
        src = (Path(__file__).resolve().parent / "server.py").read_text(encoding="utf-8")
        self.assertIn("DEFAULT_MAX_TOKENS = gen_limits.UNLIMITED", src)
        self.assertIn("type=gen_limits.max_tokens_arg", src)
        self.assertIn('("max_tokens", "max_completion_tokens")', src)
        self.assertIn('("max_output_tokens", "max_tokens", "max_completion_tokens")', src)
        self.assertEqual(src.count("gen_limits.effective_max_tokens("), 2)


if __name__ == "__main__":
    unittest.main()
