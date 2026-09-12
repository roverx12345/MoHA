"""Terminal parsing regressions; no provider or model-specific behavior."""
import unittest

from moha.answers import (final_answer_response_format, terminal_json_answer,
                          terminal_tool_call_answer)


class TerminalAnswerTests(unittest.TestCase):
    def test_final_response_format_constrains_status_and_label(self):
        value = final_answer_response_format("ABC")
        schema = value["json_schema"]["schema"]
        self.assertEqual(value["type"], "json_schema")
        self.assertEqual(schema["required"], ["status", "answer"])
        self.assertEqual(schema["properties"]["answer"]["anyOf"][0]["enum"], ["A", "B", "C"])
        self.assertFalse(schema["additionalProperties"])

    def test_explicit_unadvertised_terminal_tool_is_recovered(self):
        for name, arguments, expected in [
            ("answer", '{"status":"answered","answer":"B"}',
             {"status": "answered", "answer": "B"}),
            ("final_answer", {"status": "abstained", "answer": None},
             {"status": "abstained", "answer": None}),
            ("final_answer", {"status": "abstained", "answer": ""},
             {"status": "abstained", "answer": None}),
        ]:
            with self.subTest(name=name, arguments=arguments):
                self.assertEqual(terminal_tool_call_answer(name, arguments, "ABC"), expected)

    def test_ambiguous_or_nonterminal_tool_is_not_recovered(self):
        for name, arguments in [
            ("observe", {"status": "answered", "answer": "A"}),
            ("answer", {"answer": "A"}),
            ("answer", {"status": "answered", "answer": "Z"}),
            ("answer", {"status": "abstained", "answer": "A"}),
            ("answer", '{"status":"answered","answer":"A","answer":"B"}'),
        ]:
            with self.subTest(name=name, arguments=arguments):
                self.assertIsNone(terminal_tool_call_answer(name, arguments, "ABC"))

    def test_explanation_followed_by_explicit_answer(self):
        text = ('The observations support the final conclusion.\n\n'
                '{"status":"answered","answer":"B"}')
        self.assertEqual(terminal_json_answer(text, "ABCDEF"),
                         {"status": "answered", "answer": "B"})

    def test_fenced_json_and_explicit_abstention(self):
        for content, expected in [
            ('Reasoning.\n```json\n{"status":"answered","answer":"E"}\n```',
             {"status": "answered", "answer": "E"}),
            ('Insufficient evidence.\n```\n{"status":"abstained","answer":null}\n```',
             {"status": "abstained", "answer": None}),
            ('The answer is A was an earlier hypothesis.\n'
             '{"status":"abstained","answer":null}',
             {"status": "abstained", "answer": None}),
        ]:
            with self.subTest(content=content):
                self.assertEqual(terminal_json_answer(content, "ABCDEF"), expected)

    def test_invalid_or_ambiguous_objects_are_not_repaired(self):
        for content in [
            '{"status":"answered","answer":"Z"}',
            '{"status":"answered","answer":null}',
            '{"status":"abstained","answer":"A"}',
            '{"status":"abstained"}',
            '{"answer":"A"}',
            '{"status":"answered","answer":"A","answer":"B"}',
            '{"status":"answered","answer":"A"',
            '{"status":"answered","answer":"A"} followed by another action',
            'The likely answer is (option A).',
            '{"status":"answered","answer":["A"]}',
            '```python\n{"status":"answered","answer":"A"}\n```',
            None,
        ]:
            with self.subTest(content=content):
                self.assertIsNone(terminal_json_answer(content, "ABCDEF"))

    def test_tool_and_reasoning_markup_is_not_an_answer(self):
        obj = '{"status":"answered","answer":"A"}'
        for content in [
            '<tool_call>' + obj + '</tool_call>',
            '<tool_call>' + obj,
            '<think>' + obj + '</think>',
            '<think>' + obj,
            obj + '\n<tool_call><function=observe></function></tool_call>',
        ]:
            with self.subTest(content=content):
                self.assertIsNone(terminal_json_answer(content, "AB"))
        self.assertEqual(terminal_json_answer('<think>analysis</think>\n' + obj, "AB"),
                         {"status": "answered", "answer": "A"})


if __name__ == "__main__":
    unittest.main()
