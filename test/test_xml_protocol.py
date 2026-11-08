import json
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'inference'))
from xml_protocol import parse_xml_calls, strict_json

TOOLS = {'write': {'type': 'object', 'properties': {'path': {'type': 'string'}, 'content': {'type': 'string'}}, 'required': ['path', 'content'], 'additionalProperties': False}}
CALL = {'name': 'write', 'arguments': {'path': 'x.txt', 'content': '中文\n</tool_call> \\ " & < >'}}
TEXT = '<tool_call>\n' + json.dumps(CALL, ensure_ascii=False) + '\n</tool_call>'

class XMLProtocolTests(unittest.TestCase):
    def test_exact_strings_and_all_chunk_boundaries(self):
        for end in range(len(TEXT)):
            calls, _ = parse_xml_calls(TEXT[:end], TOOLS, partial=True)
            self.assertEqual(calls, [])
        self.assertEqual(parse_xml_calls(TEXT, TOOLS)[0], [CALL])
        self.assertEqual(parse_xml_calls(TEXT + '\n' + TEXT, TOOLS)[0], [CALL, CALL])

    def test_observed_plural_and_nested_function(self):
        wrapped = {'type': 'function', 'function': CALL}
        text = '<tool_calls>' + json.dumps([wrapped, wrapped]) + '</tool_calls>'
        self.assertEqual(parse_xml_calls(text, TOOLS)[0], [CALL, CALL])
        string_args = {'name': 'write', 'arguments': json.dumps(CALL['arguments'])}
        self.assertEqual(parse_xml_calls('<tool_call>' + json.dumps(string_args) + '</tool_call>', TOOLS)[0], [CALL])

    def test_invalid_whole_responses_cannot_execute(self):
        invalid = [TEXT[:-1], TEXT + ' explanation', 'Example: ' + TEXT, json.dumps(CALL),
                   '<tool_call>{"name":"write","name":"write","arguments":{}}</tool_call>',
                   TEXT.replace('"write"', '"unknown"'), TEXT.replace('"x.txt"', '12'),
                   '<tool_call>{"name":"write","arguments":{"path":"x","path":"y","content":"z"}}</tool_call>',
                   '<tool_call>{"name":"write","arguments":{"path":"x","content":NaN}}</tool_call>']
        for text in invalid:
            with self.subTest(text=text), self.assertRaises(Exception): parse_xml_calls(text, TOOLS)
        with self.assertRaises(ValueError): parse_xml_calls(TEXT, TOOLS, 'none')
        with self.assertRaises(ValueError): parse_xml_calls(TEXT, TOOLS, {'function': {'name': 'read'}})
        with self.assertRaises(ValueError): parse_xml_calls(TEXT, {})
        with self.assertRaises(ValueError): parse_xml_calls(TEXT * 17, TOOLS)
        with self.assertRaises(ValueError): strict_json('{"x":1,"x":2}')

    def test_plural_frame_is_atomic(self):
        wrapped = '<tool_calls>' + json.dumps([CALL, {'name':'missing', 'arguments':{}}]) + '</tool_calls>'
        with self.assertRaises(ValueError): parse_xml_calls(wrapped, TOOLS, partial=True)

if __name__ == '__main__': unittest.main()
