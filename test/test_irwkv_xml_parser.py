"""Run in the pinned irwkv environment, not a mock of its parser API."""
import json
import os
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'inference'))
from rwkv_xml_parser import RwkvXmlToolParser
from irwkv.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from irwkv.tool_parsers.abstract_tool_parser import ToolParserManager

class NativeParserTests(unittest.TestCase):
    def setUp(self):
        self.request = ChatCompletionRequest(model='probe', messages=[{'role':'user', 'content':'Read.'}], tools=[
            {'type':'function', 'function': {'name':'read', 'parameters': {'type':'object', 'properties':{'path':{'type':'string'}}, 'required':['path']}}}], tool_choice='auto')
        self.text = '<tool_call>{"name":"read","arguments":{"path":"中文</tool_call>.txt"}}</tool_call>'
    def test_registry_and_nonstream(self):
        self.assertIs(ToolParserManager.get_tool_parser('rwkv_xml'), RwkvXmlToolParser)
        parsed = RwkvXmlToolParser(None, self.request.tools).extract_tool_calls(self.text, self.request)
        self.assertTrue(parsed.tools_called)
        self.assertEqual(json.loads(parsed.tool_calls[0].function.arguments), {'path':'中文</tool_call>.txt'})
    def test_streaming_single_character_chunks_and_no_reemission(self):
        parser = RwkvXmlToolParser(None, self.request.tools)
        deltas = []
        for index in range(1,len(self.text)+1):
            out = parser.extract_tool_calls_streaming(self.text[:index-1], self.text[:index], self.text[index-1:index], [], [], [], self.request)
            if out: deltas.extend(out.tool_calls or [])
        self.assertEqual(len(deltas),1)
        self.assertTrue(deltas[0].id)
        self.assertEqual(deltas[0].function.name,'read')
        self.assertEqual(parser.get_remaining_unstreamed_args(),'')
        self.assertIsNone(parser.extract_tool_calls_streaming(self.text,self.text,'',[],[],[],self.request))
    def test_plain_text_and_invalid_calls(self):
        parser = RwkvXmlToolParser(None, self.request.tools)
        self.assertFalse(parser.extract_tool_calls('Example: '+self.text,self.request).tools_called)
        self.request.tool_choice='none'
        with self.assertRaises(ValueError): parser.extract_tool_calls(self.text,self.request)

    def test_native_content_plus_calls_and_fenced_examples(self):
        text = "I'll read the file.\n" + self.text
        parser = RwkvXmlToolParser(None, self.request.tools)
        result = parser.extract_tool_calls(text, self.request)
        self.assertEqual(result.content,"I'll read the file.\n")
        self.assertTrue(result.tools_called)
        emitted, calls = '', []
        for index in range(1,len(text)+1):
            delta=parser.extract_tool_calls_streaming(text[:index-1],text[:index],text[index-1:index],[],[],[],self.request)
            if delta:
                emitted += delta.content or ''
                calls.extend(delta.tool_calls or [])
        self.assertEqual(emitted,result.content)
        self.assertEqual(len(calls),1)
        for fence in ('```xml','~~~xml'):
            example='Example:\n'+fence+'\n'+self.text+'\n'+fence[:3]
            self.assertFalse(RwkvXmlToolParser(None,self.request.tools).extract_tool_calls(example,self.request).tools_called)

    def test_deployed_training_template_roundtrips_native_arguments(self):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(os.environ['RWKV_XML_ARTIFACT'], local_files_only=True)
        args = {'path': '中文\n</tool_call>.txt'}
        rendered = tokenizer.apply_chat_template([
            {'role':'user','content':'Read.'},
            {'role':'assistant','content':'<think></think>', 'tool_calls':[
                {'type':'function','function':{'name':'read','arguments':args}}]},
            {'role':'tool','tool_call_id':'probe','content':'exact\nobservation\n'},
        ], tools=[t.model_dump() for t in self.request.tools], tokenize=False,
            add_generation_prompt=False, thinking=True, bos_token='')
        call_text = rendered.split('Assistant: <think></think>',1)[1].split('\n\nUser:',1)[0]
        result = RwkvXmlToolParser(tokenizer,self.request.tools).extract_tool_calls(call_text,self.request)
        self.assertEqual(json.loads(result.tool_calls[0].function.arguments),args)
        self.assertIn('<tool_response>\nexact\nobservation\n\n</tool_response>',rendered)

if __name__ == '__main__': unittest.main()
