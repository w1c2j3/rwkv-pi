"""Out-of-tree irwkv plugin: RWKV XML -> native OpenAI tool_calls.

Load with --tool-parser-plugin /absolute/path/rwkv_xml_parser.py
          --tool-call-parser rwkv_xml. Keep xml_protocol.py beside this file.
"""
import importlib.util
import json
from pathlib import Path

from irwkv.entrypoints.chat_utils import make_tool_call_id
from irwkv.entrypoints.generate.base.protocol import (
    DeltaFunctionCall, DeltaMessage, DeltaToolCall,
    ExtractedToolCallInformation, FunctionCall, ToolCall,
)
from irwkv.tool_parsers.abstract_tool_parser import ToolParser, ToolParserManager

spec = importlib.util.spec_from_file_location("rwkv_pi_xml_protocol", Path(__file__).with_name("xml_protocol.py"))
protocol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(protocol)


class RwkvXmlToolParser(ToolParser):
    # XML is parsed directly; do not activate the old Markdown grammar.
    supports_required_and_named = False
    structural_tag_model = None

    def __init__(self, tokenizer, tools=None):
        super().__init__(tokenizer, tools)
        self._sent_content = 0

    def _calls(self, text, request, partial=False):
        schemas = {tool.function.name: tool.function.parameters for tool in self.tools}
        choice = request.tool_choice
        if hasattr(choice, "model_dump"):
            choice = choice.model_dump()
        return protocol.parse_xml_calls(text, schemas, choice, partial=partial)[0]

    def extract_tool_calls(self, model_output, request):
        start = protocol.call_start(model_output)
        if start is None:
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)
        calls = self._calls(model_output[start:], request)
        return ExtractedToolCallInformation(tools_called=True, content=model_output[:start] or None, tool_calls=[
            ToolCall(type="function", function=FunctionCall(name=call["name"],
                arguments=json.dumps(call["arguments"], ensure_ascii=False))) for call in calls])

    def extract_tool_calls_streaming(self, previous_text, current_text, delta_text,
                                     previous_token_ids, current_token_ids, delta_token_ids, request):
        start = protocol.call_start(current_text, partial=True)
        content_end = len(current_text) if start is None else start
        content = current_text[self._sent_content:content_end]
        self._sent_content = content_end
        if start is None:
            return DeltaMessage(content=content) if content else None
        calls = self._calls(current_text[start:], request, partial=True)
        deltas = []
        for index in range(len(self.prev_tool_call_arr), len(calls)):
            call = calls[index]
            arguments = json.dumps(call["arguments"], ensure_ascii=False)
            self.prev_tool_call_arr.append({"name": call["name"], "arguments": arguments})
            self.streamed_args_for_tool.append(arguments)
            deltas.append(DeltaToolCall(index=index, type="function", id=make_tool_call_id(),
                function=DeltaFunctionCall(name=call["name"], arguments=arguments)))
        return DeltaMessage(content=content or None, tool_calls=deltas) if content or deltas else None


ToolParserManager.register_module("rwkv_xml", module=RwkvXmlToolParser)
