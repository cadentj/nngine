import json
from typing import Dict, Any

from ray import serve
from nnsight import LanguageModel

from .core import compile, Graph
from .core.schema import NNsightRequestModel    

@serve.deployment()
class ModelDeployment:
    def __init__(self, model_name: str):
        self.model = LanguageModel(model_name, device_map="auto")
        self.tok = self.model.tokenizer

    def __call__(self, request: NNsightRequestModel):

        method = getattr(self, request.op)

        code = compile(request.graph)

        return method(code, request.graph)
    
    def code(self, code: str, graph: Graph):
        return {
            "code": code
        }

    def run(self, code: str, graph: Graph):
        loc = {}

        exec(code, None, loc)

        return self.prepare_result(loc, graph)

    def chat(self, code: str, graph: Graph):
        # Get all chat nodes and tokenize their inputs
        chat_nodes = graph.get_nodes(["chat"])

        loc = {}
        for node in chat_nodes:
            node.tokenize(self.tok)

            loc[node.id + "_content"] = node.data.tokens

        exec(code, None, loc)

        return self.prepare_result(loc, graph)
    
    def prepare_result(self, loc: Dict[str, Any], graph: Graph):
        results = {}
        output_types = ["graph", "chat"]

        for node in graph.get_nodes(output_types):
            nid = node.id
            rs = loc[nid].value

            # TODO: Set this method on the nodes themselves.
            if "chat" in nid:
                input_length = len(node.data.tokens)

                resp = self.tok.decode(rs[0][input_length:], skip_special_tokens=True)

                node.data.messages.append({"role": "assisstant", "content": resp})

                rs = node.data.messages

            results[nid] = json.dumps(rs)

        return results  

