from typing import Callable, Union, Any

from nnsight.intervention import InterventionProxy

from .envoy import Envoy

class FnEnvoy(Envoy):
    
    def __init__(
            self, 
            base: Envoy, 
            fn: Callable, 
            inverse: Callable = None, 
            io: str = "output",
            replace: bool = True
        ):
        super().__init__(base._module)

        self._base = base

        # I don't remember what this is for
        self._base._sub_envoys.append(self)

        self._fn = fn
        self._inverse = inverse
        self._replace = replace
        self._io = io

        self._output = None

    def __repr__(self):
        
        return "Placeholder"

    @property
    def output(self):
        if self._output is None:

            self._output = self._fn(self._base)

        if self._replace:
            if self._io == "output":
                self._base.output = self._inverse(self._output)
            
            elif self._io == "input":
                self._base.input = self._inverse(self._output)

        return self._output

    @output.setter
    def output(self, value: Union[InterventionProxy, Any]) -> None:
        value = self._inverse(value)

        self._tracer._graph.add(
            target="swap", args=[self.output.node, value], value=True
        )

        if self._replace:

            if self._io == "output":
                self._base.output = value
            
            elif self._io == "input":
                self._base.input = value

        self._output = None