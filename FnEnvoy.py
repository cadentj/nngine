from typing import Callable, Union, Any

from nnsight.intervention import InterventionProxy

from .envoy import Envoy

class FnEnvoy(Envoy):
    
    def __init__(
            self, 
            envoy: Envoy, 
            fn: Callable, 
            inverse: Callable = None, 
            io: str = "output",
            replace: bool = True
        ):
        super().__init__(envoy._module)

        if not inverse and replace: 
            raise ValueError("Inverse function required for replaceable FnEnvoy")

        self._envoy = envoy

        self._envoy._sub_envoys.append(self)
        self._fn = fn
        self._inverse = inverse

        self._io = io
        self._replace = replace
        self._output = None

    def __repr__(self):
        
        return "Placeholder"

    @property
    def output(self):
        if self._output is None:

            if self._io == "output":
                self._output = self._fn(self._envoy.output)
            else:
                self._output = self._fn(self._envoy.input)

        return self._output

    @output.setter
    def output(self, value: Union[InterventionProxy, Any]) -> None:
        if not self._replace:
            raise ValueError("Cannot replace a wrapper output")

        value = self._inverse(value)

        self._tracer._graph.add(
            target="swap", args=[self.output.node, value], value=True
        )

        self._output = None