from typing import Callable, Union, Any

from nnsight.intervention import InterventionProxy

from .envoy import Envoy

class FnEnvoy(Envoy):
    
    def __init__(
            self, 
            base: Envoy, 
            fn: Callable, 
            inverse: Callable = None,
            replace: bool = True
        ):
        super().__init__(base._module)

        self._base = base

        # I don't remember what this is for
        # self._base._sub_envoys.append(self)

        self._fn = fn
        self._inverse = inverse
        self._replace = replace

        self._output = None
        self._input = None

    def __repr__(self):
        
        return "Placeholder"

    @property
    def input(self):
        
        if self._replace:
            self.output

        if self._input is None:
            self._input = self._base.output
        
        return self._input

    @property
    def output(self):
        if self._output is None:

            self._output = self._fn(self._base)

        if self._replace:
            self._inverse(self._base, self._output)

        return self._output

    @output.setter
    def output(self, value: Union[InterventionProxy, Any]) -> None:
        if self._replace:
            self._inverse(self._base, value)

        self._output = None