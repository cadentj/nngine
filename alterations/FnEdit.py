from typing import Callable

from .Editor import Edit
from ..util import fetch_sub_envoy
from ..envoy import Envoy

class FnEdit(Edit):

    def __init__(
        self, 
        name: str,
        parent: str,
        base: str, 
        target: str, 
        fn_hook: Callable,
    ) -> None:
        super().__init__()

        self._name = name
        self._parent = parent
        self._base = base
        self._target = target 
        self._fn_hook = fn_hook
        
    def edit(self, obj: Envoy):

        fn_envoy = self._fn_hook(
            fetch_sub_envoy(obj, self._base),
            fetch_sub_envoy(obj, self._target)
        )

        setattr(
            fetch_sub_envoy(obj, self._parent),
            self._name,
            fn_envoy
        )

    def restore(self, obj: Envoy):
        
        delattr(
            fetch_sub_envoy(obj, self._parent),
            self._name
        )
        