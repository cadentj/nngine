from typing import Callable

from .Editor import Edit
from ..util import fetch_sub_envoy
from ..envoy import Envoy

class FnEdit(Edit):

    def __init__(
        self, 
        name: str,
        base: str,
        target: str,
        fn_hook: Callable,
    ) -> None:
        super().__init__()

        self._name = name
        self._base = base
        self._target = target
        self._fn_hook = fn_hook
        
    def edit(self, obj: Envoy):

        base = fetch_sub_envoy(obj, self._base)
        target = fetch_sub_envoy(base, self._target)

        edit = self._fn_hook(base)

        setattr(
            target,
            self._name,
            edit
        )

    def restore(self, obj: Envoy):
        
        delattr(
            fetch_sub_envoy(obj, self._base),
            self._name
        )
        