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
        fn_hook: Callable,
    ) -> None:
        super().__init__()

        self._name = name
        self._parent = parent
        self._base = base 
        self._fn_hook = fn_hook
        
    def edit(self, obj: Envoy):

        parent = fetch_sub_envoy(obj, self._parent)
        base = fetch_sub_envoy(parent, self._base)

        edit = self._fn_hook(base)

        setattr(
            parent,
            self._name,
            edit
        )

    def restore(self, obj: Envoy):
        
        delattr(
            fetch_sub_envoy(obj, self._parent),
            self._name
        )
        