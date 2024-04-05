from typing import Callable

from .Editor import Edit
from ..util import fetch_sub_envoy
from ..envoy import Envoy

class FnEdit(Edit):

    def __init__(
        self, 
        name: str,
        base: str | list[str],
        target: str | list[str],
        fn_hook: Callable,
    ) -> None:
        super().__init__()

        if isinstance(base, str):
            if not isinstance(target, str):
                raise ValueError("base is a single module but target is not")
            if len(base) != len(target):
                raise ValueError("base and target are different lengths")
            
            else:
                self._base = [base]
                self._target = [target]

        self._name = name
        self._base = base
        self._target = target
        self._fn_hook = fn_hook
        
    def edit(self, obj: Envoy):

        for base, target in zip(self._base, self._target):
            base = fetch_sub_envoy(obj, base)
            target = fetch_sub_envoy(base, target)

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
        