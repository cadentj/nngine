import torch

from nnsight.envoy import Envoy as NNsightEnvoy

class Envoy(NNsightEnvoy) :

    def __init__(self, module: torch.nn.Module, module_path: str = "", attr_map: dict = {}, hidden = []):

        self._attr_map = attr_map
        self._hidden = hidden

        super(Envoy, self).__init__(module, module_path)

    def _add_envoy(self, module: torch.nn.Module, name: str):

        envoy = Envoy(module, module_path=f"{self._module_path}.{name}", attr_map=self._attr_map, hidden=self._hidden)

        self._sub_envoys.append(envoy)

        # If the module already has a sub-module named 'input' or 'output',
        # mount the proxy access to 'nns_input' or 'nns_output instead.
        if hasattr(Envoy, name):

            self._handle_overloaded_mount(envoy, name)

        elif name in self._attr_map:
            
            setattr(self, self._attr_map[name], envoy)

        else:

            setattr(self, name, envoy)

    def __repr__(self) -> str:
        """Wrapper method for underlying module's string representation.
        Returns:
            str: String.
        """
        if isinstance(self._module, torch.nn.ModuleList):
            return self._repr_module_list()
        extra_lines = []
        extra_repr = self._module.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split("\n")
        child_lines = []
        for attribute_name, attribute in self.__dict__.items():
            attribute_name = self._attr_map.get(attribute_name, attribute_name)
            if isinstance(attribute, Envoy) and attribute_name not in self._hidden:
                mod_str = repr(attribute)
                mod_str = torch.nn.modules.module._addindent(mod_str, 2)
                child_lines.append("(" + attribute_name + "): " + mod_str)

        lines = extra_lines + child_lines
        main_str = self._module._get_name() + "("
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += "\n  " + "\n  ".join(lines) + "\n"
        main_str += ")"
        return main_str