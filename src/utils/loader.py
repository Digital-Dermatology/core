import os
import re

import yaml


class Loader(yaml.SafeLoader):
    def __init__(self, stream):
        self._root = os.path.split(stream.name)[0]
        super(Loader, self).__init__(stream)

    def include(self, node):
        filename = os.path.join(self._root, self.construct_scalar(node))
        with open(filename, "r") as f:
            return yaml.load(f, Loader)

    def construct_scalar(self, node):
        value = super().construct_scalar(node)
        # Handle environment variable substitution with syntax ${VAR:default}
        if isinstance(value, str) and value.startswith("${") and "}" in value:
            # Simple parsing approach for ${VAR:default} syntax
            if value.endswith("}"):
                # Extract content between ${ and }
                content = value[2:-1]  # Remove ${ and }
                if ":" in content:
                    var_name, default_value = content.split(":", 1)
                else:
                    var_name = content
                    default_value = ""

                value = os.environ.get(var_name, default_value)
        return value


Loader.add_constructor("!include", Loader.include)


def merge_dicts(a, b):
    """Recursively merge dict b into dict a."""
    for key, value in b.items():
        if key in a and isinstance(a[key], dict) and isinstance(value, dict):
            merge_dicts(a[key], value)
        else:
            a[key] = value
    return a
