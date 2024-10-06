# %%

import json
import sys

sys.path.append("..")
from core.model import load

repo_id = "Qwen/Qwen2.5-0.5B-Instruct"

pytree = load(repo_id)

name = repo_id.replace("/", "_")
with open(f"{name}.json", "w") as f:
    json.dump(pytree, f)