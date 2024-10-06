# %%

import json
import sys

sys.path.append("..")
from services.src.model import load

repo_id = "EleutherAI/gpt-j-6b"

pytree = load(repo_id)

name = repo_id.replace("/", "_")
with open(f"{name}.json", "w") as f:
    json.dump(pytree, f)