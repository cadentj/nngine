# %%

from nnsight import LanguageModel

model = LanguageModel("gpt2")

# %%

shapes = {}

with model.scan("hello"):

    shapes['test'] = model.transformer.h[0].output.shape

shapes['test'][0]