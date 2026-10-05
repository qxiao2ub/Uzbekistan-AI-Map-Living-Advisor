"""Static regression check for the historical startup NameError."""
from __future__ import annotations

import ast
from pathlib import Path

APP = Path(__file__).with_name("app.py")
SOURCE = APP.read_text(encoding="utf-8")
TREE = ast.parse(SOURCE, filename=str(APP))

translation_node = None
for node in TREE.body:
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "TRANSLATIONS":
                translation_node = node.value
                break

assert isinstance(translation_node, ast.Dict), "TRANSLATIONS must be a dictionary"

for key_node, value_node in zip(translation_node.keys, translation_node.values):
    if isinstance(key_node, ast.Constant) and key_node.value == "en":
        for child in ast.walk(value_node):
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name) and child.func.id == "t":
                raise AssertionError("Do not call t() while constructing TRANSLATIONS['en'].")
        break
else:
    raise AssertionError("TRANSLATIONS['en'] was not found")

compile(SOURCE, str(APP), "exec")
print("PASS: no module-level t() calls in TRANSLATIONS['en']; app.py compiles successfully.")
