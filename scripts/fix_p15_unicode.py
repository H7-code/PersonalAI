import re

with open('scripts/phase15_offline_isolation_benchmark.py', 'r', encoding='utf-8') as f:
    src = f.read()

replacements = [
    ('\u2713', '[OK]'),
    ('\u2717', '[!!]'),
    ('\u26a0', '[??]'),
    ('\u2192', '->'),
]
for old, new in replacements:
    src = src.replace(old, new)

# Ensure stdout is UTF-8
header = 'import sys\nsys.stdout.reconfigure(encoding="utf-8", errors="replace")\n'
if 'reconfigure' not in src:
    src = header + src

with open('scripts/phase15_offline_isolation_benchmark.py', 'w', encoding='utf-8') as f:
    f.write(src)

print('Done - replaced unicode symbols')
