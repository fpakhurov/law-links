"""Search law_aliases.json by substring (annotation helper).

Usage: python -m research.aliases_q "580-ФЗ" "такси"
"""

import json
import sys

from law_links import DEFAULT_ALIASES_PATH

data = json.loads(DEFAULT_ALIASES_PATH.read_text("utf-8"))
for query in sys.argv[1:]:
    q = query.lower()
    laws = {}
    for law_id, names in data.items():
        hits = [n for n in names if q in n.lower()]
        if hits:
            laws[law_id] = hits
    print(f"== {query}: {len(laws)} laws")
    for law_id, hits in list(laws.items())[:8]:
        print(f"  {law_id}: {min(hits, key=len)[:150]}")
