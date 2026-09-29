"""Model suggestions as annotation rows (tasks in "suggest" mode only).

Links found in one fragment are merged into one row with comma lists when
they form the full cartesian product of their values, as the annotator
would type them; otherwise every link gets its own row.
"""

from collections import OrderedDict
from typing import Dict, List

from law_links.extractor import Extractor


def _join(values: List[object]) -> str:
    unique = list(dict.fromkeys(v for v in values if v is not None))
    return ", ".join(str(v) for v in unique)


def suggestion_rows(extractor: Extractor, text: str) -> List[Dict[str, object]]:
    groups: "OrderedDict[tuple, list]" = OrderedDict()
    for d in extractor.extract_detailed(text):
        groups.setdefault((d.start, d.end, d.link.law_id), []).append(d.link)
    rows = []
    for (start, end, law_id), links in groups.items():
        fragment = text[start:end]
        fields = [[getattr(l, f) for l in links] for f in ("article", "point_article", "subpoint_article")]
        sizes = [max(1, len({v for v in vals if v is not None})) for vals in fields]
        if sizes[0] * sizes[1] * sizes[2] == len(links):
            rows.append({
                "fragment": fragment, "law_id": law_id, "law_name": "",
                "article": _join(fields[0]), "point": _join(fields[1]), "subpoint": _join(fields[2]),
            })
        else:
            for l in links:
                rows.append({
                    "fragment": fragment, "law_id": law_id, "law_name": "",
                    "article": l.article or "", "point": l.point_article or "", "subpoint": l.subpoint_article or "",
                })
    return rows
