"""A0' projection: generic, non-cherry-picked bounding of FHIR tool output before it enters context.

Two operations, both honest (no field is kept because the gold needs it):
  1. strip universally-bulky, answer-irrelevant keys (narrative, meta, extensions, contained);
  2. recency-cap each resource type to the N most recent, with an explicit truncation marker.

This isolates "keep the blob out of the 32k window" from typed tools / SQL / multi-turn. See
plans/bonfire-a0prime-spec.md.
"""

BULKY = {"text", "meta", "extension", "modifierExtension", "contained"}
DATE_KEYS = ("effectiveDateTime", "recordedDate", "authoredOn", "performedDateTime",
             "issued", "onsetDateTime", "date", "started", "period")


def _strip(x):
    if isinstance(x, dict):
        return {k: _strip(v) for k, v in x.items() if k not in BULKY}
    if isinstance(x, list):
        return [_strip(v) for v in x]
    return x


def _date_key(r):
    """Best-effort recency key; resources missing a date sort oldest (empty string)."""
    if not isinstance(r, dict):
        return ""
    for k in DATE_KEYS:
        v = r.get(k)
        if isinstance(v, str):
            return v
        if isinstance(v, dict):
            if isinstance(v.get("start"), str):
                return v["start"]
    return ""


def project(tool_output, max_per_type=200):
    """tool_output is {resource_type: [resources]} (or {"error": ...}). Return a stripped, recency-capped view."""
    if not isinstance(tool_output, dict):
        return tool_output
    out = {}
    for rtype, items in tool_output.items():
        if not isinstance(items, list):
            out[rtype] = _strip(items)  # passes {"error": ...} or scalars through, stripped
            continue
        ranked = sorted(items, key=_date_key, reverse=True)
        kept = [_strip(r) for r in ranked[:max_per_type]]
        out[rtype] = {
            "resources": kept,
            "_total": len(items),
            "_returned": len(kept),
            "_truncated": len(items) > max_per_type,
        }
    return out
