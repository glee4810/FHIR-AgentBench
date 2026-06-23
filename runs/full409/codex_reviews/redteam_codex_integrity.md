No material defects.

Numbers reproduced:
- Matched-budget: `n=140`, code `95/140=67.9%`, resource `100/140=71.4%`, delta `-3.57pp`, McNemar `p=0.179688`.
- Resource-real: `n=147`, code `95/147=64.6%`, resource `104/147=70.7%`, delta `-6.12pp`, `p=0.022461`; code error `52/147=35.4%`.
- Pooled: code `267/409=65.3%`, resource `104/409=25.4%`, delta `+39.85pp`. Decomposition: overflow rows `+172`, non-overflow rows `-9`, net `+163`, so the positive pooled result is entirely from resource-overflow rows.
- Overflow count: `262/409`.
- Numeric judge leaderboard: GPT-5-mini `68/111=61.3%`; Claude `109/111=98.2%`; Codex is effectively `99.1%` but `human_review.json` has one missing Codex numeric label, so literal non-null is `109/110=99.1%`, while counting the missing-but-final-correct row gives `110/111=99.1%`.
- Boolean bug fixed in current artifacts: before fix, boolean deterministic labels disagreed with final on `102/228` arm slots; after fix, real boolean answers are no longer deterministically auto-graded, and boolean deterministic labels are only `62` failure labels with `0` disagreements.

Only caveat: `resource-real` significance depends on adding the 7 `other` rows to the 140 matched-budget rows; those 7 contribute `-4` net to code and move p from `0.179688` to `0.022461`. That is a sensitivity caveat, not a falsified claim.