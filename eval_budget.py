"""Hard $-budget ledger for the ablation eval.

The eval cost is dominated by frontier-model output tokens across (surfaces x tools x models x
seeds x questions). This ledger makes cost a FIRST-CLASS, CAPPED quantity instead of a hope:

  - `BudgetLedger(cap_usd)` accumulates real spend from each LLM response and raises
    `BudgetExceeded` the moment a charge would cross the cap -> the runner stops cleanly with
    partial-but-honest results rather than silently blowing past $100.
  - `project(...)` prints an up-front cost estimate from a tiny pilot BEFORE the big run, so a
    run is a decision, not a surprise (Codex: "cost/runtime hand-waved").

Cost source of truth: litellm.completion_cost() (uses litellm's maintained pricing DB). If a
model id isn't in that DB yet (newest models lag), we fall back to a RATES table you can override
via env (EVAL_RATES='model:in_per_mtok/out_per_mtok,...'). Always prefer measured over assumed.
"""
import os
import threading


class BudgetExceeded(Exception):
    pass


# Fallback $ / 1M tokens (input, output). Override/extend via EVAL_RATES env.
# Keep conservative (slightly high) so the cap binds early rather than late.
_DEFAULT_RATES = {
    "claude-opus-4-8": (5.0, 25.0),
    "gpt-5": (1.25, 10.0),
    "gpt-5.1": (1.25, 10.0),
    "gpt-5-mini": (0.25, 2.0),
    "_default": (3.0, 15.0),
}


def _load_rates():
    rates = dict(_DEFAULT_RATES)
    raw = os.environ.get("EVAL_RATES", "").strip()
    for part in [p for p in raw.split(",") if p.strip()]:
        try:
            model, nums = part.split(":", 1)
            i, o = nums.split("/", 1)
            rates[model.strip()] = (float(i), float(o))
        except Exception:
            continue
    return rates


def _rate_for(model, rates):
    if model in rates:
        return rates[model]
    for k, v in rates.items():
        if k != "_default" and k in model:
            return v
    return rates["_default"]


def cost_of(model, prompt_tokens, completion_tokens, rates=None):
    """Best-effort USD cost. Tries litellm's pricing DB first, then the rate table."""
    try:
        import litellm
        c = litellm.cost_per_token(model=model, prompt_tokens=int(prompt_tokens),
                                   completion_tokens=int(completion_tokens))
        if isinstance(c, (tuple, list)):
            total = float(c[0]) + float(c[1])
            if total > 0:
                return total
    except Exception:
        pass
    rates = rates or _load_rates()
    i, o = _rate_for(model, rates)
    return (prompt_tokens / 1e6) * i + (completion_tokens / 1e6) * o


class BudgetLedger:
    """Thread-safe running $ ledger with a hard cap. Charge per LLM call; stops the run at the cap."""

    def __init__(self, cap_usd, rates=None):
        self.cap = float(cap_usd)
        self.rates = rates or _load_rates()
        self._spent = 0.0
        self._by_model = {}
        self._calls = 0
        self._lock = threading.Lock()

    def charge(self, model, prompt_tokens=0, completion_tokens=0):
        """Add the cost of one call. Raises BudgetExceeded if it crosses the cap (after recording,
        so totals stay accurate and the runner can report exactly where it stopped)."""
        c = cost_of(model, prompt_tokens or 0, completion_tokens or 0, self.rates)
        with self._lock:
            self._spent += c
            self._calls += 1
            self._by_model[model] = self._by_model.get(model, 0.0) + c
            spent = self._spent
        if spent > self.cap:
            raise BudgetExceeded(f"${spent:.2f} > cap ${self.cap:.2f} after {self._calls} calls")
        return c

    def charge_usage(self, model, usage):
        """Charge from a litellm usage dict/obj ({prompt_tokens, completion_tokens})."""
        if usage is None:
            return 0.0
        get = (lambda k: usage.get(k, 0)) if isinstance(usage, dict) else (lambda k: getattr(usage, k, 0))
        return self.charge(model, get("prompt_tokens"), get("completion_tokens"))

    @property
    def spent(self):
        return self._spent

    @property
    def remaining(self):
        return max(0.0, self.cap - self._spent)

    def would_exceed(self, est_usd):
        return (self._spent + est_usd) > self.cap

    def report(self):
        with self._lock:
            return {"cap": round(self.cap, 2), "spent": round(self._spent, 4),
                    "remaining": round(self.remaining, 4), "calls": self._calls,
                    "by_model": {m: round(v, 4) for m, v in self._by_model.items()}}


def project(cost_per_question, n_questions, n_cells, label="run"):
    """Print an up-front cost projection for the matrix. cost_per_question from a pilot."""
    total = cost_per_question * n_questions * n_cells
    print(f"[budget] PROJECTION {label}: ${cost_per_question:.4f}/q x {n_questions} q x {n_cells} "
          f"cells = ${total:.2f}", flush=True)
    return total


if __name__ == "__main__":
    # quick self-check
    L = BudgetLedger(0.10)
    try:
        for i in range(100):
            L.charge("claude-opus-4-8", 2000, 800)
    except BudgetExceeded as e:
        print("stopped:", e)
    print(L.report())
