"""The numeric limits and constants the tools enforce. Changing one changes what an agent sees."""

from __future__ import annotations

OPERATORS = (">", ">=", "<", "<=", "==", "!=")  # the comparisons query_pool accepts
DEFAULT_QUERY_LIMIT = 20  # rows query_pool returns when the agent gives no limit
MAX_QUERY_LIMIT = 50  # rows query_pool returns at most
MAX_RANK = 50  # designs fit_and_rank and fit_custom_model return at most
MAX_SIMILAR = 20  # neighbours similar_designs returns at most
CALIBRATION_MIN_ROUNDS = 3  # realized model predictions needed before calibrating (D12)
FLOAT_GUARD = 1e-9  # keeps floor((1 - eps) * k) from losing one to float error
Z90 = 1.6448536269514722  # 95th percentile of the standard normal: a 90% two-sided interval
