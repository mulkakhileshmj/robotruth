"""robotruth: Robot CI for learned robot policies.

Three truths, one report:

- configuration truth: is the policy you evaluated the policy you deployed?
  (`robotruth.contract`)
- statistical truth: is checkpoint B really better than A? (`robotruth.stats`)
- outcome truth: did the episode succeed, when did it fail, and why?
  (`robotruth.schema`, `robotruth.judge`, later modules)
"""

__version__ = "0.1.3"
