"""The organism's own learner: a world-time recurrent streaming actor-critic.

Written fresh in plain numpy (float64) with hand-written gradients.  No
module here imports torch; the tests use torch only as an independent
reference.
"""

from __future__ import annotations
