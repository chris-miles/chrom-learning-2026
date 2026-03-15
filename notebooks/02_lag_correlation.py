# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("..").resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chromlearn.analysis.lag_correlation import compute_lag_correlation, plot_lag_correlation
from chromlearn.io.catalog import load_condition

# %%
cells = load_condition("rpe18_ctr")
result = compute_lag_correlation(cells, lag_max=40, smooth_window=31)

# %%
plot_lag_correlation(result)
plt.show()

# %%
peak_index = int(np.nanargmax(result.median))
print(f"Peak correlation lag: {result.lags[peak_index]:.0f} seconds")
print(f"Peak correlation value: {result.median[peak_index]:.3f}")
