import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
data = rng.uniform(0, 100, 200)
offset = 50

centered = data - offset                       # offset 50 -> 0
log_centered = np.log10(np.abs(centered) + 1)   # log10(|data-offset| + 1)
log_plain = np.log10(data + 1)                  # log10(data + 1)

x = np.arange(len(data))
fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

axes[0].plot(x, data, ".-", color="steelblue")
axes[0].axhline(offset, color="gray", ls="--", lw=1, label=f"offset={offset}")
axes[0].set_title("Raw random data (0~100)")
axes[0].legend()

axes[1].plot(x, log_centered, ".-", color="darkorange")
axes[1].set_title("log10(|data - offset| + 1)")

axes[2].plot(x, log_plain, ".-", color="seagreen")
axes[2].set_title("log10(data + 1)")
axes[2].set_xlabel("index")

fig.tight_layout()
fig.savefig("compare_log.png", dpi=120)
print("saved compare_log.png")
plt.show()
