#
# Copyright 2024 Sony Semiconductor Solutions Corp. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import time
import numpy as np
import matplotlib.pyplot as plt

from collections import deque
from contextlib import contextmanager
from modlib.devices import AiCamera
from modlib.models.zoo import DeepLabV3Plus


def calculate_cv(values) -> float:
    """Calculate coefficient of variation"""
    if not values:
        return float("inf")
    mean = sum(values) / len(values)
    if mean == 0:
        return float("inf")
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return (variance**0.5) / mean


def is_stable(values: deque, threshold: float = 0.1) -> bool:
    """Check if values have stabilized based on coefficient of variation"""
    if len(values) < 5:  # Need at least 5 samples to determine stability
        return False
    # Calculate coefficient of variation (CV) for each metric
    fps_cv = calculate_cv([v[0] for v in values])
    dps_cv = calculate_cv([v[1] for v in values])
    rps_cv = calculate_cv([v[2] for v in values])
    # print(f"CV values (FPS, DPS, RPS): {fps_cv:.3f}, {dps_cv:.3f}, {rps_cv:.3f}")
    # Check if all metrics are stable
    return all(cv <= threshold for cv in [fps_cv, dps_cv, rps_cv])


def measure_rates(device):
    stabilization_threshold = 0.015  # 1.5% variation threshold
    max_stabilization_time = 60  # seconds
    stabilization_values = deque(maxlen=10)
    final_values = deque(maxlen=30)
    stabilized = False
    stabilization_start_time = None

    with device as stream:
        for frame in stream:
            current_values = (frame.fps, frame.dps, stream.rps.value)

            # If not yet stabilized, collect stabilization data
            if not stabilized:
                stabilization_values.append(current_values)
                if stabilization_start_time is None:
                    stabilization_start_time = time.time()
                if is_stable(stabilization_values, stabilization_threshold):
                    stabilized = True
                    print("Rate values stabilized.")
                if time.time() - stabilization_start_time > max_stabilization_time:
                    stabilized = True
                    print("Stabilization time exceeded. Continuing...")
            # If stabilized, collect final average data
            else:
                final_values.append(current_values)
                if len(final_values) >= final_values.maxlen:
                    break

            # frame.display()

    # Calculate final averages
    avg_fps = sum(v[0] for v in final_values) / len(final_values)
    avg_dps = sum(v[1] for v in final_values) / len(final_values)
    avg_rps = sum(v[2] for v in final_values) / len(final_values)

    return avg_fps, avg_dps, avg_rps


def plot_metrics(rps, dps, model_name=""):
    fig, ax = plt.subplots(figsize=(12, 8))
    (line,) = ax.plot(rps, dps, "o-", lw=2, ms=8, mfc="white", mew=2, mec="#1f77b4", label="DPS vs RPS")
    ax.fill_between(rps, dps, alpha=0.2, color=line.get_color())

    i = np.argmax(dps)
    ax.plot(rps[i], dps[i], "o", ms=12, color="green", label=f"Max DPS: {dps[i]:.2f} @ frame_rate={int(round(rps[i]))}")
    ax.annotate(
        f"DPS: {dps[i]:.2f}\nRPS: {rps[i]:.2f}\nFrame Rate: {int(round(rps[i]))}",
        xy=(rps[i], dps[i]),
        xytext=(20, 20),
        textcoords="offset points",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.5", fc="green", alpha=0.2),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2"),
    )

    ax.tick_params(labelsize=10)
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend(loc="upper left", frameon=True, fancybox=True, shadow=True)
    ax.set(xlabel="Request Rate (RPS)", ylabel="Detection Rate (DPS)", title=f"{model_name}: DPS vs RPS")
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.show()


@contextmanager
def suppress_output():
    with open(os.devnull, "w") as d, os.fdopen(os.dup(2), "w") as o2:
        os.dup2(d.fileno(), 2)
        try:
            yield
        finally:
            os.dup2(o2.fileno(), 2)


if __name__ == "__main__":
    model = DeepLabV3Plus()

    frame_rates = range(5, 31)
    dps_values, rps_values = [], []

    for fr in frame_rates:
        print(f"\nMeasuring rates for frame rate: {fr}")
        with suppress_output():
            device = AiCamera(enable_input_tensor=False, frame_rate=fr)
            device.deploy(model, overwrite=False)
            avg_fps, avg_dps, avg_rps = measure_rates(device)

        dps_values.append(avg_dps)
        rps_values.append(avg_rps)
        print(f"Average FPS: {avg_fps:.2f}")
        print(f"Average DPS: {avg_dps:.2f}")
        print(f"Average RPS: {avg_rps:.2f}")

    plot_metrics(rps_values, dps_values, model_name=model.__class__.__name__)
