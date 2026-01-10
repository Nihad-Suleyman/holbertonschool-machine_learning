#!/usr/bin/env python3
"""now we will delve into plotting"""


import matplotlib.pyplot as plt
import numpy as np


def line():
    """we will evaluate a cubic function"""
    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))

    plt.plot(y, color='red')
    plt.xlim(0, 10)
