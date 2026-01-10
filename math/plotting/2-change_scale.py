#!/usr/bin/env python3
"""we will now use exponential function"""


import numpy as np
import matplotlib.pyplot as plt


def change_scale():
    """NOw we are changing scale here"""
    x = np.arange(0, 28651, 5730)
    r = np.log(0.5)
    t = 5730
    y = np.exp((r / t) * x)
    plt.figure(figsize=(6.4, 4.8))
    plt.plot(x, y)
    plt.ylabel('Fraction Remaining')
    plt.xlim(0, 28650)
    plt.xlabel('Time (years)')
    plt.xscale('log')
    plt.title('Exponential Decay of C-14')
    plt.show()
