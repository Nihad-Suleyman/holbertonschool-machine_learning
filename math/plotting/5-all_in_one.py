#!/usr/bin/env python3
""" 5-all_in_one.py """
import numpy as np
import matplotlib.pyplot as plt

def all_in_one():
    """ SUBPLOOOOOTS """

    y0 = np.arange(0, 11) ** 3

    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)
    x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
    y1 += 180

    x2 = np.arange(0, 28651, 5730)
    r2 = np.log(0.5)
    t2 = 5730
    y2 = np.exp((r2 / t2) * x2)

    x3 = np.arange(0, 21000, 1000)
    r3 = np.log(0.5)
    t31 = 5730
    t32 = 1600
    y31 = np.exp((r3 / t31) * x3)
    y32 = np.exp((r3 / t32) * x3)

    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)

    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(3, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(y0, color='r')
    ax1.axis([0, 10, None, None])
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title("Men's Height vs Weight", size='x-small')
    ax2.set_xlabel('Height (in)', size='x-small')
    ax2.set_ylabel('Weight (lbs)', size='x-small')
    ax2.scatter(x1, y1, color='m')
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_title('Exponential Decay of C-14', size='x-small')
    ax3.set_xlabel('Time (years)', size='x-small')
    ax3.set_ylabel('Fraction Remaining', size='x-small')
    ax3.set_yscale('log')
    ax3.axis([0, 28650, None, None])
    ax3.plot(x2, y2)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_title('Exponential Decay of Radioactive Elements', size='x-small')
    ax4.set_xlabel('Time (years)', size='x-small')
    ax4.set_ylabel('Fraction Remaining', size='x-small')
    ax4.axis([0, 20000, 0, 1])
    ax4.plot(x3, y31, '--r')
    ax4.plot(x3, y32, 'g')
    ax4.legend(['C-14', 'Ra-226'], fontsize='x-small')
    ax5 = fig.add_sconstrained_layout=Trueubplot(gs[2, :])
    ax5.hist(student_grades, bins=[i for i in range(0, 101, 10)], edgecolor='black')
    ax5.set_xticks([i for i in range(0, 101, 10)])
    ax5.axis([0, 100, 0, 30])
    ax5.set_title('Project A', size='x-small')
    fig.suptitle('All in One')
