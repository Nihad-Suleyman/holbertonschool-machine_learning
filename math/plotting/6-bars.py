#!/usr/bin/env python3
""" 6-bars.py """
import numpy as np
import matplotlib.pyplot as plt


def bars():
    """ bargraph """

    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    plt.figure(figsize=(6.4, 4.8))
    x = ['Farrah', 'Fred', 'Felicia']

    plt.bar(x, fruit[0], width=0.5, color='r')
    plt.bar(x, fruit[1], width=0.5, bottom=fruit[0], color='yellow')
    plt.bar(x, fruit[2], width=0.5, bottom=fruit[0]+fruit[1], color='#ff8000')
    plt.bar(
        x, fruit[3],
        width=0.5,
        bottom=fruit[0]+fruit[1]+fruit[2],
        color='#ffe5b4'
    )
    plt.title('Number of Fruit per Person')
    plt.ylabel('Quantity of Fruit')
    plt.yticks([i for i in range(0, 81, 10)])
    plt.legend(['apples', 'bananas', 'oranges', 'peaches'])
    plt.show()
