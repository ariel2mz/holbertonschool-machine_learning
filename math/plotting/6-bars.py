#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4,3))
plt.figure(figsize=(6.4, 4.8))

categories = ['Farrah', 'Fred',  'Felicia']
fruit_labels = ['apples', 'bananas', 'oranges', 'pears']
colors = ['red', 'yellow', 'orange', 'green']


# Plot
for i in range(4):  # for each fruit type
    plt.bar(categories, fruit[i], bottom=bottom, label=fruit_labels[i], color=colors[i])
    bottom += fruit[i]  #

# Labels and legend
plt.xlabel('Category')
plt.ylabel('Quantity of Fruit')
plt.title('Number of Fruit per Person')
plt.legend()

plt.show()