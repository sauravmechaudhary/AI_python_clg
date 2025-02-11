import matplotlib
import numpy as np
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

def simulate_dice_rolls():
    ns = [500, 1000, 2000, 5000, 10000, 15000, 20000, 50000, 100000]
    for n in ns:
        s = np.random.randint(1, 7, n) + np.random.randint(1, 7, n)
        h, h2 = np.histogram(s, bins=range(2, 14))
        plt.bar(h2[:-1], h / n)
        plt.title(f"Histogram of Dice Rolls (n={n})")
        plt.xlabel("Sum of Two Dice")
        plt.ylabel("Probability")
        plt.show()

if __name__ == "__main__":
    simulate_dice_rolls()
