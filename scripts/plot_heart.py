import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    n = 800
    x = np.linspace(-1.5, 1.5, n)
    y = np.linspace(-1.5, 1.5, n)
    X, Y = np.meshgrid(x, y)
    F = (X**2 + Y**2 - 1)**3 - X**2 * Y**3

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.contour(X, Y, F, levels=[0], colors=('crimson',), linewidths=(2,))
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.axis('off')

    out = os.path.join(os.path.dirname(__file__), 'heart.png')
    fig.savefig(out, dpi=300, bbox_inches='tight', pad_inches=0)
    print(out)

if __name__ == '__main__':
    main()
