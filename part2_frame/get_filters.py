#  In this file, you will define the get_filters() function.
#  The function whill plot all the filters we used in the paper.
# you should enlarge each filter to 64*64
from filters import get_filters
import numpy as np


def draw_filters():
    filters = get_filters()
    print("we have: ", len(filters), "filtters in all.")

    # enlarge them to 64*64

    for i, filter in enumerate(filters):
        filter = np.repeat(filter, 8, axis=0)
        filter = np.repeat(filter, 8, axis=1)
        print(filter.shape)
        import matplotlib.pyplot as plt
        # Save the filter without plt axis
        plt.imshow(filter, cmap='gray')
        plt.axis('off')
        plt.savefig(f"filters/filter{i}.png", pad_inches=0, bbox_inches='tight')


draw_filters()