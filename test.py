# from functions import imshow
# import matplotlib.pyplot as plt


# image = "images/1-s2.0-S0140673620303706-fx1_lrg.jpg"

# img = plt.imread(image)
# plt.imshow(img)
# plt.show()

# imshow(img)


import pandas as pd
import numpy as np

metadata = pd.read_csv("metadata.csv")

print(np.unique(metadata["finding"], return_counts=True))