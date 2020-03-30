import pickle
import matplotlib.pyplot as plt
import numpy as np

# load data
files = ["AL_scripts/results/CIFAR10_RandomSampling_accuracies.pkl", "AL_scripts/results/CIFAR10_UncertaintySampling_accuracies.pkl", "AL_scripts/results/CIFAR10_MarginSampling_accuracies.pkl"]

with open("AL_scripts/results/parameters", "rb") as f:
    parameters = pickle.load(f)

# collect results
results = []
for file in files:
    with open(file, "rb") as f:
        result = pickle.load(f)
        results.append(result)

# make plot
x = np.linspace(parameters["n_train"], parameters["n_train"]+ parameters["n_rounds"]*parameters["n_query"], parameters["n_rounds"]+1)

for i in range(len(results)):
    plt.plot(x, results[i], linewidth = 1)
    plt.scatter(x, results[i],  s = 10)

plt.title("Test accuracy for the different query strategies")
plt.xlabel("Number of training samples")
plt.ylabel("Accuracy [%]")
plt.legend(parameters["strategy"], title = "Query strategy")
plt.savefig("AL_scripts/results/CIFAR10_results")
