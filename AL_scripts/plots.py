import pickle
import matplotlib.pyplot as plt
import numpy as np

# load data
# files = ["AL_scripts/Cluster/CIFAR10/CIFAR10_RandomSampling_accuracies_2.pkl", 
# "AL_scripts/Cluster/CIFAR10/CIFAR10_UncertaintySampling_accuracies_2.pkl",
# "AL_scripts/Cluster/CIFAR10/CIFAR10_MarginSampling_accuracies_2.pkl",
# "AL_scripts/Cluster/CIFAR10/CIFAR10_QBCSampling_accuracies_2.pkl"]

prefix = "AL_scripts/Cluster/Xray/"
files = [prefix + "Xray_RandomSampling_accuracies.pkl",
prefix + "Xray_UncertaintySampling_accuracies.pkl",
prefix + "Xray_MarginSampling_accuracies.pkl",
prefix + "Xray_QBCSampling_accuracies.pkl"]

# with open("AL_scripts/Cluster/CIFAR10/CIFAR10_parameters_2.pkl", "rb") as f:
#     parameters = pickle.load(f)

with open(prefix + "parameters_Xray.pkl", "rb") as f:
    parameters = pickle.load(f)

# collect results
results = []
for file in files:
    with open(file, "rb") as f:
        result = pickle.load(f)
        results.append(result)

# make plot
x = np.linspace(parameters["n_train"], parameters["n_train"]+ parameters["n_rounds"]*parameters["n_query"], parameters["n_rounds"]+1)
# print(results, x)
for i in range(len(results)):
    plt.plot(x, results[i]*100, linewidth = 1)
    plt.scatter(x, results[i]*100,  s = 5)

plt.title("Test accuracy for the different query strategies")
plt.xlabel("Number of training samples")
plt.ylabel("Accuracy [%]")
plt.legend(parameters["strategy"], title = "Query strategy")
# plt.savefig("AL_scripts/results/"+parameters["dataset"]+"_results")
plt.savefig(prefix + parameters["dataset"]+"_results.png")