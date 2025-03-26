import matplotlib.pyplot as plt
import numpy as np



def plot_factors(factors, img_name = "test"):

    yaxis =  [i for i in range(16)]
    xaxis = factors * 10
    factors_labels = ["Warmth", "Reasoning", "Emotional stability", "Dominance",
                        "Liveliness", "Rule-consciousness", "Social boldness", "Sensitivity",
                        "Vigilance", "Abstractedness", "Privateness", "Apprehension",
                        "Openness to change", "Self-reliance", "Perfectionism", "Tension"
                    ]

    plt.rcParams["figure.figsize"] = (18, 10.5)  
    fig, axis = plt.subplots()
    axis.plot(xaxis, yaxis[::-1], marker = "o", markersize = 10)
    axis.set_yticks([x for x in range(16)], factors_labels[::-1])
    axis.set_xlim(left = 0.0, right = 1.0)
    axis.set_xticks([x for x in range(11)])
    axis.tick_params(labelsize=15)
    axis.grid(True, which="both")
    axis.margins(y=0.07)

      
    plt.savefig(f"imgs/{img_name}.png") 


def plot_elbow(visualizer):
    visualizer.show(outpath = "imgs/elbow.png")


def plot_clusters(data_reduced, labels, centroids):
    
    fig, axis = plt.subplots()

    unique_labels = sorted(np.unique(labels))
    num_colors = len(unique_labels)

    if (num_colors > 6):
        colormap = plt.get_cmap("gist_rainbow")
        axis.set_prop_cycle(color = [colormap(1.*i/num_colors) for i in range(num_colors)])


    for i in unique_labels:
        axis.scatter(data_reduced[labels == i, 0], data_reduced[labels == i, 1], label = f"Cluster {i}", s = 2)
    axis.scatter(centroids[:, 0], centroids[:, 1], s = 15, color = "k", label = "Centroids")
    plt.legend()
    plt.savefig(f"imgs/clusters.png", dpi = 500)

def plot_learning_curves(model_name, sizes, train_errors, test_errors):
    fig, axis = plt.subplots()
    axis.set_xlabel("Training set sizes")
    axis.set_ylabel("Errors")
    axis.plot(sizes, train_errors, label = "Train errors")
    axis.plot(sizes, test_errors, "-", label = "Test errors")
    axis.legend()
    fig.savefig(f"imgs/{model_name}_learning_curve.png")

