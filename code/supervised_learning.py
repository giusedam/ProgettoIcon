from copy import deepcopy
import data_preprocessing
from data_visualization import plot_learning_curves, plot_factors
import random
from sklearn import tree, linear_model
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import KFold, cross_val_score, train_test_split, GridSearchCV, learning_curve
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import pickle


hyperparams = {
    "DecisionTree":{
        "criterion":["gini", "entropy", "log_loss"],
        "splitter":["best"],
        "max_depth":[None, 1, 2, 5, 10, 25, 50],
        "min_samples_split": [2, 5, 10, 25, 50, 100],
        "min_samples_leaf": [1, 2, 5, 10, 25, 50, 100], 
    },
    "LogisticRegression": {
        "C":[0.001, 0.01, 0.1, 1, 10, 100, 1000],
        "solver":["lbfgs", "newton-cg", "sag", "saga"],
        "max_iter": list(range(250,2500,250))
    },
    "RandomForest": {
        "n_estimators": [5, 10, 25, 50, 100],
        "criterion":["gini", "entropy", "log_loss"],
        "max_depth":[None, 1, 2, 5, 10, 25, 50],
        "min_samples_split": [2, 5, 10, 25, 50, 100],
        "min_samples_leaf": [1, 2, 5, 10, 25, 50, 100], 
    },
    "GradientBoosting": {
        "loss":["log_loss", "exponential"],
        "learning_rate":[0.0, 0.1, 1, 10],
        "n_estimators":[10, 50, 100, 250],
        "max_depth":[None, 1, 2, 5, 10, 25],
        "criterion":["friedman_mse", "squared_error"],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 5, 10],
    }
}

# Same parameter configurations as in "params_config.pkl", saved as a global variable for ease of use

best_hyperparameters = {
    "DecisionTree": {'criterion': 'log_loss', 'max_depth': None, 'min_samples_leaf': 5, 'min_samples_split': 2, 'splitter': 'best'},
    "LogisticRegression": {'C': 1000, 'max_iter': 250, 'solver': 'newton-cg'},
    "RandomForest": {'criterion': 'gini', 'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100},
    "GradientBoosting": {'criterion': 'squared_error', 'learning_rate': 1, 'loss': 'log_loss', 'max_depth': 5, 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 250}
}



models = [(tree.DecisionTreeClassifier, "DecisionTree"), (linear_model.LogisticRegression, "LogisticRegression"),
                (RandomForestClassifier, "RandomForest"), (GradientBoostingClassifier, "GradientBoosting")]

class Dataset(object):
    
    def __init__(self, data = None, test_prob = 0.20, target_index = 0, seed = None):

        if seed:
            random.seed(seed)

        if data is None:
            return

        train_targets = []
        test_targets = []
        train = []
        test = []
        labels = [chr(c) for c in range(ord("A"), ord("P")+1)]
        target_label = "Cluster"
        random.shuffle(data)
        for example in data:
            target = example.pop(target_index)
            is_test = random.random() < test_prob
            if is_test:
                test.append(example)
                test_targets.append(target)
            else:
                train.append(example)
                train_targets.append(target)
        self.train = train
        self.test = test
        self.train_targets = train_targets
        self.test_targets = test_targets
        self.labels = labels
        self.target_label = target_label




def hyperparameter_tuning(dataset, model, model_name):
    search = GridSearchCV(model, hyperparams[model_name], cv=5, n_jobs=-1, verbose=3)
    search.fit(dataset.train, dataset.train_targets)
    return search.best_params_
    

def model_validation(dataset, model):
    results = {}
    metrics = ["accuracy", "f1_macro", "precision_macro", "recall_macro"]

    kfold = KFold(n_splits = 10)
    for metric in metrics:
        results[metric] = np.mean(cross_val_score(model, dataset.train, dataset.train_targets, scoring=metric, cv = kfold, n_jobs=-1))

    return results




def test_model(dataset, model, num_tests = 5):
    train_scores = []
    test_scores = []
    train_sizes = [0.1, 0.33, 0.5, 0.75, 1]

    for size in train_sizes:
            print(f"\tTest size: {int(size*len(dataset.train))}")
            train_accuracies = []
            test_accuracies = []
            for i in range(num_tests):
                if size != 1:
                    train_set, _, train_targets, _ = train_test_split(dataset.train, dataset.train_targets, train_size=size)
                    model.fit(train_set, train_targets)
                    train_predictions = model.predict(train_set)
                    train_accuracies.append(accuracy_score(train_targets, train_predictions))
                    test_predictions = model.predict(dataset.test)
                    test_accuracies.append(accuracy_score(dataset.test_targets, test_predictions))
                else:
                    data = dataset.train + dataset.test
                    targets = dataset.train_targets + dataset.test_targets
                    train_set, test_set, train_targets, test_targets = train_test_split(data, targets, train_size=0.8)
                    model.fit(train_set, train_targets)
                    train_predictions = model.predict(train_set)
                    train_accuracies.append(accuracy_score(train_targets, train_predictions))
                    test_predictions = model.predict(test_set)
                    test_accuracies.append(accuracy_score(test_targets, test_predictions))


            train_scores.append(train_accuracies)
            test_scores.append(test_accuracies)
    return train_scores, test_scores

def view_result(model_name, train_sizes, result):
    train_acc = np.array(result["train_accuracy"])
    test_acc = np.array(result["test_accuracy"])
    train_errors = 1 - train_acc
    test_errors = 1 - test_acc

    train_errors_mean = 1 - np.mean(train_acc, axis = 1)
    test_errors_mean = 1 - np.mean(test_acc, axis = 1)

    train_std = np.std(train_errors, axis = 1)
    train_var = np.var(train_errors, axis = 1)
    test_std = np.std(test_errors, axis = 1)
    test_var = np.var(test_errors, axis = 1)

    float_formatter = "{:.10f}".format
    np.set_printoptions(formatter={'float_kind':float_formatter})
    result_str = f"Train:\n\tError Variance: {train_var}\n\tError Std Deviation: {train_std}\nTest:\n\tError Variance: {test_var}\n\tError Std Deviation: {test_std}\n"
    print(f"{"-"*40} {model_name} {"-"*40}\n{result_str}")


    plot_learning_curves(model_name, train_sizes, train_errors_mean, test_errors_mean)

def save_item(filepath, item):
    with open(filepath, "wb") as f:
        pickle.dump(item, f)


def load_item(filepath):
    tree = None
    with open(filepath, "rb") as f:
        tree = pickle.load(f)
    return tree


def tune_models():
    print("[*] Querying the dataset...")
    data = data_preprocessing.query_to_dataset(data_preprocessing.data_files["small_prolog_clustered"], data_preprocessing.queries["factors_all_clustered"])
    print("[+] Dataset successfully queried")
    dataset = Dataset(data, target_index=-1)
    print("Targets: ", [dataset.train_targets[i] for i in range(50) ], " ...")


    optimal_parameters = {}
    for model, model_name in models:
        print(f"[*] Searching the best hyperparameters configuration for {model_name}...")
        best_results = hyperparameter_tuning(dataset, model, model_name)
        print(f"[+] Optimal hyperparameters configuration found for {model_name}:\n{best_results}")
        optimal_parameters[model_name] = best_results
    save_item("params_config.pkl", optimal_parameters)


def validate_models():
    print("[*] Querying the dataset...")
    data = data_preprocessing.query_to_dataset(data_preprocessing.data_files["small_prolog_clustered"], data_preprocessing.queries["factors_all_clustered"])
    print("[+] Dataset successfully queried")
    dataset = Dataset(data, target_index=-1)
    print("Targets: ", [dataset.train_targets[i] for i in range(50) ], " ...")


    filename = "validation_results.pkl"
    results = {}

    for model, model_name in models:
        print(f"[*] Evaluating metrics for {model_name}...")
        model = model(**best_hyperparameters[model_name])
        result = model_validation(dataset, model)
        print(f"[+] Metrics for {model_name}:\n {result}")
        results[model_name] = result

    print(f"[*] Saving results into {filename}...")
    save_item(filename, results)
    print(f"[+] Results successfully saved into {filename}")



def test_models():
    print("[*] Querying the dataset...")
    data = data_preprocessing.query_to_dataset(data_preprocessing.data_files["small_prolog_clustered"], data_preprocessing.queries["factors_all_clustered"])
    print("[+] Dataset successfully queried")
    dataset = Dataset(data, target_index=-1)
    print("Targets: ", [dataset.train_targets[i] for i in range(50) ], " ...")

    results = {mn:{} for m, mn in models}
    
    for model, model_name in models:
        print(f"[*] Testing {model_name}")
        model = model(**best_hyperparameters[model_name])
        train_scores, test_scores = test_model(dataset, model)
        results[model_name]["train_accuracy"] = train_scores
        results[model_name]["test_accuracy"] = test_scores

    
    print(f"[+] Results for all models:\n{results}")

    filename = "test_results.pkl"
    print(f"[*] Saving results in {filename}...")
    save_item(filename, results)
    print(f"[+] Results successfully saved into {filename}")


def view_test_results():
    filepath = "test_results.pkl"
    model_test_results = load_item(filepath)
    data = data_preprocessing.query_to_dataset(data_preprocessing.data_files["small_prolog_clustered"], data_preprocessing.queries["factors_all_clustered"])
    dataset = Dataset(data, target_index=-1)

    train_size = len(dataset.train)
    train_sizes = [0.1, 0.33, 0.5, 0.75, 1]
    train_sizes = [int(factor * train_size) for factor in train_sizes]

    for model, model_name in models:
        result = model_test_results[model_name]
        view_result(model_name, train_sizes, result)
        
# Main for testing on new examples
def test_on_new():
    pilot = np.array([0.45, 0.75, 0.8, 0.6, 0.5, 0.7, 0.7, 0.3, 0.5, 0.37, 0.65, 0.35, 0.45, 0.6, 0.8, 0.25])
    artist = np.array([0.45, 0.98, 0.7, 0.75, 0.45, 0.35, 0.75, 0.85, 0.45, 0.7, 0.53, 0.5, 0.62, 0.73, 0.72, 0.71])
    writer = np.array([0.3, 0.9, 0.72, 0.7, 0.38, 0.4, 0.77, 0.9, 0.55, 0.9, 0.45, 0.5, 0.62, 0.75, 0.74, 0.73])

    data = data_preprocessing.query_to_dataset(data_preprocessing.data_files["small_prolog_clustered"], data_preprocessing.queries["factors_all_clustered"])
    dataset = Dataset(data, target_index=-1)
    model = models[1][0](**best_hyperparameters[models[1][1]])
    model.fit(dataset.train, dataset.train_targets)
    predictions = model.predict([pilot, artist, writer])
    print(f"Cluster predictions:\n\tPilot: {predictions[0]}\n\tArtist: {predictions[1]}\n\tWriter: {predictions[2]}")






def main():
    test_on_new()

if __name__ == "__main__":
    main()



