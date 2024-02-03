import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix,
    multilabel_confusion_matrix,
    ConfusionMatrixDisplay,
)
from source.random_forest import RandomForest
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import copy
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


data = pd.read_csv("datasets/heart.csv")
# data = pd.read_csv("datasets/Parkinsson_disease.csv")
# data = data.drop(['name'], axis=1)

# data = data.iloc[1: , :]
# data = data.drop(['name'], axis=1)
# print(data.head())
# print(data['output'].value_counts())

# Selective pruning!

# startTime = datetime.now()
# rf = RandomForest(forest_size=50, selective_pruning=True)
# rf.fit(data, target_feature='output')
# data['predictions'] = data.apply(rf.predict, axis=1)
# print(accuracy_score(data['output'], data['predictions']))
# print(recall_score(data['output'], data['predictions']))
# print(precision_score(data['output'], data['predictions']))
# print(f1_score(data['output'], data['predictions']))
# print("Time for trees = :" + str(datetime.now() - startTime))

# print(confusion_matrix(data['output'], data['predictions']))

# HEATMAP
# sns.heatmap(confusion_matrix(data['output'],
# data['predictions'])/np.sum(confusion_matrix(data['output'],
# data['predictions'])), annot=True,
#             fmt='.2%', cmap='Blues')
# plt.show()

# Max depth


def get_data_for_max_depth(value):
    acc = []
    recall = []
    precision = []
    f1 = []
    time = []
    for i in range(0, 6):
        print(value - 2 * i)
        copy_data = data.copy(deep=True)
        startTime = datetime.now()
        rf = RandomForest(
            forest_size=50, selective_pruning=False, max_depth=value - 2 * i
        )
        rf.fit(copy_data, target_feature="output")
        copy_data["predictions"] = copy_data.apply(rf.predict, axis=1)
        acc.append(
            accuracy_score(copy_data["output"], copy_data["predictions"])
        )
        recall.append(
            recall_score(copy_data["output"], copy_data["predictions"])
        )
        precision.append(
            precision_score(copy_data["output"], copy_data["predictions"])
        )
        f1.append(f1_score(copy_data["output"], copy_data["predictions"]))
        time.append((datetime.now() - startTime).total_seconds())
    return acc, recall, precision, f1, time


# print(get_data_for_max_depth(13))


def get_data_for_min_rows(value):
    acc = []
    recall = []
    precision = []
    f1 = []
    time = []
    for i in range(0, 4):
        copy_data = data.copy(deep=True)
        startTime = datetime.now()
        rf = RandomForest(forest_size=50, min_rows=value + 3 * i)
        rf.fit(copy_data, target_feature="output")
        copy_data["predictions"] = copy_data.apply(rf.predict, axis=1)
        acc.append(
            accuracy_score(copy_data["output"], copy_data["predictions"])
        )
        recall.append(
            recall_score(copy_data["output"], copy_data["predictions"])
        )
        precision.append(
            precision_score(copy_data["output"], copy_data["predictions"])
        )
        f1.append(f1_score(copy_data["output"], copy_data["predictions"]))
        time.append((datetime.now() - startTime).total_seconds())
    return acc, recall, precision, f1, time


# 2 5 8 11
print(get_data_for_min_rows(2))


def get_data_for_split_density(value):
    acc = []
    recall = []
    precision = []
    f1 = []
    time = []
    for i in range(0, 5):
        copy_data = data.copy(deep=True)
        startTime = datetime.now()
        rf = RandomForest(forest_size=50, split_search_density=value - 3 * i)
        rf.fit(copy_data, target_feature="output")
        copy_data["predictions"] = copy_data.apply(rf.predict, axis=1)
        acc.append(
            accuracy_score(copy_data["output"], copy_data["predictions"])
        )
        recall.append(
            recall_score(copy_data["output"], copy_data["predictions"])
        )
        precision.append(
            precision_score(copy_data["output"], copy_data["predictions"])
        )
        f1.append(f1_score(copy_data["output"], copy_data["predictions"]))
        time.append((datetime.now() - startTime).total_seconds())
    return acc, recall, precision, f1, time


# print(get_data_for_split_density(15))


def get_data_for_impurity_threshhold(value):
    acc = []
    recall = []
    precision = []
    f1 = []
    time = []
    for i in range(0, 7):
        print(0.1 * i)
        copy_data = data.copy(deep=True)
        startTime = datetime.now()
        rf = RandomForest(
            forest_size=50, min_impurity_treshold=value + (0.1 * i)
        )
        rf.fit(copy_data, target_feature="output")
        copy_data["predictions"] = copy_data.apply(rf.predict, axis=1)
        acc.append(
            accuracy_score(copy_data["output"], copy_data["predictions"])
        )
        recall.append(
            recall_score(copy_data["output"], copy_data["predictions"])
        )
        precision.append(
            precision_score(copy_data["output"], copy_data["predictions"])
        )
        f1.append(f1_score(copy_data["output"], copy_data["predictions"]))
        time.append((datetime.now() - startTime).total_seconds())
    return acc, recall, precision, f1, time


# print(get_data_for_impurity_threshhold(0))


def get_data_for_bootstrapping(value):
    acc = []
    recall = []
    precision = []
    f1 = []
    time = []
    for i in range(0, 7):
        print(str(value - (0.1 * i)))
        copy_data = data.copy(deep=True)
        startTime = datetime.now()
        rf = RandomForest(forest_size=50, max_samples=value - (0.1 * i))
        rf.fit(copy_data, target_feature="output")
        copy_data["predictions"] = copy_data.apply(rf.predict, axis=1)
        acc.append(
            accuracy_score(copy_data["output"], copy_data["predictions"])
        )
        recall.append(
            recall_score(copy_data["output"], copy_data["predictions"])
        )
        precision.append(
            precision_score(copy_data["output"], copy_data["predictions"])
        )
        f1.append(f1_score(copy_data["output"], copy_data["predictions"]))
        time.append((datetime.now() - startTime).total_seconds())
    return acc, recall, precision, f1, time


# print(get_data_for_bootstrapping(1))


def get_data_for_feature_bagging(value):
    acc = []
    recall = []
    precision = []
    f1 = []
    time = []
    for i in range(0, 5):
        print(str(value - (0.1 * i)))
        copy_data = data.copy(deep=True)
        startTime = datetime.now()
        rf = RandomForest(forest_size=50, max_features=value - (0.1 * i))
        rf.fit(copy_data, target_feature="output")
        copy_data["predictions"] = copy_data.apply(rf.predict, axis=1)
        acc.append(
            accuracy_score(copy_data["output"], copy_data["predictions"])
        )
        recall.append(
            recall_score(copy_data["output"], copy_data["predictions"])
        )
        precision.append(
            precision_score(copy_data["output"], copy_data["predictions"])
        )
        f1.append(f1_score(copy_data["output"], copy_data["predictions"]))
        time.append((datetime.now() - startTime).total_seconds())
    return acc, recall, precision, f1, time


# print(get_data_for_feature_bagging(1))


def get_vs_scikit_forest_size(data, target_feature):
    own = {"acc": [], "recall": [], "precision": [], "f1": [], "time": []}
    scikit = copy.deepcopy(own)
    train, test_arr = train_test_split(data, test_size=0.25)
    for f_size in [10, 100, 200, 400, 800, 1600]:
        print("curr forest size: " + str(f_size))
        print("own")
        copy_data = train.copy(deep=True)
        test_arr_own = test_arr.copy(deep=True)
        startTime = datetime.now()
        rf = RandomForest(
            forest_size=f_size, max_features=0.5, min_impurity_treshold=0.3
        )
        rf.fit(copy_data, target_feature=target_feature)
        test_arr_own["predictions"] = test_arr_own.apply(rf.predict, axis=1)
        own["acc"].append(
            accuracy_score(
                test_arr_own[target_feature], test_arr_own["predictions"]
            )
        )
        own["recall"].append(
            recall_score(
                test_arr_own[target_feature], test_arr_own["predictions"]
            )
        )
        own["precision"].append(
            precision_score(
                test_arr_own[target_feature], test_arr_own["predictions"]
            )
        )
        own["f1"].append(
            f1_score(test_arr_own[target_feature], test_arr_own["predictions"])
        )
        own["time"].append((datetime.now() - startTime).total_seconds())
        print("scikit")

        copy_data = train.copy(deep=True)
        test_arr_scikit = test_arr.copy(deep=True)
        startTime = datetime.now()
        rf = RandomForestClassifier(n_estimators=f_size)
        rf.fit(
            copy_data.drop(columns=[target_feature]),
            copy_data[target_feature].values.ravel(),
        )
        test_arr_scikit["predictions"] = rf.predict(
            test_arr_scikit.drop(columns=[target_feature])
        )
        scikit["acc"].append(
            accuracy_score(
                test_arr_scikit[target_feature], test_arr_scikit["predictions"]
            )
        )
        scikit["recall"].append(
            recall_score(
                test_arr_scikit[target_feature], test_arr_scikit["predictions"]
            )
        )
        scikit["precision"].append(
            precision_score(
                test_arr_scikit[target_feature], test_arr_scikit["predictions"]
            )
        )
        scikit["f1"].append(
            f1_score(
                test_arr_scikit[target_feature], test_arr_scikit["predictions"]
            )
        )
        scikit["time"].append((datetime.now() - startTime).total_seconds())
    with open("own_vs_scikit.txt", "a") as f:
        print(own, file=f)
        print(scikit, file=f)
    fig, ax = plt.subplots(1, 2)
    # Heatmap from CM for own
    sns.heatmap(
        confusion_matrix(
            test_arr_own[target_feature], test_arr_own["predictions"]
        )
        / np.sum(
            confusion_matrix(
                test_arr_own[target_feature], test_arr_own["predictions"]
            )
        ),
        annot=True,
        fmt=".2%",
        cmap="Blues",
        ax=ax[0],
    )
    # Heatmap from CM for scikits
    sns.heatmap(
        confusion_matrix(
            test_arr_scikit[target_feature], test_arr_scikit["predictions"]
        )
        / np.sum(
            confusion_matrix(
                test_arr_scikit[target_feature], test_arr_scikit["predictions"]
            )
        ),
        annot=True,
        fmt=".2%",
        cmap="Blues",
        ax=ax[1],
    )
    ax[0].title.set_text("Nasza implementacja")
    ax[1].title.set_text("Scikit")
    plt.show()
    return own, scikit


# print(get_vs_scikit_forest_size(data, "status"))


def get_vs_scikit_multi_class(data, target_feature, all_labels):
    own = {"acc": [], "recall": [], "precision": [], "f1": [], "time": []}
    scikit = copy.deepcopy(own)
    train, test_arr = train_test_split(data, test_size=0.25)
    for f_size in [20]:
        print("curr forest size: " + str(f_size))
        print("own")
        copy_data = train.copy(deep=True)
        test_arr_own = test_arr.copy(deep=True)
        startTime = datetime.now()
        rf = RandomForest(
            forest_size=f_size, max_features=0.5, min_impurity_treshold=0.3
        )
        rf.fit(copy_data, target_feature=target_feature)
        test_arr_own["predictions"] = test_arr_own.apply(rf.predict, axis=1)
        own["acc"].append(
            accuracy_score(
                test_arr_own[target_feature], test_arr_own["predictions"]
            )
        )
        own["recall"].append(
            recall_score(
                test_arr_own[target_feature],
                test_arr_own["predictions"],
                average="micro",
            )
        )
        own["precision"].append(
            precision_score(
                test_arr_own[target_feature],
                test_arr_own["predictions"],
                average="micro",
            )
        )
        own["f1"].append(
            f1_score(
                test_arr_own[target_feature],
                test_arr_own["predictions"],
                average="micro",
            )
        )
        own["time"].append((datetime.now() - startTime).total_seconds())
        print("scikit")

        copy_data = train.copy(deep=True)
        test_arr_scikit = test_arr.copy(deep=True)
        startTime = datetime.now()
        rf = RandomForestClassifier(n_estimators=f_size)
        rf.fit(
            copy_data.drop(columns=[target_feature]),
            copy_data[target_feature].values.ravel(),
        )
        test_arr_scikit["predictions"] = rf.predict(
            test_arr_scikit.drop(columns=[target_feature])
        )
        scikit["acc"].append(
            accuracy_score(
                test_arr_scikit[target_feature], test_arr_scikit["predictions"]
            )
        )
        scikit["recall"].append(
            recall_score(
                test_arr_scikit[target_feature],
                test_arr_scikit["predictions"],
                average="micro",
            )
        )
        scikit["precision"].append(
            precision_score(
                test_arr_scikit[target_feature],
                test_arr_scikit["predictions"],
                average="micro",
            )
        )
        scikit["f1"].append(
            f1_score(
                test_arr_scikit[target_feature],
                test_arr_scikit["predictions"],
                average="micro",
            ),
        )
        scikit["time"].append((datetime.now() - startTime).total_seconds())
    with open("own_vs_scikit.txt", "a") as f:
        print(target_feature, file=f)
        print(own, file=f)
        print(scikit, file=f)

        # plt.title("Scikit")
        # cm_sci = confusion_matrix(test_arr_scikit[target_feature], test_arr_scikit["predictions"])
        # disp = ConfusionMatrixDisplay(confusion_matrix=cm_sci, display_labels=all_labels)
        # disp.plot()

        plt.title("Nasza implementacja")
        cm_our = confusion_matrix(
            test_arr_own[target_feature], test_arr_own["predictions"]
        )
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm_our, display_labels=all_labels
        )
        disp.plot()

        plt.show()

    # fig, ax = plt.subplots(1,2)
    # # Heatmap from CM for own
    # sns.heatmap(confusion_matrix(test_arr_own[target_feature],
    # test_arr_own["predictions"])/np.sum(confusion_matrix(test_arr_own[target_feature],
    # test_arr_own["predictions"])), annot=True,
    #             fmt='.2%', cmap='Blues', ax=ax[0])
    # # Heatmap from CM for scikits
    # sns.heatmap(confusion_matrix(test_arr_scikit[target_feature],
    # test_arr_scikit["predictions"])/np.sum(confusion_matrix(test_arr_scikit[target_feature],
    # test_arr_scikit["predictions"])), annot=True,
    #             fmt='.2%', cmap='Blues', ax=ax[1])
    # ax[0].title.set_text("Nasza implementacja")
    # ax[1].title.set_text("Scikit")
    # plt.show()
    return own, scikit


# data = pd.read_csv("datasets/WineQuality.csv")
# data = data.drop(['Id'], axis=1)
# print(get_vs_scikit_multi_class(data, "quality",['5', '6', '7', '4', '8', '3']))


# data = pd.read_csv("datasets/Date_Fruit_Datasets.csv")

# print(get_vs_scikit_multi_class(data, "Class", ['BERHI', 'DEGLET', 'DOKOL', 'IRAQI', 'ROTANA', 'SAFAVI', 'SOGAY']))


# data = pd.read_csv("datasets/Rice_MSC_Dataset.csv")

# print(get_vs_scikit_multi_class(data, "CLASS", ['Basmati', 'Arborio', 'Jasmine', 'Ipsala', 'Karacadag']))

# print(data['CLASS'].unique())


# sns.heatmap(multilabel_confusion_matrix(
# data['quality'], data['predictions']))
