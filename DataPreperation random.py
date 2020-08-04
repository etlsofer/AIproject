import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import accuracy_score
from ReliefF import ReliefF
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.neighbors import (NeighborhoodComponentsAnalysis,KNeighborsClassifier)

def Getwinner(res):
    winner = []
    for val in res:
        if val > 0:
            winner.append(1)
        else:
            winner.append(-1)
    return pd.DataFrame(data=winner)

def AddColumns(data):
    # Add random
    data["random"] = pd.DataFrame(data=np.random.randint(-1, 2, size=(len(data), 1)))
    #Add The number of goals the home group leads
    res = data["fthg"] - data["ftag"]
    data["The number of goals the home group leads"] = res
    #Add winner
    data["winner"] = Getwinner(res)
    #data.to_csv("temp.csv")

def transform_to_numeric(data):
    header = list(data.head(0))
    header.remove("random")
    header.remove("winner")
    data = data.drop(columns=header)

    return data

#split the data
def split_data(data):
    train_data = data.head(int(60*len(data)/100))
    valid_data = data.tail(int(40*len(data)/100)).head(int(20*len(data)/100))
    test_data = data.tail(int(20*len(data)/100))
    return train_data, valid_data, test_data

#Data Cleansing
def remove_outlier(data_X, data_Y):
    # combine X and Y to consider both when detecting outliers
    data = pd.concat([data_X, data_Y], axis=1)
    clf = IsolationForest(random_state=0, contamination=0.08)
    outlier_prediction = clf.fit_predict(data)
    index_to_drop_list = []
    for i in range(len(outlier_prediction)):
        if outlier_prediction[i] == -1:
            index_to_drop_list.append(i)
    return data_X.drop(index_to_drop_list), data_Y.drop(index_to_drop_list)

def transform_season_to_one_hot(data):
    one_hot = pd.get_dummies(data['season'], prefix='season')
    data = data.drop(columns=['season'])
    data = pd.concat([data, one_hot], axis=1)
    return data

def min_max_scale(column, min, max):
    if max - min == 0:
        return column
    return column.apply(lambda x: x  / (max - min) )

def scale_data(train, valid, test):
    min_max_features = list(train.head(0))
    print(min_max_features)
    for f in min_max_features:
        max, min = train[f].max(), train[f].min()
        train[f], valid[f], test[f] = min_max_scale(train[f],min, max), min_max_scale(valid[f],min, max), min_max_scale(test[f],min, max)
    return train, valid, test

def plot_feature_with_label(data, label):
    feature = [item for item in data.head(0)]
    num = [30,31,32,33,34,35,36,37]
    for i in num:
        plt.figure()
        plt.hist2d(data[feature[i]], label,bins=(50, 50), cmap=plt.cm.BuPu)
        plt.ylabel('Label')
        plt.xlabel(feature[i])
        plt.show(block=True)

def check_performence(train_X, train_Y, valid_X, valid_Y):
    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(train_X, train_Y)
    res = clf.predict(valid_X)
    print("accuracy on validation set: {}".format(accuracy_score(valid_Y, res)))

def filter_with_relief(train_x, valid_x, test_x, train_y):
    temp_header = [item for item in train_x.head(0)]
    features_to_drop = []
    features_to_keep = 30
    data, target = train_x.to_numpy(), train_y.to_numpy()
    fs = ReliefF(n_neighbors=5, n_features_to_keep=features_to_keep)
    fs.fit(data, target)
    for index in fs.top_features[features_to_keep:]:
        features_to_drop += [temp_header[index]]
    return train_x.drop(features_to_drop, axis=1), valid_x.drop(features_to_drop, axis=1),\
           test_x.drop(features_to_drop, axis=1)

def filter_with_sfs(train_X, valid_X, test_X, train_Y, i):
    features = {item for item in train_X.head(0)}
    fs = SequentialFeatureSelector(RandomForestClassifier(n_estimators=30, random_state=0),
                                                 k_features=i,
                                                 forward=True,
                                                 verbose=0,
                                                 scoring='accuracy',
                                                 cv=4)
    fs.fit(train_X, train_Y)

    selected_features = set(fs.k_feature_names_)
    features_to_drop = list(features - selected_features)

    return train_X.drop(features_to_drop, axis=1), valid_X.drop(features_to_drop, axis=1), \
           test_X.drop(features_to_drop, axis=1)

def feature_selection(train_X, valid_X, test_X, train_Y, i):
    # remove Linear dependencies feature
    #train_X, valid_X, test_X = remove_linear_dependencies(train_X), remove_linear_dependencies(valid_X),\
    #    remove_linear_dependencies(test_X)
    # using  filter method
    train_X, valid_X, test_X = filter_with_relief(train_X, valid_X, test_X, train_Y)
    # using  wrapper method
    train_X, valid_X, test_X = filter_with_sfs(train_X, valid_X, test_X, train_Y, i)
    return train_X, valid_X, test_X

def main():
    raw_data = pd.read_csv("final data.csv", header=0)
    # Add random and winner (as label)
    AddColumns(raw_data)
    numeric_data = transform_to_numeric(raw_data)
    #numeric_data = transform_season_to_one_hot(numeric_data)
    #numeric_data.to_csv("temp.csv")
    train, valid, test = split_data(numeric_data)
    train_Y, valid_Y, test_Y = train.winner, valid.winner, test.winner
    #train_X, valid_X, test_X = train.drop(columns=['winner']), valid.drop(columns=['winner']), test.drop(columns=['winner'])
    #check perfomans before startting preparation
    ################
    #print("performence before:")
    #check_performence(train_X, train_Y, valid_X, valid_Y)
    ##############
    # check perfomans without the winning columns
    train_X, valid_X, test_X = train.drop(columns=['winner']
                                          ), valid.drop(columns=['winner']
                                                        ), test.drop(columns=['winner'])
    print("performence before:")
    check_performence(train_X, train_Y, valid_X, valid_Y)
    train_X, train_Y = remove_outlier(train_X, train_Y)
    print("performence after outlier dection:")
    check_performence(train_X, train_Y, valid_X, valid_Y)

    train_X, valid_X, test_X = scale_data(train_X, valid_X, test_X)
    print("performence after scaling:")
    check_performence(train_X, train_Y, valid_X, valid_Y)


    t_train_X, t_valid_X, t_test_X = feature_selection(train_X, valid_X, test_X, train_Y, 30)
    print("performence after " +str(i)+ " feature selection:")
    check_performence(t_train_X, train_Y, t_valid_X, valid_Y)

    final_train = pd.concat([train_Y, train_X], axis=1)
    final_train.to_csv("processedTrainData.csv", index=False)


    pd.DataFrame(data=train_X).to_csv("temp.csv")



if __name__ == "__main__":
    main()


