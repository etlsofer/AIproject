import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
from ReliefF import ReliefF
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.svm import SVC
from sklearn.neighbors import (NeighborhoodComponentsAnalysis,KNeighborsClassifier)
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import naive_bayes
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier

def Getwinner(res):
    winner = []
    for val in res:
        if val > 0:
            winner.append(1)
        elif val == 0:
            winner.append(0)
        else:
            winner.append(-1)
    return pd.DataFrame(data=winner)

def BalansData(data):
    header = list(data.head(0))
    t_data = []
    counter = 0
    for raw in data.values:
        if raw[42] == -1:
            if counter > -814:
                t_data.append(raw)
            else:
                counter +=1
                continue
        else:
            t_data.append(raw)
    return pd.DataFrame(data=t_data, columns=header)



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
    # transform the rest (categorical) values
    ObjFeat = data.keys()[data.dtypes.map(lambda x: x == 'object')]
    for f in ObjFeat:
        data[f] = data[f].astype("category")
        data[f] = data[f].cat.rename_categories(range(data[f].nunique())).astype(float)
    # fix NaN conversion
    data.fillna(np.nan)
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
    clf = IsolationForest(random_state=0, contamination=0.01)
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
    return column.apply(lambda x: 2 * (x - min) / (max - min) - 1)

def z_score_scale(column, mean, std):
    if mean == 0:
        return column
    return column.apply(lambda x: (x - mean) / std)

def scale_data(train, valid, test):
    features_for_normalization = list(train.head(0))
    feature_for_min_max = ["date", "ht", "at"]
    feature_for_z_score = list(set(features_for_normalization) - set(feature_for_min_max))
    '''
    # features_for_normalization = [item for item in headers.head(0)]
    features_for_normalization_a = features_for_normalization[0:len(features_for_normalization) // 4]
    features_for_normalization_b = features_for_normalization[3*len(features_for_normalization) // 4: len(features_for_normalization)]
    plt.figure()
    for i in range(len(features_for_normalization_b)):
        plt.subplot(len(features_for_normalization_b)/3 +1, 3, i + 1)
        plt.hist(train[features_for_normalization_b[i]], density=False, bins=100)
        plt.ylabel('Count')
        plt.xlabel(features_for_normalization_b[i])
    plt.show()
    '''
    for f in feature_for_min_max:
        max, min = train[f].max(), train[f].min()
        train[f], valid[f], test[f] = min_max_scale(train[f],min, max), min_max_scale(valid[f],min, max), min_max_scale(test[f],min, max)

    for f in feature_for_z_score:
        mean, std = train[f].mean(), train[f].std()
        train[f], valid[f], test[f] = z_score_scale(train[f], mean, std), z_score_scale(valid[f], mean,
                                                                                        std), z_score_scale(test[f],
                                                                                                            mean, std)
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
    c1 = SVC(C=0.01,kernel="linear")
    c2 = RandomForestClassifier(n_estimators=50, max_depth=10)
    c3 = KNeighborsClassifier(n_neighbors=150)
    c4 = SGDClassifier(loss="huber", penalty="l1")
    c5 = DecisionTreeClassifier(criterion="gini", min_samples_split=250)
    c6 = LinearDiscriminantAnalysis(solver="lsqr")
    c7 = naive_bayes.BernoulliNB()
    c8 = MLPClassifier(hidden_layer_sizes=(5,3))
    c9 = GradientBoostingClassifier(random_state=0, n_estimators=100, learning_rate=0.1)
    clf = VotingClassifier(estimators=[('a', c1), ('f', c6), ('h', c8) ,
                                       ('i', c9)])
    clf.fit(train_X, train_Y)
    res = clf.predict(valid_X)
    #cross_val_score_avg = cross_val_score(estimator=clf, X=train_X, y=train_Y, cv=10, scoring='accuracy').mean()
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
    c1 = SVC(C=0.01, kernel="linear")
    c2 = RandomForestClassifier(n_estimators=50, max_depth=10)
    c3 = KNeighborsClassifier(n_neighbors=150)
    c4 = SGDClassifier(loss="huber", penalty="l1")
    c5 = DecisionTreeClassifier(criterion="gini", min_samples_split=250)
    c6 = LinearDiscriminantAnalysis(solver="lsqr")
    c7 = naive_bayes.BernoulliNB()
    c8 = MLPClassifier(hidden_layer_sizes=(5, 3))
    c9 = GradientBoostingClassifier(random_state=0, n_estimators=100, learning_rate=0.1)
    c10 = VotingClassifier(estimators=[('a', c1), ('b', c2)  ,('c', c3),('d', c4), ('e', c5)  ,('f', c6),('g', c7), ('h', c8) ,
                                       ('i', c9)])
    features = {item for item in train_X.head(0)}
    fs = SequentialFeatureSelector(c10,
                                                 k_features=i,
                                                 forward=False,
                                                 verbose=0,
                                                 scoring='accuracy',
                                                 cv=4)
    fs.fit(train_X, train_Y)

    selected_features = set(fs.k_feature_names_)
    print(fs.subsets_)
    features_to_drop = list(features - selected_features)

    return train_X.drop(features_to_drop, axis=1), valid_X.drop(features_to_drop, axis=1), \
           test_X.drop(features_to_drop, axis=1)

def feature_selection_with_corr(train_X, valid_X, test_X):
    feature_to_drop = ["num of events type 9 for size 1","num of events type 9 for size 2",
                       "num of events type 1 for size 1","num of events type 1 for size 2",
                       "num of events type 3 for size 2","num of events type 3 for size 1"]

    train_X, valid_X, test_X = train_X.drop(feature_to_drop, axis=1), valid_X.drop(feature_to_drop, axis=1),\
                               test_X.drop(feature_to_drop, axis=1)
    return train_X, valid_X, test_X

def plot_feature(data, feature):
    plt.figure()
    plt.hist(data[feature], density=False, bins=100)
    plt.ylabel('Count')
    plt.xlabel(feature)
    plt.show(block=True)

def check_corr(data):
    value = 0.95
    curr_cor = data.corr(method='pearson')
    curr_cor = curr_cor.applymap(lambda x: x if abs(x) > value else 0)
    curr_cor.to_csv("final feature.csv")

def feature_selection_final(train_X, valid_X, test_X):
    features = list(train_X.head(0))
    feature_to_keap = ["odd_h",
            "num of events type 2 for size 1",
            "num of events type 14 for size 1",
            "num of events type 14 for size 2",
            "num of events type 7 for size 1",
            "num of events type 15 for size 1",
            "num of events type 8 for size 1",
            "num of events type 2 for size 2",
            "num of events type 7 for size 2",
            "num of events type 15 for size 2",
            "num of events type 10 for size 2",
            "num of events type 4 for size 1",
            "num of events type 11 for size 1",
            "num of events type 13 for size 1",
            "date",
            "num of events type 6 for size 2",
            "num of events type 12 for size 1",
            "odd_d",
            "num of events type 12 for size 2",
            "at",
            "ht"]
    feature_to_drop = ["league",
                       "country",
                       "num of events type 13 for size 1",
                       "season_2014",
                       "num of events type 5 for size 2",
                       "at",
                       "num of events type 13 for size 2",
                       "random",
                       "ht"]
    return train_X.drop(columns=feature_to_drop), valid_X.drop(columns=feature_to_drop), test_X.drop(columns=feature_to_drop)

def main():
    raw_data = pd.read_csv("final data.csv", header=0)
    # Add random and winner (as label)
    AddColumns(raw_data)
    #raw_data = BalansData(raw_data)
    numeric_data = transform_to_numeric(raw_data)
    numeric_data = transform_season_to_one_hot(numeric_data)
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
    train_X, valid_X, test_X = train.drop(columns=['winner', 'fthg', 'ftag', 'The number of goals the home group leads']
                                          ), valid.drop(columns=['winner', 'fthg', 'ftag', 'The number of goals the home group leads']
                                                        ), test.drop(columns=['winner', 'fthg', 'ftag', 'The number of goals the home group leads'])
    #print("performence before:")
    #check_performence(train_X, train_Y, valid_X, valid_Y)
    train_X, train_Y = remove_outlier(train_X, train_Y)
    #print("performence after outlier dection:")
    #check_performence(train_X, train_Y, valid_X, valid_Y)

    train_X, valid_X, test_X = scale_data(train_X, valid_X, test_X)

    #print("performence after scaling with n = "+ str(n)+" and d = "+str(d))
    train_X, valid_X, test_X = feature_selection_with_corr(train_X, valid_X, test_X)

    #print("f = "+f+" k = "+str(k))
    #check_performence(train_X, train_Y, valid_X, valid_Y)

    t_train_X, t_valid_X, t_test_X = feature_selection_final(train_X, valid_X, test_X)
    print("performence after " + str(2) + " feature selection:")
    check_performence(t_train_X, train_Y, t_test_X, test_Y)

    '''
    size = train.shape[1]
    for i in range(size-4,5,-1):
        t_train_X, t_valid_X, t_test_X = feature_selection(train_X, valid_X, test_X, train_Y, i)
        print("performence after " + str(i) + " feature selection:")
        check_performence(t_train_X, train_Y, t_valid_X, valid_Y)
    '''
    #final_train = pd.concat([train_Y, train_X], axis=1)
    #final_train.to_csv("processedTrainData.csv", index=False)


    #pd.DataFrame(data=train_X).to_csv("temp.csv")



if __name__ == "__main__":
    main()


