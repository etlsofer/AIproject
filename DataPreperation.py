import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
import numpy as np

def Getwinner(res):
    winner = []
    for val in res:
        if val > 0:
            winner.append(1)
        elif val < 0:
            winner.append(-1)
        else:
            winner.append(0)
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

def scale_data(train, valid, test):
    min_max_features = ['Occupation_Satisfaction', 'Financial_balance_score_(0-1)', '%Of_Household_Income', 'Yearly_IncomeK',
                        'Garden_sqr_meter_per_person_in_residancy_area', 'Yearly_ExpensesK', '%Time_invested_in_work',
                        'Avg_Satisfaction_with_previous_vote', 'Avg_government_satisfaction', '%_satisfaction_financial_policy',
                        'Last_school_grades']
    z_score_features = ['Avg_monthly_expense_when_under_age_21', 'Avg_lottary_expanses', 'Avg_monthly_expense_on_pets_or_plants',
                        'Avg_environmental_importance', 'Avg_size_per_room', 'Avg_Residancy_Altitude', 'Avg_education_importance',
                        'Avg_monthly_household_cost', 'Phone_minutes_10_years', 'Weighted_education_rank',
                        'Avg_monthly_income_all_years', 'Number_of_differnt_parties_voted_for', 'Political_interest_Total_Score',
                        'Number_of_valued_Kneset_members', 'Num_of_kids_born_last_10_years', 'Overall_happiness_score']

    for f in min_max_features:
        max, min = train[f].max(), train[f].min()
        train[f], valid[f], test[f] = min_max_scale(train[f],min, max), min_max_scale(valid[f],min, max), min_max_scale(test[f],min, max)

    for f in z_score_features:
        mean, std = train[f].mean(), train[f].std()
        train[f], valid[f], test[f] = z_score_scale(train[f],mean,std), z_score_scale(valid[f],mean,std), z_score_scale(test[f],mean,std)

    return train, valid, test

def main():
    raw_data = pd.read_csv("final data.csv", header=0)
    # Add random and winner (as label)
    AddColumns(raw_data)
    numeric_data = transform_to_numeric(raw_data)
    numeric_data = transform_season_to_one_hot(numeric_data)
    numeric_data.to_csv("temp.csv")
    train, valid, test = split_data(numeric_data)
    train_Y, valid_Y, test_Y = train.winner, valid.winner, test.winner
    train_X, valid_X, test_X = train.drop(columns=['winner']), valid.drop(columns=['winner']), test.drop(columns=['winner'])
    train_X, train_Y = remove_outlier(train_X, train_Y)
    #train_X, valid_X, test_X = scale_data(train_X, valid_X, test_X)



    pd.DataFrame(data=train_X).to_csv("temp.csv")



if __name__ == "__main__":
    main()