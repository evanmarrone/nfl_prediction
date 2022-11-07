
"""
NFL Predictions Functions Library

@author: Cameron Cubra
"""
import pandas as pd
pd.set_option('display.max_columns', 100)
import numpy as np
import matplotlib.pyplot as plt
# import seaborn
import seaborn as sns
from sklearn.model_selection import train_test_split # for splitting the data into train and test samples
from sklearn.metrics import classification_report # for model evaluation metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier # for random forest models
from sklearn.ensemble import GradientBoostingClassifier # for gradient boosting models
from sklearn.metrics import accuracy_score # for model evaluation metrics
from sklearn.linear_model import LogisticRegression # for logistic regression models
# import the necessary packages
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
# import cross validation
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier



# function to normalize the data with robust scaler
def normalize_data(df, quantile_range=(25.0, 75.0)):
    # normalize the data
    scaler = preprocessing.RobustScaler(quantile_range=quantile_range)
    scaled_df = scaler.fit_transform(df)
    df = pd.DataFrame(scaled_df, columns=df.columns)
    return df


# function to split the data into train, validation, and test samples with the last n games as the test sample
def split_data(dfX, Y, test_games):
    # split the data into train, validation, test samples with the last n games as the test sample
    X_train_full, X_test, y_train_full, y_test = dfX.iloc[:-test_games], dfX.iloc[-test_games:], Y[:-test_games], Y[-test_games:]
    # split the train sample into train and validation samples
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=2022)
    return X_train_full, X_train, X_val, X_test, y_train_full, y_train, y_val, y_test


# function to tune random forest parameters in grid search with 5 fold cross validation and return the best parameters and test accuracy and classification report
def tune_rf_grid(X_train_full, y_train_full, X_test, y_test, n_estimators=[1, 5, 10], max_depth=[1, 5, 10], min_samples_split=[2, 5, 10], min_samples_leaf=[1, 5, 10]):
    # tune random forest parameters in grid search with 5 fold cross validation
    # create the random forest model
    clf = RandomForestClassifier()
    # create the parameter grid with 3 values for each parameter
    param_grid = {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf
    }
    # create the grid search
    grid_search = GridSearchCV(estimator = clf, param_grid = param_grid, cv = 5, n_jobs = -1, verbose = 2)
    # fit the grid search
    grid_search.fit(X_train_full, y_train_full)
    # get the best parameters
    best_rf_params = grid_search.best_params_
    # print the best parameters
    print('Best Random Forest Parameters: ', best_rf_params)
    # create the random forest model with the best parameters
    clf = RandomForestClassifier(n_estimators=best_rf_params['n_estimators'], max_depth=best_rf_params['max_depth'], min_samples_split=best_rf_params['min_samples_split'], min_samples_leaf=best_rf_params['min_samples_leaf'])
    # fit the model
    clf = clf.fit(X_train_full, y_train_full)
    # predict the test set
    y_pred = clf.predict(X_test)
    # calculate the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    # print the results
    print('Random Forest Accuracy: ', accuracy)
    # print the classification report
    print(classification_report(y_test, y_pred))

    # return the best parameters in a dictionary with the accuracy
    return {'best_rf_params': best_rf_params, 'accuracy': accuracy}


# function to tune gradient boosting parameters in grid search with 5 fold cross validation and return the best parameters and test accuracy and classification report
def tune_gb_grid(X_train_full, y_train_full, X_test, y_test, n_estimators=[1, 5, 10], max_depth=[1, 5, 10], min_samples_split=[2, 5, 10], min_samples_leaf=[1, 5, 10]):
    # tune gradient boosting parameters in grid search with 5 fold cross validation
    # create the gradient boosting model
    clf = GradientBoostingClassifier()
    # create the parameter grid with 3 values for each parameter
    param_grid = {
        'n_estimators': n_estimators,   # number of trees
        'max_depth': max_depth,   # maximum depth of each tree
        'min_samples_split': min_samples_split,   # minimum number of samples required to split an internal node
        'min_samples_leaf': min_samples_leaf   # minimum number of samples required to be at a leaf node
    }
    # create the grid search
    grid_search = GridSearchCV(estimator = clf, param_grid = param_grid, cv = 5, n_jobs = -1, verbose = 0)
    # fit the grid search
    grid_search.fit(X_train_full, y_train_full)
    # get the best parameters
    best_gb_params = grid_search.best_params_
    # print the best parameters
    print('Best Gradient Boosting Parameters: ', best_gb_params)
    # create the gradient boosting model with the best parameters
    clf = GradientBoostingClassifier(n_estimators=best_gb_params['n_estimators'], max_depth=best_gb_params['max_depth'], min_samples_split=best_gb_params['min_samples_split'], min_samples_leaf=best_gb_params['min_samples_leaf'])
    # fit the model
    clf = clf.fit(X_train_full, y_train_full)
    # predict the test set
    y_pred = clf.predict(X_test)
    # calculate the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    # print the results
    print('Gradient Boosting Accuracy: ', accuracy)
    # print the classification report
    print(classification_report(y_test, y_pred))

    # return the best parameters in a dictionary with the accuracy
    return {'best_gb_params': best_gb_params, 'accuracy': accuracy}


# function to tune logistic regression parameters in grid search with 5 fold cross validation and return the best parameters and test accuracy and classification report
def tune_lr_grid(X_train_full, y_train_full, X_test, y_test, penalty = ['l2'], c = [0.001, 0.01, 0.1, 1, 10, 100, 1000]):
    # note the default penalty and c values above ^^^
    # tune logistic regression parameters in grid search with 5 fold cross validation
    # create the logistic regression model
    clf = LogisticRegression()
    # create the parameter grid with 3 values for each parameter
    param_grid = {
        'penalty': penalty,   # l1 = lasso, l2 = ridge
        'C': c   # inverse of regularization strength
    }
    # create the grid search
    grid_search = GridSearchCV(estimator = clf, param_grid = param_grid, cv = 5, n_jobs = -1, verbose = 2)
    # fit the grid search
    grid_search.fit(X_train_full, y_train_full)
    # get the best parameters
    best_lr_params = grid_search.best_params_
    # print the best parameters
    print('Best Logistic Regression Parameters: ', best_lr_params)
    # create the logistic regression model with the best parameters
    clf = LogisticRegression(penalty=best_lr_params['penalty'], C=best_lr_params['C'])
    # fit the model
    clf = clf.fit(X_train_full, y_train_full)
    # predict the test set
    y_pred = clf.predict(X_test)
    # calculate the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    # print the results
    print('Logistic Regression Accuracy: ', accuracy)
    # print the classification report
    print(classification_report(y_test, y_pred))

    # return the best parameters in a dictionary with the accuracy
    return {'best_lr_params': best_lr_params, 'accuracy': accuracy}


# function to tune support vector machine parameters in grid search with 5 fold cross validation and return the best parameters and test accuracy and classification report
def tune_svm_grid(X_train_full, y_train_full, X_test, y_test, kernel=['linear', 'poly', 'rbf', 'sigmoid'], c=[0.001, 0.01, 0.1, 1, 10, 100, 1000]):
    # tune support vector machine parameters in grid search with 5 fold cross validation
    # create the support vector machine model
    clf = SVC()
    # create the parameter grid with 3 values for each parameter
    param_grid = {
        'kernel': kernel,   # kernel type
        'C': c   # c is the penalty parameter of the error term
    }
    # create the grid search
    grid_search = GridSearchCV(estimator = clf, param_grid = param_grid, cv = 5, n_jobs = -1, verbose = 2)
    # fit the grid search
    grid_search.fit(X_train_full, y_train_full)
    # get the best parameters
    best_svm_params = grid_search.best_params_
    # print the best parameters
    print('Best Support Vector Machine Parameters: ', best_svm_params)
    # create the support vector machine model with the best parameters
    clf = SVC(kernel=best_svm_params['kernel'], C=best_svm_params['C'])
    # fit the model
    clf = clf.fit(X_train_full, y_train_full)
    # predict the test set
    y_pred = clf.predict(X_test)
    # calculate the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    # print the results
    print('Support Vector Machine Accuracy: ', accuracy)
    # print the classification report
    print(classification_report(y_test, y_pred))

    # return the best parameters in a dictionary with the accuracy
    return {'best_svm_params': best_svm_params, 'accuracy': accuracy}


# function to tune k-nearest neighbors parameters in grid search with 5 fold cross validation and return the best parameters and test accuracy and classification report
def tune_knn_grid(X_train_full, y_train_full, X_test, y_test, n_neighbors=[3, 5, 10], weights=['uniform', 'distance'], algorithm=['auto', 'ball_tree', 'kd_tree', 'brute']):
    # tune k-nearest neighbors parameters in grid search with 5 fold cross validation
    # create the k-nearest neighbors model
    clf = KNeighborsClassifier()
    # create the parameter grid with 3 values for each parameter
    param_grid = {
        'n_neighbors': n_neighbors,   # number of neighbors to use
        'weights': weights,   # weight function used in prediction
        'algorithm': algorithm,   # algorithm used to compute the nearest neighbors
    }
    # create the grid search
    grid_search = GridSearchCV(estimator = clf, param_grid = param_grid, cv = 5, n_jobs = -1, verbose = 2)
    # fit the grid search
    grid_search.fit(X_train_full, y_train_full)
    # get the best parameters
    best_knn_params = grid_search.best_params_
    # print the best parameters
    print('Best K-Nearest Neighbors Parameters: ', best_knn_params)
    # create the k-nearest neighbors model with the best parameters
    clf = KNeighborsClassifier(n_neighbors=best_knn_params['n_neighbors'], weights=best_knn_params['weights'], algorithm=best_knn_params['algorithm'])
    # fit the model
    clf = clf.fit(X_train_full, y_train_full)
    # predict the test set
    y_pred = clf.predict(X_test)
    # calculate the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    # print the results
    print('K-Nearest Neighbors Accuracy: ', accuracy)
    # print the classification report
    print(classification_report(y_test, y_pred))

    # return the best parameters in a dictionary with the accuracy
    return {'best_knn_params': best_knn_params, 'accuracy': accuracy}


# function to tune decision tree parameters in grid search with 5 fold cross validation and return the best parameters and test accuracy and classification report
def tune_dt_grid(X_train_full, y_train_full, X_test, y_test, criterion=['gini', 'entropy'], splitter=['best', 'random'], max_depth=[None, 1, 5, 10], min_samples_split=[2, 10], min_samples_leaf=[1, 2, 4]):
    # tune decision tree parameters in grid search with 5 fold cross validation
    # create the decision tree model
    clf = DecisionTreeClassifier()
    # create the parameter grid with 3 values for each parameter
    param_grid = {
        'criterion': criterion,   # function to measure the quality of a split
        'splitter': splitter,   # strategy used to choose the split at each node
        'max_depth': max_depth,   # maximum depth of the tree
        'min_samples_split': min_samples_split,   # minimum number of samples required to split an internal node
        'min_samples_leaf': min_samples_leaf   # minimum number of samples required to be at a leaf node
    }
    # create the grid search
    grid_search = GridSearchCV(estimator = clf, param_grid = param_grid, cv = 5, n_jobs = -1, verbose = 2)
    # fit the grid search
    grid_search.fit(X_train_full, y_train_full)
    # get the best parameters
    best_dt_params = grid_search.best_params_
    # print the best parameters
    print('Best Decision Tree Parameters: ', best_dt_params)
    # create the decision tree model with the best parameters
    clf = DecisionTreeClassifier(criterion=best_dt_params['criterion'], splitter=best_dt_params['splitter'], max_depth=best_dt_params['max_depth'], min_samples_split=best_dt_params['min_samples_split'], min_samples_leaf=best_dt_params['min_samples_leaf'])
    # fit the model
    clf = clf.fit(X_train_full, y_train_full)
    # predict the test set
    y_pred = clf.predict(X_test)
    # calculate the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    # print the results
    print('Decision Tree Accuracy: ', accuracy)
    # print the classification report
    print(classification_report(y_test, y_pred))

    # return the best parameters in a dictionary with the accuracy
    return {'best_dt_params': best_dt_params, 'accuracy': accuracy}


# function to tune neural network parameters in grid search with 5 fold cross validation and return the best parameters and test accuracy and classification report
def tune_nn_grid(X_train_full, y_train_full, X_test, y_test, hidden_layer_sizes=[(10,),(20,),(30,),(40,),(50,),(60,),(70,),(80,),(90,),(100,)], activation=['relu', 'logistic'], solver=['adam', 'sgd'], alpha=[0.0001, 0.05], learning_rate=['constant', 'adaptive']):
    # tune neural network parameters in grid search with 5 fold cross validation
    # create the neural network model
    clf = MLPClassifier()
    # create the parameter grid with 3 values for each parameter
    param_grid = {
        'hidden_layer_sizes': hidden_layer_sizes,   # number of neurons in each hidden layer
        'activation': activation,   # activation function for the hidden layer
        'solver': solver,   # the solver for weight optimization
        'alpha': alpha,   # L2 penalty (regularization term) parameter
        'learning_rate': learning_rate   # learning rate schedule for weight updates
    }
    # create the grid search
    grid_search = GridSearchCV(estimator = clf, param_grid = param_grid, cv = 5, n_jobs = -1, verbose = 2)
    # fit the grid search
    grid_search.fit(X_train_full, y_train_full)
    # get the best parameters
    best_nn_params = grid_search.best_params_
    # print the best parameters
    print('Best Neural Network Parameters: ', best_nn_params)
    # create the neural network model with the best parameters
    clf = MLPClassifier(hidden_layer_sizes=best_nn_params['hidden_layer_sizes'], activation=best_nn_params['activation'], solver=best_nn_params['solver'], alpha=best_nn_params['alpha'], learning_rate=best_nn_params['learning_rate'])
    # fit the model
    clf = clf.fit(X_train_full, y_train_full)
    # predict the test set
    y_pred = clf.predict(X_test)
    # calculate the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    # print the results
    print('Neural Network Accuracy: ', accuracy)
    # print the classification report
    print(classification_report(y_test, y_pred))

    # return the best parameters in a dictionary with the accuracy
    return {'best_nn_params': best_nn_params, 'accuracy': accuracy}


    # function to call all the tuning functions and store the results in a dictionary with an argument for which models to tune
def tune_models(X_train_full, y_train_full, X_test, y_test, models=['rf', 'gb', 'lr', 'svm', 'dt']):
    # create a dictionary to store the results
    results = {}
    # tune random forest parameters
    if 'rf' in models:
        results['rf'] = tune_rf_grid(X_train_full, y_train_full, X_test, y_test)
    # tune gradient boosting parameters
    if 'gb' in models:
        results['gb'] = tune_gb_grid(X_train_full, y_train_full, X_test, y_test)
    # tune logistic regression parameters
    if 'lr' in models:
        results['lr'] = tune_lr_grid(X_train_full, y_train_full, X_test, y_test)
    # tune support vector machine parameters
    if 'svm' in models:
        results['svm'] = tune_svm_grid(X_train_full, y_train_full, X_test, y_test)
    # tune k-nearest neighbors parameters
    if 'knn' in models:
        results['knn'] = tune_knn_grid(X_train_full, y_train_full, X_test, y_test)
    # tune decision tree parameters
    if 'dt' in models:
        results['dt'] = tune_dt_grid(X_train_full, y_train_full, X_test, y_test)
    # tune neural network parameters
    if 'nn' in models:
        results['nn'] = tune_nn_grid(X_train_full, y_train_full, X_test, y_test)
    # print the best overall model
    print('Best Overall Model: ', max(results, key=lambda key: results[key]['accuracy']))
    # return the results
    return results


# create a function that returns a dataframe with all rows for a given seasons list
def get_seasons(seasons, df):
    df_s = pd.DataFrame()
    for season in seasons:
        # using pd.concat to append the rows of the dataframe for each season
        df_s = pd.concat([df_s, df[df['season'] == season]])
    return df_s



# function that creates per team dataframes
def get_team_results(df):
    team_names = np.unique(df.team)
    team_results = {} 
    for name in team_names:
        team_results[name] = df.loc[df['team'] == name].sort_values(by=['day_number'], inplace=False)
    return team_names, team_results

# create a function that creates rolling weekly average stats for each team
# this could be useful for single season or 2 season predictions
def get_team_results_expon_rolling_avg(team_names, team_results, span_avg):
    column_indices_to_avg = np.where([x in team_results['Steelers'].select_dtypes(['number']).columns for x in team_results['Steelers'].columns])[0]
    team_results_avg = {}     
    for name in team_names:
        team_results_avg[name] = team_results[name].copy()
    for name in team_names:
        # add the columns from team_results[name].ewm(span=5, adjust=False).mean() to team_results_avg[name]
        era_col = team_results[name].ewm(span=span_avg, adjust=False).mean()
        # drop the columns that are in column_indices_to_avg list
        team_results_avg[name].drop(team_results_avg[name].columns[column_indices_to_avg], axis=1, inplace=True)
        team_results_avg[name] = pd.concat([team_results_avg[name], era_col], axis=1)
        team_results_avg[name].iloc[:,4:-2] = team_results_avg[name].iloc[:,4:-2].shift(1)
    return team_results_avg


# create a function that creates weighted weekly average stats for each team
def get_team_results_weighted(team_names, team_results, N=20, d1=200, d2=400):
    column_indices_to_avg = np.where([x in team_results['Steelers'].select_dtypes(['number']).columns for x in team_results['Steelers'].columns])
    column_indices_to_avg = column_indices_to_avg[0][:-1]
    team_results_weighted = {}     
    for name in team_names:
        team_results_weighted[name] = team_results[name].copy()
    for name in team_names:
        for idx in range(N,team_results[name].shape[0]):
            # Determine the days to the previous games, then weights for then weighted average.
            days = team_results[name].iloc[(idx-N):(idx-1)]['day_number'].to_numpy(dtype=float)
            days = days-days[-1]
            coeff = np.exp(days/d1)/np.sum(np.exp(days/d2))
            for col_idx in column_indices_to_avg:

                team_results_weighted[name].iloc[idx,col_idx] = np.dot(coeff,team_results[name].iloc[(idx-N):(idx-1),col_idx])
        team_results_weighted[name].iloc[:,4:-2] = team_results_weighted[name].iloc[:,4:-2].shift(1)
    return team_results_weighted


# Create a function that creates the data_home and data_away data frames
def get_data_home_away_avg_joined(data, team_results_avg, seasons=[2020,2021]):
    # Create initial data_home and data_away data frames as empty data frames with the correct column names
    data_home = pd.DataFrame(columns = team_results_avg['Eagles'].columns) 
    data_away = pd.DataFrame(columns = team_results_avg['Eagles'].columns) 
    # create data_new from data that contains all seasons in a list
    data_new = data[data['season'].isin(seasons)]

    for idx in range(data_new.shape[0]):
        # get the date for this row
        date = data_new.iloc[idx].date
        # get the home and away team for this row
        home_team = data_new.iloc[idx].home
        away_team = data_new.iloc[idx].away
        home_data = team_results_avg[home_team].loc[team_results_avg[home_team]['date'] == date]
        away_data = team_results_avg[away_team].loc[team_results_avg[away_team]['date'] == date]
        # concatonate the rows to each data_home, data_away
        data_home = pd.concat([data_home,home_data])
        data_away = pd.concat([data_away,away_data])
    
    # append '_homeAvg' and '_awayAvg' to the home and away datframes
    for name in data_home.columns:
        data_home = data_home.rename({name: name+'_homeAvg'}, axis=1)
        data_away = data_away.rename({name: name+'_awayAvg'}, axis=1)

    # drop un-needed columns the home and away datframes
    data_home = data_home.drop(['date_homeAvg', 'team_homeAvg', 'opponent_homeAvg'], axis=1)
    data_away = data_away.drop(['date_awayAvg', 'team_awayAvg', 'opponent_awayAvg'], axis=1)
    
    df = pd.concat([data_new[['date','home','away','winner','score_home','score_away','score_diff']].reset_index(drop=True),data_home.reset_index(drop=True),data_away.reset_index(drop=True)], axis=1, sort=False)
    return df


def get_next_week_x_train(previous_games, next_homes, next_aways):
    home_dict = {}
    away_dict = {}
    important_away = {}
    important_home = {}

    for team in next_homes:
        last_home = previous_games[previous_games.home == team].iloc[-1]
        last_away = previous_games[previous_games.away == team].iloc[-1]
        if last_home.date > last_away.date:
            last_game = last_home
        else:
            last_game = last_away
        if last_game.home == team:
            unneeded_previous_games = last_game.filter(regex="away")
        else:
            unneeded_previous_games = last_game.filter(regex="home")
        # # drop the unneeded columns from team_results_weighted
        important_home[team] = last_game.drop(unneeded_previous_games.keys())
        data_home = pd.DataFrame(columns=important_home[team].keys())
        data_home = data_home.append(important_home[team])
        try:
            data_home.drop(['date', 'away', 'winner', 'score_away', 'score_diff'],axis=1,inplace=True)
        except:
            data_home.drop(['date', 'home', 'winner', 'score_home', 'score_diff'],axis=1,inplace=True)
        for name in data_home.keys():
            # take the last 8 digits away from each column name
            data_home = data_home.rename(columns={name: name[:-8]})
        for name in data_home.keys():
            data_home = data_home.rename(columns={name: name+"_homeAvg"})
        home_dict[team] = data_home

    for team in next_aways:
        last_home = previous_games[previous_games.home == team].iloc[-1]
        last_away = previous_games[previous_games.away == team].iloc[-1]
        if last_home.date > last_away.date:
            last_game = last_home
        else:
            last_game = last_away
        if last_game.home == team:
            unneeded_previous_games = last_game.filter(regex="away")
        else:
            unneeded_previous_games = last_game.filter(regex="home")
        # # drop the unneeded columns from team_results_weighted
        important_away[team] = last_game.drop(unneeded_previous_games.keys())

        data_away = pd.DataFrame(columns=important_away[team].keys())
        data_away = data_away.append(important_away[team])
        try:
            data_away.drop(['date', 'away', 'winner', 'score_away', 'score_diff'],axis=1,inplace=True)
        except:
            data_away.drop(['date', 'home', 'winner', 'score_home', 'score_diff'],axis=1,inplace=True)
        for name in data_away.keys():
            data_away = data_away.rename(columns={name: name[0:-8]})
        for name in data_away.keys():
            data_away = data_away.rename(columns={name: name+"_awayAvg"})
        away_dict[team] = data_away
    for i in range(len(next_homes)):
        home_team = home_dict[next_homes.iloc[i]]
        away_team = away_dict[next_aways.iloc[i]]
        home_team.loc[:, away_team.columns] = away_team.to_numpy()
        try:
            home_team.drop('home_homeAvg', axis=1,inplace=True)
        except:
            pass
        try:
            home_team.drop('away_awayAvg', axis=1,inplace=True)
        except:
            pass
        try:
            home_team.drop('away_homeAvg', axis=1,inplace=True)
        except:
            pass
        try:
            home_team.drop('home_awayAvg', axis=1,inplace=True)
        except:
            pass

        if i == 0:
            x_train = pd.DataFrame(columns=[list(home_team.columns)], index=range(len(next_homes)))
        x_train.loc[i] = list(home_team.iloc[0])
    return x_train


def week_num(date):
    # turn sting date into datetime
    end_of_week_1 = datetime.datetime(2021, 9, 13)
    if date <= end_of_week_1:
        return 1
    elif date <= end_of_week_1 + datetime.timedelta(days=7):
        return 2
    elif date <= end_of_week_1 + datetime.timedelta(days=14):
        return 3
    elif date <= end_of_week_1 + datetime.timedelta(days=21):
        return 4
    elif date <= end_of_week_1 + datetime.timedelta(days=28):
        return 5
    elif date <= end_of_week_1 + datetime.timedelta(days=35):
        return 6
    elif date <= end_of_week_1 + datetime.timedelta(days=42):
        return 7
    elif date <= end_of_week_1 + datetime.timedelta(days=49):
        return 8
    elif date <= end_of_week_1 + datetime.timedelta(days=56):
        return 9
    elif date <= end_of_week_1 + datetime.timedelta(days=63):
        return 10
    elif date <= end_of_week_1 + datetime.timedelta(days=70):
        return 11
    elif date <= end_of_week_1 + datetime.timedelta(days=77):
        return 12
    elif date <= end_of_week_1 + datetime.timedelta(days=84):
        return 13
    elif date <= end_of_week_1 + datetime.timedelta(days=91):
        return 14
    elif date <= end_of_week_1 + datetime.timedelta(days=98):
        return 15
    elif date <= end_of_week_1 + datetime.timedelta(days=105):
        return 16
    elif date <= end_of_week_1 + datetime.timedelta(days=112):
        return 17
    elif date <= end_of_week_1 + datetime.timedelta(days=119):
        return 18
    else:
        return 19