from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from numpy import mean
from numpy import std
from numpy import absolute
from sklearn.linear_model import Ridge
import datetime

def remove_outliers(df):
    mean, std = np.mean(df)['AverageTemperature'], np.std(df)['AverageTemperature']
    limit = std * 1.5
    lower_lim, upper_lim = mean - limit, mean+limit

    rows_dropped = 0
    for index, row in df.iterrows():
        if row['AverageTemperature'] < lower_lim or row['AverageTemperature'] > upper_lim:
            rows_dropped += 1
            df.drop(index)
    print("Number of rows dropped:", rows_dropped)
    return df

def find_best_alpha(model, X, y):
    alpha = [0.0001, 0.001,0.01, 0.1, 1, 10]
    scores_graph = []
    for i in alpha:
        if model == "r":
            regressor = Ridge(alpha = i, normalize=True)
        else:
            regressor = Lasso(alpha = i, normalize=True)

        cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
        # evaluate model
        scores = cross_val_score(regressor, X, y, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)
        scores = np.sqrt(np.abs(scores))
        scores_graph.append(mean(scores))

    plt.clf()
    df = pd.DataFrame({'Alpha Values': alpha, 'RMSE': scores_graph})
    plt.plot('Alpha Values', 'RMSE', data=df)
    plt.xlabel("Alpha values")
    plt.ylabel("RMSE")
    plt.title("RMSE vs Regression Alpha Values")
    plt.show()
    plt.clf()

    print("\n\n\n\n\nBest Alpha: ", alpha[scores_graph.index(min(scores_graph))],"\n\n\n\n\n")
    return alpha[scores_graph.index(min(scores_graph))]

#Andrew's Linear Regression Method
def lasso_linear_regression(df, city_name, month, temp):
    df.drop(df.index[df['City'].str.contains(city_name) == False], inplace = True)
    df.drop(df.index[df['dt'].dt.month != month], inplace = True)
    # df = remove_outliers(df)
    SEED = 42
    X = df['dt'].dt.year.values.reshape(-1, 1)
    y = df['AverageTemperature'].values.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state= SEED)


    ## CHANGE BETWEEN RIDGE, LASSO, AND LINEAR REGRESSION HERE
    #regressor = LinearRegression(normalize=True)

    ## If running linear comment the line below out and set best_alpha = 0
    best_alpha = find_best_alpha("l", X, y)
    # best_alpha = 0

    regressor = Lasso(alpha = best_alpha, normalize=True)
    #regressor = Ridge(alpha = best_alpha, normalize=True)

    #Evaluate Model
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    # evaluate model
    scores = cross_val_score(regressor, X, y, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)
    scores = np.sqrt(np.abs(scores))
    print('\n\n\nMean RMSE: ', mean(scores))
    print('Best Alpha: ', best_alpha, '\n\n\n')

    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)

    df_predictions = pd.DataFrame({'Actual': y_test.squeeze(), 'Predicted': y_pred.squeeze()})
    print("R2 Score: " + str(r2_score(y_test, y_pred)))
    high_temp_year = (temp - regressor.intercept_)/regressor.coef_
    print("Average temp of " + str(temp) + " will be reached in: " + str(high_temp_year))

    plt.scatter(X_train, y_train, color = "Black")
    plt.plot(X_test, y_pred, color = "Orange")
    datetime_object = datetime.datetime.strptime(str(month), "%m")
    month_name = datetime_object.strftime("%b")
    plt.xlabel("Year")
    plt.ylabel("Temperature in Celsius on " + str(month_name) + " 1st")
    plt.title("Temperature (C) in " + str(city_name) + " vs Time\nRidge Regression")
    plt.show()

#Allison's Linear Regression Method
def linear_regression(df, city_name, month, temp):
    df.drop(df.index[df['City'].str.contains(city_name) == False], inplace = True)
    df.drop(df.index[df['dt'].dt.month != month], inplace = True)
    SEED = 42
    X = df['dt'].dt.year.values.reshape(-1, 1)
    y = df['AverageTemperature'].values.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state= SEED)

    regressor = LinearRegression()

    regressor.fit(X_train, y_train)

    print("Regressor line intercept: " + str(regressor.intercept_))
    print("Regressor coefficient: " + str(regressor.coef_))

    score = regressor.predict([[temp]])
    print("Score: " + str(score))

    y_pred = regressor.predict(X_test)
    df_predictions = pd.DataFrame({'Actual': y_test.squeeze(), 'Predicted': y_pred.squeeze()})
    #print(df_predictions)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    print(f'Mean absolute error: {mae:.2f}')
    print(f'Mean squared error: {mse:.2f}')
    print(f'Root mean squared error: {rmse:.2f}')


    #high_temp = (temp)*(regressor.coef_) + regressor.intercept_
    high_temp_year = (temp - regressor.intercept_)/regressor.coef_
    print("Average temp of " + str(temp) + " will be reached in: " + str(high_temp_year))

    plt.scatter(X_train, y_train, color = "Black")
    plt.plot(X_test, y_pred, color = "Orange")
    datetime_object = datetime.datetime.strptime(str(month), "%m")
    month_name = datetime_object.strftime("%b")

    plt.xlabel("Year")
    plt.ylabel("Temperature in Celsius on " + str(month_name) + " 1st")
    plt.title("Temperature (C) in " + str(city_name) + " vs Time\nLinear Regression")
    plt.show()