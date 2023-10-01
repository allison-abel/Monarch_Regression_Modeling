from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def poly_regression(df, city_name):
    df.drop(df.index[df['City'].str.contains(city_name) == False], inplace = True)
    df.drop(df.index[df['dt'].dt.month != 7], inplace = True)

    SEED = 42
    X = df['dt'].dt.year.values.reshape(-1, 1)
    y = df['AverageTemperature'].values.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state= SEED)

    poly = PolynomialFeatures(degree=2, include_bias=False)

    poly_features = poly.fit_transform(X_train)
    poly_features_test = poly.fit_transform(X_test)

    poly_regressor = LinearRegression()

    poly_regressor.fit(poly_features, y_train)
    


    # Print intercept and coefficient of the regressor
    print("Poly regressor line intercept: " + str(poly_regressor.intercept_))
    print("Poly regressor coefficient: " + str(poly_regressor.coef_))

    # Get and print the score of the poly regressor
    score = poly_regressor.predict([[36, 36]])
    print("Poly Score: " + str(score))

    y_pred = poly_regressor.predict(poly_features_test)

    print("R2 Score: " + str(r2_score(y_test, y_pred)))
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    # Print the MAE, MSE, and RMSE
    print(f'Mean absolute error: {mae:.2f}')
    print(f'Mean squared error: {mse:.2f}')
    print(f'Root mean squared error: {rmse:.2f}')

    # Get and print the year that the average temperature will be too high
    high_temp_year = (36 - poly_regressor.intercept_) / poly_regressor.coef_
    print("Average temp of 36 will be reached in: " + str(high_temp_year))

    # Plot the data
    plt.scatter(X_train, y_train, color = "Black")
    plt.plot(sorted(X_test), sorted(y_pred), color = "Orange")
    plt.xlabel("Year")
    plt.ylabel("Temperature in Celsius")
    plt.title("Temperature (C) in " + str(city_name) + " vs Time")
    plt.show()
