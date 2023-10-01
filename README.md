# engr-ALDA-Fall2022-P7

process_data.py processes the GlobalLandTemperaturesByCity.csv to create two data sets: one including only the summer breeding grounds of the monarch butterfly as well as the breeding months of the butterfly and the other containing only the winter hibernation grounds and months of the monarch butterfly. This class also removes all empty/ incomplete data. The processed datasets that we will use for our project are saved as ProcessedData_Hyp1.csv and ProcessedData_Hyp2.csv within the data folder.

visualize_data.py plots the data. There is currently only one method in this class which plots the data in the center of the Monarch's summer breeding range during their peak breeding month (July). 

run_regressions.py is where we put the method(s) to run the linear regression model, lasso regression model, and ridge regression model on our datasets in order to tell if climate change poses an immediate risk to the monarch butterfly population.

run_polynomial_regression.py is the class where the method to run a polynomial regression model was created. We tried this to predict an exponential/ non-linear rate of temperature change to account for the exponential growth/ usage of fossil fuels. We did not end up using the results from this class.

main.py is the main class which calls the methods in the other classes
