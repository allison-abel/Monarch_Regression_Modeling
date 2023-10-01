import pandas as pd
from visualize_data import visualize_southern_part_of_breeding_ground, visualize_middle_of_breeding_ground, visualize_northern_part_of_breeding_ground
from run_regressions import linear_regression, lasso_linear_regression
from run_polynomial_regression import poly_regression

def hypothesis_1():
    df_hyp1 = pd.read_csv('data/ProcessedData_Hyp1.csv')
    df_hyp1['dt']= pd.to_datetime(df_hyp1['dt'])
    city_name = "Ottawa"
    month = 8
    temp = 36
    linear_regression(df_hyp1, city_name, month, temp)
    lasso_linear_regression(df_hyp1, city_name, month, temp)
    #print()
    #poly_regression(df_hyp1, city_name)
    
    #visualizes Peoria in July (southern area of the breeding range during peak breeding month)
    #df_peoria = visualize_southern_part_of_breeding_ground(df_hyp1)

    #visualizes Minneapolis in July (center of the breeding range during peak breeding month)
    #df_minneapolis = visualize_middle_of_breeding_ground(df_hyp1)

    #visualizes Ottowa in July (center of the breeding range during peak breeding month)
    #df_ottawa = visualize_northern_part_of_breeding_ground(df_hyp1)

def hypothesis_2():
    df_hyp2 = pd.read_csv('data/ProcessedData_Hyp2.csv')
    df_hyp2['dt']= pd.to_datetime(df_hyp2['dt'])
    city_name = "Toluca"
    month = 12
    #15 celsius is too hot for butterflies in winter
    temp = 15 
    linear_regression(df_hyp2, city_name, month, temp)
    lasso_linear_regression(df_hyp2, city_name, month, temp)

def main():
    #hypothesis_1()

    hypothesis_2()

if __name__ == "__main__":
    main() 