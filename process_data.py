"""
This file condenses the data so that it only contains points that:
- are within the Monarch butterfly's summer breeding ground range
- are within the date range that Monarch's breed in the summer
- do not have an empty value for the average temperature
"""
import pandas as pd

def process_hyp_1():
    df = pd.read_csv('data/GlobalLandTemperaturesByCity.csv')
    df['dt']= pd.to_datetime(df['dt'])
    #filter to just rows with an average temp. value
    df = df[df['AverageTemperature'].notna()]
    #drop the latitudes we don't need
    df.drop(df.index[df['Latitude'].str.contains('S')], inplace = True)

    #make sure data is contained within Monarch's summer breeding grounds
    df.drop(df.index[df['Latitude'].str.replace('N', '').astype(float) < 40], inplace = True)
    df.drop(df.index[df['Latitude'].str.replace('N', '').astype(float) > 50], inplace = True)

    #drop the longitude we don't need
    df.drop(df.index[df['Longitude'].str.contains('E')], inplace = True)
    
    #make sure data is contained within Monarch's summer breeding grounds
    df.drop(df.index[df['Longitude'].str.replace('W', '').astype(float) < 75], inplace = True)
    df.drop(df.index[df['Longitude'].str.replace('W', '').astype(float) > 95], inplace = True)

    #this line is in case we decide to focus on one city in particular
    #df.drop(df.index[df['City'].str.contains('Peoria') == False], inplace = True)

    #drop months we do not need

    #this line is if we decide to do the same day- pick the month we wish to see
    #df.drop(df.index[df['dt'].dt.month != 7], inplace = True)

    #these lines are for if we decide to do every day within the summer breeding months
    df.drop(df.index[df['dt'].dt.month < 5], inplace = True)
    df.drop(df.index[df['dt'].dt.month > 8], inplace = True)

    #print(df.head(5))
    df.to_csv(path_or_buf= 'data/ProcessedData_Hyp1.csv')

def process_hyp_2():
    df = pd.read_csv('data/GlobalLandTemperaturesByCity.csv')
    df['dt']= pd.to_datetime(df['dt'])
    #filter to just rows with an average temp. value
    df = df[df['AverageTemperature'].notna()]
    #drop the latitudes we don't need
    df.drop(df.index[df['Latitude'].str.contains('S')], inplace = True)

    #make sure data is contained within Monarch's summer breeding grounds
    df.drop(df.index[df['Latitude'].str.replace('N', '').astype(float) < 15], inplace = True)
    df.drop(df.index[df['Latitude'].str.replace('N', '').astype(float) > 30], inplace = True)

    #drop the longitude we don't need
    df.drop(df.index[df['Longitude'].str.contains('E')], inplace = True)
    
    #make sure data is contained within Monarch's summer breeding grounds
    df.drop(df.index[df['Longitude'].str.replace('W', '').astype(float) < 90], inplace = True)
    df.drop(df.index[df['Longitude'].str.replace('W', '').astype(float) > 110], inplace = True)

    #this line is in case we decide to focus on one city in particular
    #df.drop(df.index[df['City'].str.contains('Peoria') == False], inplace = True)

    #drop months we do not need

    #this line is if we decide to do the same day- pick the month we wish to see
    #df.drop(df.index[df['dt'].dt.month != 7], inplace = True)

    #these lines are for if we decide to do every day within the summer breeding months
    df.drop(df.index[df['dt'].dt.month == 3], inplace = True)
    df.drop(df.index[df['dt'].dt.month == 4], inplace = True)
    df.drop(df.index[df['dt'].dt.month == 5], inplace = True)
    df.drop(df.index[df['dt'].dt.month == 6], inplace = True)
    df.drop(df.index[df['dt'].dt.month == 7], inplace = True)
    df.drop(df.index[df['dt'].dt.month == 8], inplace = True)
    df.drop(df.index[df['dt'].dt.month == 9], inplace = True)
    #print(df.head(5))
    df.to_csv(path_or_buf= 'data/ProcessedData_Hyp2.csv')

def main():
    #process_hyp_1()
    process_hyp_2()

if __name__ == "__main__":
    main() 