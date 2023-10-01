import matplotlib.pyplot as plt

def visualize_southern_part_of_breeding_ground(df):
    df_copy = df.copy()
    df_copy.drop(df_copy.index[df_copy['City'].str.contains('Peoria') == False], inplace = True)
    df_copy.drop(df_copy.index[df_copy['dt'].dt.month != 7], inplace = True)
    plt.scatter(df_copy['dt'].dt.year, df_copy['AverageTemperature'], color = 'orange')
    plt.xlabel("Year")
    plt.ylabel("Temperature in Celsius")
    plt.title("Temperature (C) in Peoria vs Time")
    plt.show()
    return df_copy

def visualize_middle_of_breeding_ground(df):
    df_copy = df.copy()
    df_copy.drop(df_copy.index[df_copy['City'].str.contains('Minneapolis') == False], inplace = True)
    df_copy.drop(df_copy.index[df_copy['dt'].dt.month != 7], inplace = True)
    plt.scatter(df_copy['dt'].dt.year, df_copy['AverageTemperature'], color = 'orange')
    plt.xlabel("Year")
    plt.ylabel("Temperature in Celsius")
    plt.title("Temperature (C) in Minneapolis vs Time")
    plt.show()
    return df_copy

def visualize_northern_part_of_breeding_ground(df):
    df_copy = df.copy()
    df_copy.drop(df_copy.index[df_copy['City'].str.contains('Ottawa') == False], inplace = True)
    df_copy.drop(df_copy.index[df_copy['dt'].dt.month != 7], inplace = True)
    plt.scatter(df_copy['dt'].dt.year, df_copy['AverageTemperature'], color = 'orange')
    plt.xlabel("Year")
    plt.ylabel("Temperature in Celsius")
    plt.title("Temperature (C) in Ottawa vs Time")
    plt.show()
    return df_copy