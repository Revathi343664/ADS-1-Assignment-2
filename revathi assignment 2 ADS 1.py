# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 17:19:39 2023

@author: Revathi Nanubala
"""

# importing numpy as np
import numpy as np
# importing pandas as pd
import pandas as pd
# importing matplotlib.pyplot as plt
import matplotlib.pyplot as plt
# importing seaborn as sns
import seaborn as sns
# Reads a comma-separated values (csv) file into DataFrame.
df = pd.read_csv(
    'C:/Users/Revathi Nanubala/Downloads/worldbankdata.csv.csv', index_col='Country Name')
# To display maximum columns present in the DataFrame
pd.set_option('display.max_columns', 1500)
# Print the dataframe
print(df.head(50))
# Renaming the column names
df.rename(columns={"2010 [YR2010]": "2010", '2011 [YR2011]': '2011', '2012 [YR2012]': '2012', '2013 [YR2013]': '2013', '2014 [YR2014]':'2014'}, inplace=True)

# Renaming the column names
df.rename(columns={'2015 [YR2015]':'2015', '2016 [YR2016]':'2016', '2017 [YR2017]':'2017', '2018 [YR2018]':'2018', '2019 [YR2019]':'2019', '2020 [YR2020]':'2020'}, inplace=True)

# Returns the number of missing values in the DataFrame
missing_values_count = df.isnull().sum()

# Print the missing values count
print(missing_values_count)
# Dropping null values in the DataFrame
df.dropna(inplace=True)
# Coverting the Object type data into float type using apply() function
df['2015'] = df['2015'].apply(lambda x: float(x.split()[0].replace('..', '0')))
df['2016'] = df['2016'].apply(lambda x: float(x.split()[0].replace('..', '0')))
df['2017'] = df['2017'].apply(lambda x: float(x.split()[0].replace('..', '0')))
df['2018'] = df['2018'].apply(lambda x: float(x.split()[0].replace('..', '0')))
df['2019'] = df['2019'].apply(lambda x: float(x.split()[0].replace('..', '0')))
df['2020'] = df['2020'].apply(lambda x: float(x.split()[0].replace('..', '0')))
# Returns a Series with the data type of each column
data_types_series = df.dtypes

# Print the data types series
print(data_types_series)
# Creating a Sub Dataframe with selected rows and columns from the DataFrame(df) using Groupby() function
dt=df.groupby('Series Code')
agricultural_land= dt.get_group('AG.LND.AGRI.ZS')
print(agricultural_land)
#provides a discriptive summary of the given dataset
print(agricultural_land.describe())

def data(ax):
    '''
    Add the values in "ax" 

    Here we are labeling the data to form a bar graph
    '''
    ax.set_xlabel("Country", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    plt.show()

# In variable rec we are storing the data Frame
rec = agricultural_land[['2020']].plot(kind='bar', title="Agricultural land(%)", figsize=(4, 3), legend=True, fontsize=9, color='maroon')

# Here we are calling the data function which passes the rec to the above function
data(rec)

# Calculate the correlation matrix
correlation_matrix = agricultural_land.corr()

# Create a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap - Agricultural Land")
plt.show()
# Creating a Sub DataFrame with selected rows and columns from the DataFrame(df) using Groupby() function
dt = df.groupby('Series Code')
c02_emmision = dt.get_group('EN.ATM.CO2E.KT')

# Output the first 10 rows of the c02_emmision DataFrame
print(c02_emmision.head(10))

# Calculate skewness of the individual values in the c02_emmision DataFrame Skewness = 0 when the distribution is normal
'''Skewness > 0 or positive when more weight is on the left side of the distribution.
Skewness < 0 or negative when more weight is on the right side of the distribution.'''

skewness = c02_emmision.skew(axis=0, skipna=True)

# Output the skewness values for each column
print(skewness.head())

# Calculate kurtosis of the individual values in the c02_emmision DataFrame
kurtosis = c02_emmision.kurtosis(axis=0, skipna=True)

# Output the kurtosis values for each column
print(kurtosis.head())


def data(ax):
    '''
    Add the values in "ax" 

    Here we are labeling the data to form a bar graph
    '''
    ax.set_xlabel("Country", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    plt.show()

# In variable rec we are storing the data Frame
rec = c02_emmision[['2010', '2012', '2014', '2016', '2018']].plot(kind='bar', title="CO2 Emission (kt)", figsize=(8, 8), legend=True, fontsize=12, color=['blue', 'green', 'orange', 'red', 'purple'])

# Here we are calling the data function which passes the rec to the above function
data(rec)
# Creating a Sub DataFrame with selected rows and columns from the DataFrame(df) using Groupby() function
dt = df.groupby('Series Code')
electric_usage = dt.get_group('EG.USE.ELEC.KH.PC')

# Output the first 10 rows of the electric_usage DataFrame
print(electric_usage.head(10))
# Drop the specified columns from the electric_usage DataFrame
electric_usage.drop(["2015", "2016", "2017", "2018", "2019", "2020"], axis=1, inplace=True)

# Assuming electric_usage is your DataFrame
electric_usage['Country Code'] = electric_usage['Country Code'].replace(
    ['IND', 'USA', 'CHN', 'QAT', 'ARE', 'DEU', 'JPN'],
    ['India', 'United States', 'China', 'Qatar', 'United Arab Emirates', 'Germany', 'Japan']
)

# Assuming electric_usage is your DataFrame
Trans = electric_usage.transpose()

# Modify the dataframe by changing the row elements into column elements
# and the column elements into row elements
Trans.rename(columns=Trans.iloc[2], inplace=True)

# Create a new DataFrame with the modified array
electric_transpose = Trans.iloc[2:]

# Display the resulting DataFrame
print(electric_transpose)

# Assuming electric_transpose is your DataFrame
electric_transpose.fillna(0, inplace=True)

# Display the DataFrame after removing null values
print(electric_transpose)
# Assuming electric_transpose is your DataFrame
print(electric_transpose.head(10))
# Assuming electric_transpose is your DataFrame
electric_transpose.dtypes
# Assuming electric_transpose is your DataFrame
electric_transpose['India'] = pd.to_numeric(electric_transpose['India'], errors='coerce')
electric_transpose['United States'] = pd.to_numeric(electric_transpose['United States'], errors='coerce')
electric_transpose['China'] = pd.to_numeric(electric_transpose['China'], errors='coerce')
electric_transpose['Qatar'] = pd.to_numeric(electric_transpose['Qatar'], errors='coerce')
electric_transpose['United Arab Emirates'] = pd.to_numeric(electric_transpose['United Arab Emirates'], errors='coerce')
electric_transpose['Germany'] = pd.to_numeric(electric_transpose['Germany'], errors='coerce')
electric_transpose['Japan'] = pd.to_numeric(electric_transpose['Japan'], errors='coerce')

# Display the DataFrame after conversion
print(electric_transpose)

def data(ax):
    '''
    Add the values in "ax" 

    Here we are labeling the data to form a bar graph
    '''
    ax.set_xlabel("Country Name", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)

# Assuming electric_transpose is your DataFrame
rec = electric_transpose[['India', 'China', 'United States', 'Qatar', 'United Arab Emirates', 'Germany', 'Japan']].plot(
    kind='line', title="Electric power consumption (kWh per capita)", figsize=(8, 8), legend=True, fontsize=12)

# Call the data function passing the rec to the above function
data(rec)

# Display the plot
plt.show()

# Creating a Sub Dataframe with selected rows and columns using Groupby() function
dt = df.groupby('Series Code')
renewable_energy = dt.get_group('EG.FEC.RNEW.ZS')

# Displaying the first 10 rows of the new dataframe
print(renewable_energy.head(10))
# Mapping Country Code to Country Names
country_mapping = {'IND': 'India', 'USA': 'United States', 'CHN': 'China', 'QAT': 'Qatar',
                   'ARE': 'United Arab Emirates', 'DEU': 'Germany', 'JPN': 'Japan'}

# Replacing 'Country Code' values with corresponding country names
renewable_energy['Country Code'] = renewable_energy['Country Code'].replace(country_mapping)

# Displaying the modified dataframe
print(renewable_energy.head(10))
renewable_energy.drop(["2020"], axis=1, inplace=True)

# Displaying the modified dataframe
print(renewable_energy.head(10))

# Transposing the dataframe
Trans = renewable_energy.transpose()

# Modifying the column names using the values from the third row
Trans.rename(columns=Trans.iloc[2], inplace=True)

# Selecting rows from the third row onwards
renewable_energy_transpose = Trans.iloc[2:]

# Displaying the modified transposed dataframe
print(renewable_energy_transpose.head(10))

# Removing null values by filling them with 0
renewable_energy_transpose.fillna(0, inplace=True)

# Displaying the modified dataframe

print(renewable_energy_transpose.head(20))
column_data_types = renewable_energy_transpose.dtypes

# Displaying the Series with data types
print(column_data_types)

# Converting Object type data to float type using pd.to_numeric
renewable_energy_transpose['India'] = pd.to_numeric(renewable_energy_transpose['India'], errors='coerce')
renewable_energy_transpose['United States'] = pd.to_numeric(renewable_energy_transpose['United States'], errors='coerce')
renewable_energy_transpose['China'] = pd.to_numeric(renewable_energy_transpose['China'], errors='coerce')
renewable_energy_transpose['Qatar'] = pd.to_numeric(renewable_energy_transpose['Qatar'], errors='coerce')
renewable_energy_transpose['United Arab Emirates'] = pd.to_numeric(renewable_energy_transpose['United Arab Emirates'], errors='coerce')
renewable_energy_transpose['Germany'] = pd.to_numeric(renewable_energy_transpose['Germany'], errors='coerce')
renewable_energy_transpose['Japan'] = pd.to_numeric(renewable_energy_transpose['Japan'], errors='coerce')

# Displaying the modified dataframe
print(renewable_energy_transpose.head(10))
def data(ax):
    '''
    Add the values in "ax"
    Here we are labeling the data to form a bar graph
    '''
    ax.set_xlabel("Country Name", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)


# Selecting specific columns for the plot
columns_to_plot = ['India', 'China', 'United States', 'Qatar', 'United Arab Emirates', 'Germany', 'Japan']

# Plotting the selected columns as a line graph
rec =renewable_energy_transpose[columns_to_plot].plot(
    kind='line', title="Renewable energy consumption", figsize=(8, 8), legend=True, fontsize=12
)

# Calling the data function to add labels
data(rec)

# Displaying the plot
plt.show()

print(df.head(100))
# Creating a Sub Dataframe with selected rows and columns using Groupby() function
dt = df.groupby('Series Code')
renewable_energy = dt.get_group('SP.POP.TOTL')

# Displaying the first 10 rows of the new dataframe
print(renewable_energy.head(10))
renewable_energy['Country Code'] = renewable_energy['Country Code'].replace(['IND', 'USA', 'CHN', 'QAT', 'ARE', 'DEU','JPN'], ['India', 'United States', 'China', 'Qatar','United Arab Emirates', 'Germany', 'Japan' ])
# Dropping the '2020' column from the dataframe
renewable_energy.drop(["2020"], axis=1, inplace=True)
# Transposing the dataframe
Trans = renewable_energy.transpose()

# Modifying the column names using the values from the third row
Trans.rename(columns=Trans.iloc[2], inplace=True)

# Selecting rows from the third row onwards
electric_transpose = Trans.iloc[2:]

# Displaying the modified transposed dataframe
print(electric_transpose.head(10))
# Removing null values by filling them with 0
electric_transpose.fillna(0, inplace=True)

# Displaying the modified dataframe
print(electric_transpose.head(10))

# Converting Object type data to float type using pd.to_numeric
electric_transpose['India'] = pd.to_numeric(electric_transpose['India'], errors='coerce')
electric_transpose['United States'] = pd.to_numeric(electric_transpose['United States'], errors='coerce')
electric_transpose['China'] = pd.to_numeric(electric_transpose['China'], errors='coerce')
electric_transpose['Qatar'] = pd.to_numeric(electric_transpose['Qatar'], errors='coerce')
electric_transpose['United Arab Emirates'] = pd.to_numeric(electric_transpose['United Arab Emirates'], errors='coerce')
electric_transpose['Germany'] = pd.to_numeric(electric_transpose['Germany'], errors='coerce')
electric_transpose['Japan'] = pd.to_numeric(electric_transpose['Japan'], errors='coerce')

# Displaying the modified dataframe
print(electric_transpose.head(10))


# Assuming 'electric_transpose' is your DataFrame
correlation_matrix = electric_transpose.corr()

# Plotting the correlation matrix as a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix Renewable Energy consumption')
plt.show()


def data(ax):
    '''
    Add the values in "ax"
    Here we are labeling the data to form a line graph
    '''
    ax.set_xlabel("Country Name", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)

# Plotting the specified columns as a line graph
rec = electric_transpose[['India', 'China', 'United States', 'Qatar', 'United Arab Emirates', 'Germany', 'Japan']] \
    .plot(kind='line', title="Total population", figsize=(8, 8), legend=True, fontsize=12)

# Calling the data function to add labels
data(rec)

# Displaying the plot
plt.show()

# Creating a Sub Dataframe with selected rows and columns using Groupby() function
dt = df.groupby('Series Code')
industry = dt.get_group('NV.IND.TOTL.KD.ZG')

# Displaying the first 10 rows of the new dataframe
print(industry.head(10))

# Replacing 'Country Code' values with corresponding country names
industry['Country Code'] = industry['Country Code'].replace(['IND', 'USA', 'CHN', 'QAT', 'ARE', 'DEU', 'JPN'],
                                                            ['India', 'United States', 'China', 'Qatar',
                                                             'United Arab Emirates', 'Germany', 'Japan'])

# Dropping the '2020' column from the dataframe
industry.drop(["2020"], axis=1, inplace=True)

# Transposing the dataframe
Trans = industry.transpose()

# Modifying the column names using the values from the third row
Trans.rename(columns=Trans.iloc[2], inplace=True)

# Selecting rows from the third row onwards
industry_transpose = Trans.iloc[2:]

# Displaying the modified transposed dataframe
print(industry_transpose.head(10))

# Removing null values by filling them with 0
industry_transpose.fillna(0, inplace=True)

# Displaying the modified dataframe
print(industry_transpose.head(10))

# Converting Object type data to float type using pd.to_numeric
industry_transpose['India'] = pd.to_numeric(industry_transpose['India'], errors='coerce')
industry_transpose['United States'] = pd.to_numeric(industry_transpose['United States'], errors='coerce')
industry_transpose['China'] = pd.to_numeric(industry_transpose['China'], errors='coerce')
industry_transpose['Qatar'] = pd.to_numeric(industry_transpose['Qatar'], errors='coerce')
industry_transpose['United Arab Emirates'] = pd.to_numeric(industry_transpose['United Arab Emirates'], errors='coerce')
industry_transpose['Germany'] = pd.to_numeric(industry_transpose['Germany'], errors='coerce')
industry_transpose['Japan'] = pd.to_numeric(industry_transpose['Japan'], errors='coerce')

# Displaying the modified dataframe
print(industry_transpose.head(10))

# Plotting the specified columns as a line graph
rec_industry = industry_transpose[['India', 'China', 'United States', 'Qatar', 'United Arab Emirates', 'Germany', 'Japan']] \
    .plot(kind='line', title="Industry Growth Rate", figsize=(8, 8), legend=True, fontsize=12)

# Calling the data function to add labels
data(rec_industry)

# Displaying the plot
plt.show()
# Calculate skewness of the individual values in the industry_transpose DataFrame
skewness_industry = industry_transpose.skew(axis=0, skipna=True)

# Output the skewness values for each column
print(skewness_industry.head())

# Calculate kurtosis of the individual values in the industry_transpose DataFrame
kurtosis_industry = industry_transpose.kurtosis(axis=0, skipna=True)

# Output the kurtosis values for each column
print(kurtosis_industry.head())
# Assuming 'industry_transpose' is your DataFrame
correlation_matrix_industry = industry_transpose.corr()

# Plotting the correlation matrix for industry data as a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix_industry, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix - Industry Growth Rate')
plt.show()
# Displaying the pair plot for industry data
sns.pairplot(industry_transpose[['India', 'China', 'United States', 'Qatar', 'United Arab Emirates', 'Germany', 'Japan']])
plt.suptitle('Pair Plot - Industry Growth Rate', y=1.02)
plt.show()

