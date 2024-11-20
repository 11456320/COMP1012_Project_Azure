import csv
import pandas as pd

# raw data from world bank
# p.s. does not include Taiwan
# https://data.worldbank.org/
gdp_data = pd.read_csv("08_Oct_BB/08_Oct/data/Global_GDP_data.csv",header = 2)
net_migrants_data = pd.read_csv("08_Oct_BB/08_Oct/data/Global_Net_Migration_data.csv",header = 2)
urbanization_data = pd.read_csv("08_Oct_BB/08_Oct/data/Global_Urbanization_Rate_data.csv",header = 2)

# raw Taiwan data
# Source: https://nstatdb.dgbas.gov.tw/dgbasall/webMain.aspx?k=engmain (GDP, Yearly)
# https://www.worldometers.info/world-population/taiwan-population/ (Net migration, Urbanization, per 5 year)
taiwan_economic_data = pd.read_csv("08_Oct_BB/08_Oct/data/Taiwan_economic_data.csv", header = 3)

# Since the source only have Net Migration and Urbanization every 5 year,
# We use linear interpolation method in Pandas to fill in the missing Urbanization Rate
# while keeping a good estimation of time series data where values tend to change gradually over time.
taiwan_economic_data["Urbanization Rate"] = taiwan_economic_data["Urbanization Rate"].interpolate(method='linear')

print(taiwan_economic_data["Urbanization Rate"])

# For Immigration Rate, since the data does not look like a linear trend,
# we use mean imputation to fill in the data
net_migration_mean = taiwan_economic_data["Immigration Rate"].mean()
taiwan_economic_data["Immigration Rate"].fillna(net_migration_mean, inplace=True)

taiwan_economic_data.to_csv('08_Oct_BB/08_Oct/data/filled_taiwan_economic_data.csv', index=False)

# print(taiwan_economic_data["GDP"])

column_list = gdp_data.columns.tolist()
for i in range(3):
    column_list.pop(1)
column_list.pop()

year_list = column_list[1:]

asian_tiger_countries = ['Hong Kong SAR, China', 'Singapore', 'Korea, Rep.', 'Taiwan']

country_indice = gdp_data.index[gdp_data["Country Name"].isin(asian_tiger_countries)].tolist()

economic_data = []
for country_index in country_indice:
    
    gdp_row_data = gdp_data.iloc[country_index]
    net_migrants_row_data = net_migrants_data.iloc[country_index]
    urbanization_row_data = urbanization_data.iloc[country_index]

    country_name = gdp_data.loc[country_index, "Country Name"]
    if country_name == "Hong Kong SAR, China":
        country_name = "Hong Kong"
    if country_name == "Korea, Rep.":
        country_name = "South Korea"

    for year in year_list:
        economic_data.append([country_name, year, gdp_row_data[year], urbanization_row_data[year], net_migrants_row_data[year]])

for i in range(len(year_list)):
    economic_data.append(["Taiwan", year_list[i], taiwan_economic_data["GDP"][i] * 1000000, taiwan_economic_data["Urbanization Rate"][i], taiwan_economic_data["Immigration Rate"][i]])

column_format = ["Country Name","Year","GDP","Urbanization Rate","Immigration Rate"]

df = pd.DataFrame(economic_data, columns=column_format)

# print(economic_data)
# print(df)

df.to_csv("08_Oct_BB/08_Oct/economic_data_custom.csv", index = False)
