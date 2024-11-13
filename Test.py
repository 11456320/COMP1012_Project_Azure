import csv
import pandas as pd

# raw data from world bank
# https://data.worldbank.org/
gdp_data = pd.read_csv("08_Oct_BB/08_Oct/data/API_NY.GDP.MKTP.CD_DS2_en_csv_v2_9865.csv",header = 2)
net_migrants_data = pd.read_csv("08_Oct_BB/08_Oct/data/API_SM.POP.NETM_DS2_en_csv_v2_10087.csv",header = 2)
urbanization_data = pd.read_csv("08_Oct_BB/08_Oct/data/API_SP.URB.TOTL.IN.ZS_DS2_en_csv_v2_10236.csv",header = 2)

column_list = gdp_data.columns.tolist()
for i in range(3):
    column_list.pop(1)
column_list.pop()

year_list = column_list[1:]

asian_tiger_countries = ['Hong Kong SAR, China', 'Singapore', 'Korea, Rep.', 'Taiwan']

country_indice = gdp_data.index[gdp_data["Country Name"].isin(asian_tiger_countries)].tolist()
print(country_indice)

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

column_format = ["Country Name","Year","GDP","Urbanization Rate","Immigration Rate"]

df = pd.DataFrame(economic_data, columns=column_format)

print(economic_data)
print(df)

df.to_csv("08_Oct_BB/08_Oct/economic_data_custom.csv", index = False)
