# This program will help us to combine the population_data.csv and economic_data.csv 
# files together

import pandas as pd

# Load the economic data  
economic_data_path = "08_Oct_BB/08_Oct/economic_data_custom.csv"
economic_data = pd.read_csv(economic_data_path)

# Load the population data  
population_data_path = "08_Oct_BB/08_Oct/population_data.csv"
population_data = pd.read_csv(population_data_path)

# Print the column names for clarity  
print("Economic Data Columns:", economic_data.columns.tolist())
print("Population Data Columns:", population_data.columns.tolist())

# Reshape the population data from wide to long format  
population_long = pd.melt(population_data, id_vars=['Country Name'], var_name='Year', value_name='Population')

# Convert 'Year' column to integer type  
population_long['Year'] = population_long['Year'].astype(int)

# Print the reshaped population data for verification  
print("Reshaped Population Data Sample:")
print(population_long.head())

# Merge the economic data with the reshaped population data  
combined_data = pd.merge(economic_data, population_long, on=['Country Name', 'Year'], how='inner')

# Save the combined DataFrame to a new CSV file  
combined_data.to_csv("08_Oct_BB/08_Oct/combined_data.csv", index=False)

print("Combined data saved to combined_data.csv")