# This file will help you to generate random dataset for GDP, Urbanization and Immigration
# Task 1: Your task is to find real world data online and replace with randomly generated data
# Task 2: The plots are plotted until 2023/2204, but you need to plot for another 30 years 

import pandas as pd  
import numpy as np

# Define the years and countries  
years = np.arange(1960, 2024)
countries = ['Hong Kong', 'Singapore', 'South Korea', 'Taiwan']

# Create a list to hold the data  
data = []

# Generate sample data  
for country in countries:
    for year in years:
        gdp = np.random.uniform(10000, 50000)  # Sample GDP between 10,000 and 50,000  
        urbanization_rate = np.random.uniform(30, 100)  # Sample urbanization rate between 30% and 100%
        immigration_rate = np.random.uniform(0, 20)  # Sample immigration rate between 0 and 20 per 1,000 people

        # Append the data to the list  
        data.append({
            'Country Name': country,
            'Year': year,
            'GDP': gdp,
            'Urbanization Rate': urbanization_rate,
            'Immigration Rate': immigration_rate  
        })

# Convert the list into a DataFrame  
data_df = pd.DataFrame(data)

# Save the DataFrame to a CSV file  
data_df.to_csv("08_Oct_BB/08_Oct/economic_data.csv", index=False)

print("Economic data saved to economic_data.csv")
