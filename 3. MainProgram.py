# After combining both csv files then you can run this main program
# NOTE 1: You need to still replace random economic data with real data (if possible)
# NOTE 2: You also need to plot the data for the next 30 years (.i.e.; 2023+30years) 
# NOTE 3: Why the economic_data plots are almost equal to zero. Is it because of random data?
# Exponential Model is suppose to fit the actual data curve perfectly... Can we design this?

import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt

# Load combined data from CSV file  
data = pd.read_csv("08_Oct_BB/08_Oct/combined_data.csv", header=0)

# Strip whitespace from column names (if necessary)
data.columns = data.columns.str.strip()

# List of Asian Tiger countries  
asian_tiger_countries = ['Hong Kong', 'Singapore', 'South Korea', 'Taiwan']

# Define the exponential growth function  
def exponential_growth(initial_population, growth_rate, time):
    return initial_population * np.exp(growth_rate * time)

# Function for plotting population data  
def plot_population_data(country_name, years, population, predicted_population):
    plt.figure(figsize=(10, 6))
    plt.plot(years, population, label='Actual Population', marker='o', color='blue')
    plt.plot(years, predicted_population, label='Predicted Exponential Growth', linestyle='--', color='orange')
    plt.title(f'Population Growth for {country_name}')
    plt.xlabel('Year')
    plt.ylabel('Population')
    plt.legend()
    plt.grid()
    plt.show()

# Sensitivity Analysis Function  
def sensitivity_analysis(country_data, initial_population):
    growth_rates = np.linspace(0.01, 0.05, 5)  # Example growth rates  
    years = np.arange(1960, 2024)
    
    for rate in growth_rates:
        predicted_population = exponential_growth(initial_population, rate, years - years[0])
        plt.plot(years, predicted_population, label=f'Growth Rate: {rate:.2f}')
    
    plt.title('Sensitivity Analysis of Growth Rates')
    plt.xlabel('Year')
    plt.ylabel('Population')
    plt.legend()
    plt.grid()
    plt.show()

# Monte Carlo Simulation Function  
def monte_carlo_simulation(country_name, initial_population, iterations=1000):
    growth_rates = np.random.uniform(0.01, 0.05, size=iterations)  # Random growth rates  
    years = np.arange(1960, 2024)
    final_populations = []

    for rate in growth_rates:
        final_population = exponential_growth(initial_population, rate, years - years[0])  # Generate the full population array  
        final_populations.append(final_population[-1])  # Append only the last value

    plt.hist(final_populations, bins=30, alpha=0.7)
    plt.title(f'Monte Carlo Simulation of Final Population for {country_name}')
    plt.xlabel('Final Population')
    plt.ylabel('Frequency')
    plt.grid()
    plt.show()

# Economic Influences Analysis Function  
def economic_influence_analysis(country_data):
    gdp = country_data['GDP'].values  # Assuming the GDP data is in the CSV  
    population = country_data['Population'].values  # Population from 1960 to 2023  
    years = country_data['Year'].values  # Extract years directly from the DataFrame

    plt.figure(figsize=(10, 6))
    plt.plot(years, population, label='Population', marker='o', color='blue')
    plt.plot(years, gdp, label='GDP', marker='x', color='orange')
    plt.title('Population vs GDP for ' + country_data['Country Name'].values[0])
    plt.xlabel('Year')
    plt.ylabel('Value')
    plt.legend()
    plt.grid()
    plt.show()

# Environmental Factors Analysis Function  
def environmental_factors_analysis(country_data):
    urbanization_rate = country_data['Urbanization Rate'].values  # Assuming this data exists  
    population = country_data['Population'].values  # Population from 1960 to 2023  
    years = country_data['Year'].values  # Extract years directly from the DataFrame

    plt.figure(figsize=(10, 6))
    plt.plot(years, population, label='Population', marker='o', color='blue')
    plt.plot(years, urbanization_rate, label='Urbanization Rate', marker='x', color='green')
    plt.title('Population vs Urbanization Rate for ' + country_data['Country Name'].values[0])
    plt.xlabel('Year')
    plt.ylabel('Value')
    plt.legend()
    plt.grid()
    plt.show()

# Immigration Impact Analysis Function  
def immigration_impact_analysis(country_data):
    immigration_rate = country_data['Immigration Rate'].values  # Assuming this data exists  
    population = country_data['Population'].values  # Population from 1960 to 2023  
    years = country_data['Year'].values  # Extract years directly from the DataFrame

    plt.figure(figsize=(10, 6))
    plt.plot(years, population, label='Population', marker='o', color='blue')
    plt.plot(years, immigration_rate, label='Immigration Rate', marker='x', color='red')
    plt.title('Population vs Immigration Rate for ' + country_data['Country Name'].values[0])
    plt.xlabel('Year')
    plt.ylabel('Value')
    plt.legend()
    plt.grid()
    plt.show()

# Main Analysis Loop for Each Country  
for country_name in asian_tiger_countries:
    country_data = data[data['Country Name'] == country_name]

    if not country_data.empty:
        # Extract population data for the selected country  
        years = country_data['Year'].values  # Get the years directly from the DataFrame  
        population = country_data['Population'].values  # Get the population directly from the DataFrame
        gdp = country_data['GDP'].values  # Extract GDP data  
        urbanization_rate = country_data['Urbanization Rate'].values  # Extract urbanization rate data  
        immigration_rate = country_data['Immigration Rate'].values  # Extract immigration rate data

                # Print the data for debugging  
        print(f"Data for {country_name}:")
        print("Years:", years)
        print("Population:", population)
        print("GDP:", gdp)
        print("Urbanization Rate:", urbanization_rate)
        print("Immigration Rate:", immigration_rate)
        print("GDP Min:", gdp.min(), "Max:", gdp.max())
        print("Urbanization Rate Min:", urbanization_rate.min(), "Max:", urbanization_rate.max())
        print("Immigration Rate Min:", immigration_rate.min(), "Max:", immigration_rate.max())
        print("NaNs in GDP:", country_data['GDP'].isnull().sum())

        # Verify the length of the population data  
        if len(population) != len(years):
            print(f"Warning: Population data length ({len(population)}) does not match years length ({len(years)}) for {country_name}.")
            continue

        # Fit the exponential model to the data to estimate growth rate  
        initial_population = population[0]  # First entry in the dataset  

        # Plot population data  
        predicted_population = exponential_growth(initial_population, 0.02, years - years[0])  # Example growth rate  
        plot_population_data(country_name, years, population, predicted_population)

        # Sensitivity analysis  
        sensitivity_analysis(country_data, initial_population)

        # Monte Carlo Simulation  
        monte_carlo_simulation(country_name, initial_population)

        # Economic Influences Analysis  
        economic_influence_analysis(country_data)

        # Environmental Factors Analysis  
        environmental_factors_analysis(country_data)

        # Immigration Impact Analysis  
        immigration_impact_analysis(country_data)
    else:
        print(f"No data found for {country_name}")