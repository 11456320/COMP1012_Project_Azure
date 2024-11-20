# After combining both csv files then you can run this main program
# NOTE 1: You need to still replace random economic data with real data (if possible)
# NOTE 2: You also need to plot the data for the next 30 years (.i.e.; 2023+30years) 
# NOTE 3: Why the economic_data plots are almost equal to zero. Is it because of random data?
# Exponential Model is suppose to fit the actual data curve perfectly... Can we design this?

import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Load combined data from CSV file  
data = pd.read_csv("08_Oct_BB/08_Oct/combined_data.csv", header=0)

population_data = pd.read_csv("08_Oct_BB/08_Oct/population_data.csv")

economic_data = pd.read_csv("08_Oct_BB/08_Oct/economic_data_custom.csv")

pollution_data = pd.read_csv('08_Nov/08_Nov/pollution_data_1960_2023.csv')

# Strip whitespace from column names (if necessary)
data.columns = data.columns.str.strip()

# List of Asian Tiger countries  
asian_tiger_countries = ['Hong Kong', 'Singapore', 'South Korea', 'Taiwan']

# # Function for plotting population data  
# def plot_population_data(country_name, years, population, exp_pred, poly_pred, gompertz_pred):
#     # Create subplot  
#     plt.subplot(2, 2, i + 1)
#     plt.scatter(years, population, label='Real Data', color='blue')
#     plt.plot(extended_years, exp_pred, label='Exponential Model', color='orange')
#     plt.plot(extended_years, poly_pred, label='Polynomial Model', color='green')
#     plt.plot(extended_years, gompertz_pred, label='Gompertz Model', color='red')
#     plt.title(f'Population Growth in {country_name}')
#     plt.xlabel('Year')
#     plt.ylabel('Population')
#     plt.xticks(np.arange(years[0], years[-1] + 31, 5))
#     plt.legend()
#     plt.grid()
#     plt.show()

# Function to conduct sensitivity analysis  
def sensitivity_analysis(df):
    # Print column names for debugging  
    # print("Columns in DataFrame:", df.columns)

    # Strip whitespace from column names  
    df.columns = df.columns.str.strip()

    # Check if 'Country Name' column exists  
    if 'Country Name' not in df.columns:
        print("Error: 'Country Name' column not found in the DataFrame.")
        return

    results = {}
    
    # ==================================================================================================
    # Define country-specific death and immigration rates  
    # Source: https://www.macrotrends.net/global-metrics/countries/SGP/singapore/birth-rate
    # The major www.macrotrends.net is the source of the data. Just search the country you want.
    country_rates = {
        'Hong Kong': {'death_rate': 7.274, 'immigration_rate': 3.442},
        'South Korea': {'death_rate': 7.099, 'immigration_rate': 0.429},
        'Taiwan': {'death_rate': 8.293, 'immigration_rate': 1.004},
        'Singapore': {'death_rate': 5.403, 'immigration_rate': 4.502},
    }
    asian_tiger_countries = ['Hong Kong', 'Singapore', 'South Korea', 'Taiwan']

    df = df[df["Country Name"].isin(asian_tiger_countries)]
    # ==================================================================================================

    for country in df['Country Name'].unique():

        # Example: Calculate growth based on varying birth rates  
        birth_rates = np.arange(1.0, 15.0, 0.1)  # Adjusted range for more variation  

        growth_rates = []
        for rate in birth_rates:
            # Use country-specific rates for calculation  
            death_rate = country_rates[country]['death_rate']
            immigration_rate = country_rates[country]['immigration_rate']
            growth = rate - death_rate + immigration_rate  
            growth_rates.append(growth)

        results[country] = growth_rates
        
        # Debugging: Print growth rates for each country  
        # print(f"{country} Growth Rates: {growth_rates}")

    # Plotting results  
    plt.figure(figsize=(10, 6))
    for country, growth in results.items():
        plt.plot(birth_rates, growth, label=country)

    plt.title('Sensitivity Analysis of Population Growth')
    plt.xlabel('Adjusted Birth Rate')
    plt.ylabel('Population Growth Rate')
    plt.axhline(0, color='red', linestyle='--')  # Zero growth line  
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
def economic_influence_analysis():
    # Load the population dataset  
    population_df = population_data
    # "08_Nov/08_Nov/1population_data.csv"
    # "08_Oct_BB/08_Oct/combined_data.csv"

    asian_tiger_countries = ['Hong Kong', 'Singapore', 'South Korea', 'Taiwan']

    population_df = population_df[population_df["Country Name"].isin(asian_tiger_countries)]

    print(population_df)

    # Transform the population DataFrame to long format  
    population_df = population_df.melt(id_vars=['Country Name'], 
                                        var_name='Year', 
                                        value_name='Population')

    # Convert Year to string to match other datasets  
    population_df['Year'] = population_df['Year'].astype(str)

    # Load the economic dataset  
    economic_df = economic_data
    economic_df['Year'] = economic_df['Year'].astype(str)  # Ensure Year is a string

    # Load the pollution dataset  
    pollution_df = pollution_data
    pollution_df['Year'] = pollution_df['Year'].astype(str)  # Ensure Year is a string

    # Strip whitespace from column names  
    population_df.columns = population_df.columns.str.strip()
    economic_df.columns = economic_df.columns.str.strip()
    pollution_df.columns = pollution_df.columns.str.strip()

    # Merge datasets on 'Country Name' and 'Year'
    merged_df = pd.merge(population_df, economic_df, on=['Country Name', 'Year'], how='inner')
    merged_df = pd.merge(merged_df, pollution_df, on=['Country Name', 'Year'], how='inner')  # Merge with pollution data

    # Check if the merge was successful  
    print("Merged DataFrame:")
    print(merged_df.head())
    print("Merged DataFrame Columns:", merged_df.columns)

    # Calculate GDP Growth Percentage  
    merged_df['GDP Growth (%)'] = merged_df.groupby('Country Name')['GDP'].pct_change() * 100

    # Calculate Population Growth Percentage  
    merged_df['Population Growth (%)'] = merged_df.groupby('Country Name')['Population'].pct_change() * 100

    # Plotting GDP growth vs Population growth  
    plt.figure(figsize=(10, 6))
    for country in merged_df['Country Name'].unique():
        country_data = merged_df[merged_df['Country Name'] == country]
        plt.scatter(country_data['GDP Growth (%)'], country_data['Population Growth (%)'], label=country)

    plt.title('Economic Influences on Population Growth in Asian Tigers')
    plt.xlabel('GDP Growth (%)')
    plt.ylabel('Population Growth (%)')
    plt.grid()
    plt.legend()
    plt.show()

# Environmental Factors Analysis Function  
def environmental_factors_analysis(country_data):
    urbanization_rate = country_data['Urbanization Rate'].values  # Assuming this data exists  
    population = country_data['Population'].values  # Population from 1960 to 2023  
    years = country_data['Year'].values  # Extract years directly from the DataFrame

    plt.figure(figsize=(10, 6))
    plt.plot(years, population, label='Population', marker='o', color='blue')

    # ==================================================================================================
    plt.plot(years, population * urbanization_rate / 100, label='Urbanization Rate', marker='x', color='green')
    # ==================================================================================================

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

# Define the models  
def exponential_growth(t, P0, r):
    return P0 * np.exp(r * t)

def polynomial_model(t, a, b, c):
    return a * t**2 + b * t + c

def gompertz(t, K, A, r):
    return K * np.exp(-A * np.exp(-r * t))

# Main Analysis Loop for Each Country  
for i, country_name in enumerate(asian_tiger_countries):
    country_data = data[data['Country Name'] == country_name]

    if not country_data.empty:
        # Extract population data for the selected country  
        years = country_data['Year'].values  # Get the years directly from the DataFrame  
        population = country_data['Population'].values  # Get the population directly from the DataFrame
        gdp = country_data['GDP'].values  # Extract GDP data  
        urbanization_rate = country_data['Urbanization Rate'].values  # Extract urbanization rate data  
        immigration_rate = country_data['Immigration Rate'].values  # Extract immigration rate data

                # Print the data for debugging  
        # print(f"Data for {country_name}:")
        # print("Years:", years)
        # print("Population:", population)
        # print("GDP:", gdp)
        # print("Urbanization Rate:", urbanization_rate)
        # print("Immigration Rate:", immigration_rate)
        # print("GDP Min:", gdp.min(), "Max:", gdp.max())
        # print("Urbanization Rate Min:", urbanization_rate.min(), "Max:", urbanization_rate.max())
        # print("Immigration Rate Min:", immigration_rate.min(), "Max:", immigration_rate.max())
        # print("NaNs in GDP:", country_data['GDP'].isnull().sum())

        # Verify the length of the population data  
        if len(population) != len(years):
            print(f"Warning: Population data length ({len(population)}) does not match years length ({len(years)}) for {country_name}.")
            continue

        # Fit models  
        t = years - years[0]  # Normalize years  
        popt_exp, _ = curve_fit(exponential_growth, t, population, p0=[population[0], 0.01])
        popt_poly, _ = curve_fit(polynomial_model, t, population)
        popt_gom, _ = curve_fit(gompertz, t, population, p0=[population.max(), 0.1, 0.03])

        # Predict future population for the next 30 years  
        extended_years = np.arange(years[0], years[-1] + 31)
        exp_pred = exponential_growth(extended_years - years[0], *popt_exp)
        poly_pred = polynomial_model(extended_years - years[0], *popt_poly)
        gompertz_pred = gompertz(extended_years - years[0], *popt_gom)

        # Create subplot  
        plt.subplot(2, 2, i + 1)
        plt.scatter(years, population, label='Real Data', color='blue')
        plt.plot(extended_years, exp_pred, label='Exponential Model', color='orange')
        plt.plot(extended_years, poly_pred, label='Polynomial Model', color='green')
        plt.plot(extended_years, gompertz_pred, label='Gompertz Model', color='red')
        plt.title(f'Population Growth in {country_name}')
        plt.xlabel('Year')
        plt.ylabel('Population')
        plt.xticks(np.arange(years[0], years[-1] + 31, 5), rotation = 90)
        plt.legend()
        plt.grid()

        # Monte Carlo Simulation  
        # monte_carlo_simulation(country_name, initial_population)



        # # Environmental Factors Analysis  
        # environmental_factors_analysis(country_data)

        # # Immigration Impact Analysis  
        # immigration_impact_analysis(country_data)
    else:
        print(f"No data found for {country_name}")

# Adjust layout and show plots  
plt.tight_layout()
plt.show()    

# Sensitivity analysis  
sensitivity_analysis(data)

# Economic Influences Analysis  
economic_influence_analysis()
