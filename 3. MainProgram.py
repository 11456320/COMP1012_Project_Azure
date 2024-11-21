# After combining both csv files then you can run this main program
# NOTE 1: You need to still replace random economic data with real data (if possible)
# NOTE 2: You also need to plot the data for the next 30 years (.i.e.; 2023+30years) 
# NOTE 3: Why the economic_data plots are almost equal to zero. Is it because of random data?
# Exponential Model is suppose to fit the actual data curve perfectly... Can we design this?

import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# =====================================================================================
# Load combined data from CSV file  
# Replace "combined_data.csv" to your file path here
# "combined_data.csv" is obtained from running program 1 and 2
data = pd.read_csv("08_Oct_BB/08_Oct/combined_data.csv", header=0)

# Replace "population.csv" to your file path here
population_data = pd.read_csv("08_Oct_BB/08_Oct/population_data.csv")

# Replace "economic_data_custom.csv" to your file path here
# "economic_data_custom.csv" is obtained from running program 1
economic_data = pd.read_csv("08_Oct_BB/08_Oct/economic_data_custom.csv")

# Replace "extracted_population_data.csv" to your file path here
pollution_data = pd.read_csv('08_Nov/08_Nov/extracted_pollution_data.csv')

# Replace "immigration_data_custom.csv" to your file path here
immigration_data = pd.read_csv('08_Nov/08_Nov/immigration_data_custom.csv')

# Strip whitespace from column names (if necessary)
data.columns = data.columns.str.strip()

# List of Asian Tiger countries  
asian_tiger_countries = ['Hong Kong', 'Singapore', 'South Korea', 'Taiwan']
extracted_population_data = population_data[population_data["Country Name"].isin(asian_tiger_countries)]
# =====================================================================================

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
def monte_carlo_simulation():
    # Load the population dataset  
    population_df = extracted_population_data.copy()  # Replace with your file path

    # Transform the population DataFrame to long format  
    population_df = population_df.melt(id_vars=['Country Name'], 
                                        var_name='Year', 
                                        value_name='Population')

    # Convert Year to string to match other datasets  
    population_df['Year'] = population_df['Year'].astype(str)

    # Load the economic dataset  
    economic_df = economic_data  # Replace with your file path  
    economic_df['Year'] = economic_df['Year'].astype(str)  # Ensure Year is a string

    # Load the immigration dataset  
    immigration_df = immigration_data.copy()  # Ensure this file is created and available  
    immigration_df['Year'] = immigration_df['Year'].astype(str)  # Ensure Year is a string

    # Strip whitespace from column names  
    population_df.columns = population_df.columns.str.strip()
    economic_df.columns = economic_df.columns.str.strip()
    immigration_df.columns = immigration_df.columns.str.strip()

    # Print columns of each DataFrame for debugging  
    print("Population DataFrame Columns:", population_df.columns)
    print("Economic DataFrame Columns:", economic_df.columns)
    print("Immigration DataFrame Columns:", immigration_df.columns)

    # Merge datasets on 'Country Name' and 'Year'
    merged_df = pd.merge(population_df, economic_df, on=['Country Name', 'Year'], how='inner')
    merged_df = pd.merge(merged_df, immigration_df, on=['Country Name', 'Year'], how='inner')  # Merge with immigration data

    # Print merged DataFrame columns for verification  
    print("Merged DataFrame Columns:", merged_df.columns)

    # Calculate Population Growth Percentage  
    merged_df['Population Growth (%)'] = merged_df['Population'].pct_change() * 100

    # Since 'Immigration Rate' is duplicated, we can keep one of them  
    merged_df = merged_df.drop(columns=['Immigration Rate_y'])  # Drop the redundant column  
    merged_df.rename(columns={'Immigration Rate_x': 'Immigration Rate'}, inplace=True)  # Rename the remaining column

    # Ensure that the necessary columns exist  
    if 'Population Growth (%)' not in merged_df.columns or 'Immigration Rate' not in merged_df.columns:
        raise ValueError("Merged DataFrame must contain 'Population Growth (%)' and 'Immigration Rate' columns.")

    # Define simulation parameters  
    num_simulations = 1000  # Number of Monte Carlo simulations  
    years_to_simulate = 10  # Years into the future to simulate  
    result_population = []

    # Run Monte Carlo simulations for each country  
    for country in merged_df['Country Name'].unique():
        country_data = merged_df[merged_df['Country Name'] == country]
        
        # Start with the last known population  
        last_population = country_data['Population'].iloc[-1]
        
        # Run simulations  
        simulations = []
        for _ in range(num_simulations):
            # print(_)
            simulated_population = last_population
            
            for year in range(years_to_simulate):
                # Randomly adjust population growth rate based on historical data  
                growth_rate = np.random.normal(loc=country_data['Population Growth (%)'].mean(), 
                                            scale=country_data['Population Growth (%)'].std())
                immigration_rate = np.random.normal(loc=country_data['Immigration Rate'].mean(), 
                                                    scale=country_data['Immigration Rate'].std())
                
                # Calculate new population (simple model: population growth + immigration)
                simulated_population += simulated_population * (growth_rate / 100) + immigration_rate
                
            simulations.append(simulated_population)
        
        result_population.append((country, simulations))

    # Plotting the results  
    plt.figure(figsize=(12, 8))

    for country, simulations in result_population:
        plt.hist(simulations, bins=30, alpha=0.5, label=country)

        plt.title('Monte Carlo Simulations of Future Population Growth')
        plt.xlabel('Simulated Population After 10 Years')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid()
        plt.show()

# Economic Influences Analysis Function  
def economic_influence_analysis():
    # Load the population dataset  
    population_df = extracted_population_data.copy()

    # Transform the population DataFrame to long format  
    population_df = population_df.melt(id_vars=['Country Name'], 
                                        var_name='Year', 
                                        value_name='Population')

    # Convert Year to string to match other datasets  
    population_df['Year'] = population_df['Year'].astype(str)

    # Load the economic dataset  
    economic_df = economic_data.copy()
    economic_df['Year'] = economic_df['Year'].astype(str)  # Ensure Year is a string

    # Load the pollution dataset  
    pollution_df = pollution_data.copy()
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
def environmental_factors_analysis():
    # Load the population dataset  
    population_df = extracted_population_data.copy()  # Replace with your file path

    # Transform the population DataFrame to long format  
    population_df = population_df.melt(id_vars=['Country Name'], 
                                        var_name='Year', 
                                        value_name='Population')

    # Convert Year to string to match other datasets  
    population_df['Year'] = population_df['Year'].astype(str)

    # Load the economic dataset  
    economic_df = economic_data.copy()  # Replace with your file path  
    economic_df['Year'] = economic_df['Year'].astype(str)  # Ensure Year is a string

    # Load the pollution dataset  
    pollution_df = pollution_data.copy()  # Updated path to the new pollution dataset  
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
    print("\nMerged DataFrame Summary:")
    print(merged_df.describe())

    # Now plot pollution levels from 1960 to 2023 for each country  
    plt.figure(figsize=(12, 8))
    for country in merged_df['Country Name'].unique():
        country_data = merged_df[merged_df['Country Name'] == country]

        country_data['Pollution Level (PM2.5)'] = country_data['Pollution Level (PM2.5)'].astype(float)
        
        # Convert Year to numeric for proper plotting  
        country_data['Year'] = country_data['Year'].astype(int)
        
        # Plotting pollution levels against years  
        plt.plot(country_data['Year'], country_data['Pollution Level (PM2.5)'], marker='o', label=country)

    plt.title('Pollution Levels (PM2.5) from 1960 to 2023')
    plt.xlabel('Year')
    plt.ylabel('Pollution Level (PM2.5) in µg/m³')
    plt.xticks(range(1960, 2024, 2), rotation = 90)  # Set x-ticks for every 2 years  
    plt.grid()
    plt.legend()
    plt.show()

# Immigration Impact Analysis Function  
def immigration_impact_analysis():
    # Load the population dataset  
    population_df = extracted_population_data.copy()

    # =====================================================================================
    # Load the economic dataset (Since the diagram actually only uses the immigration rate inside economic data,
    #                            we directly use immigration data instead)
    economic_df = immigration_data.copy()
    # =====================================================================================

    # Strip whitespace from column names  
    population_df.columns = population_df.columns.str.strip()
    economic_df.columns = economic_df.columns.str.strip()

    # Merge datasets on 'Country Name'
    merged_df = pd.merge(population_df, economic_df, on='Country Name', how='inner')

    # Check if the merge was successful  
    print("Merged DataFrame:")
    print(merged_df.head())

    # Plotting Immigration Rate vs Population Growth  
    plt.figure(figsize=(10, 6))
    for country in merged_df['Country Name'].unique():
        country_data = merged_df[merged_df['Country Name'] == country]
        
        # Calculate Population Growth for each year  
        # Create a new column for Population Growth (%) for every year  
        for year in range(1961, 2023):  # Assuming we have data from 1960 to 2023  
            previous_year = str(year - 1)
            current_year = str(year)
            
            if previous_year in country_data.columns and current_year in country_data.columns:
                growth = ((country_data[current_year].values[0] - country_data[previous_year].values[0]) / 
                        country_data[previous_year].values[0]) * 100
                
                # Append the growth value to the country_data DataFrame  
                country_data.loc[country_data['Year'] == year, 'Population Growth (%)'] = growth

        # Now plot the immigration rate against population growth  
        plt.scatter(country_data['Immigration Rate'], country_data['Population Growth (%)'], label=country)

    plt.title('Impact of Immigration on Population Growth in Asian Tigers')
    plt.xlabel('Immigration Rate')
    plt.ylabel('Population Growth (%)')
    plt.grid()
    plt.legend()
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

    else:
        print(f"No data found for {country_name}")

# Adjust layout and show plots  
plt.tight_layout()
plt.show()    

# Monte Carlo Simulation  
monte_carlo_simulation()

# Sensitivity analysis  
sensitivity_analysis(data)

# Economic Influences Analysis  
economic_influence_analysis()

# Environmental Factors Analysis  
environmental_factors_analysis()

# Immigration Impact Analysis  
immigration_impact_analysis()
