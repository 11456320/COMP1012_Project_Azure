1. Run the programs in sequence (1, 2 and then 3)           <--------
2. Change the path/directory inside every program file      <--------

Program 1: (1. generate_dataset_Econimic_Data_Generation_GDP_Urbanization_ImmigrationRate.py)
# This file will help you to generate random dataset for GDP, Urbanization and Immigration
# Task 1: Your task is to find real world data online and replace with randomly generated data
# Task 2: The plots are plotted until 2023/2204, but you need to plot for another 30 years 

Program 1: (test.py)                                        <--------
# This file currently will take the real world data from the world bank if you have them downloaded the data. The files are also included in this repository.
The files:
API_NY.GDP.MKTP.CD_DS2_en_csv_v2_9865.csv (GPD data)
API_SM.POP.NETM_DS2_en_csv_v2_10087.csv (Immigration Rate)
API_SP.URB.TOTL.IN.ZS_DS2_en_csv_v2_10236.csv (Urbanization rate)

P.S. The world bank data does not contain the data from Taiwan.
We need to find the data for Taiwan. :D

Program 2: (2. combining_eco_data+popu_data.py)
# This program will help us to combine the population_data.csv and economic_data.csv 
# files together

Program 3: (3. MainProgram.py)
# After combining both csv files then you can run this main program
# NOTE 1: You need to still replace random economic data with real data (if possible)
# NOTE 2: You also need to plot the data for the next 30 years (.i.e.; 2023+30years) 
# NOTE 3: Why the economic_data plots are almost equal to zero. Is it because of random data?
# Exponential Model is suppose to fit the actual data curve perfectly... Can we design this?
