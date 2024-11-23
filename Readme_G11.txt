# COMP1012 Semester Project - Project_Azure Grp 11

COMP1012 Semester Project

https://github.com/11456320/COMP1012_Project_Azure

## Group member
23079573D   TSE   YiuHang <-- Group Leader

23078385D	LI	   SingYui

23078608D	LEE	HeungWang

23078873D	TANG	ChiFung

23078286D	LAI	TszYuen

## How to run the program
1. Download this repository and unzip it to your vscode virtual environment directory.
   Like this:
   ```
   <dir>/.venv
   <dir>/COMP1012_Project_Azure-main.zip
   ```
2. Unzip the `COMP1012_Project_Azure-main.zip` file

3. Change the path/directory inside every program file

   If the zip file is placed under your vscode virtual environment properly, the path/directory of the csv file inside every program file should be
   `"COMP1012_Project_Azure-main/COMP1012_Project_Azure-main/data/<data.csv>"` and the output graph should be `"COMP1012_Project_Azure-main/COMP1012_Project_Azure-main/plotted_images/<plotted_images.csv>"`

   ```
   # Replace <your_path> to your own vscode path
   your_path = "COMP1012_Project_Azure-main/COMP1012_Project_Azure-main/"

   # This is how the program read the csv file
   dataframe = pd.read_csv(your_path + "data/<data.csv>")

   # This is how the program save the graph as .jpg format
   plt.savefig(f"{your_path}plotted_images/<plotted_images.jpg>", dpi=300)
   ```

5. Install the libraries required
   ```
   pip install pandas
   pip install numpy
   pip install matplotlib
   pip install scipy
   ```

6. Run the main program `SC_G11.py`

## Bonus module

   [0.25 Points] 1. Sensitivity Analysis:
   
   A comprehensive sensitivity analysis will be conducted to understand how changes in key parameters (such
   as birth rates, death rates, and immigration) affect population growth predictions. This analysis will identify which
   factors have the most significant impact on growth and provide insights into the robustness of the forecasts.
   
   [0.25 Points] 2. Economic Influences:
   
   You will examine how economic factors, such as GDP growth and employment rates, influence population
   growth in the Asian Tigers. This component will involve analyzing historical economic data alongside population
   trends to identify correlations and trends that can inform construction planning.
   
   [0.25 Points] 3. Environmental Factors:
   
   This analysis will assess the impact of environmental factors (e.g., pollution levels, urbanization rates) on
   population dynamics. Understanding how environmental conditions affect population growth is vital for
   sustainable construction practices and urban development policies.
   
   [0.25 Points] 4. Stochastic Modeling Using Monte Carlo Simulations:
   
   You will introduce randomness into the population growth models through Monte Carlo simulations. This
   component will simulate various scenarios that account for uncertainties in growth rates, immigration, and other
   demographic factors. By visualizing the range of possible outcomes, stakeholders will gain a clearer
   understanding of the potential variability in future population growth.
   
   [0.20 Points] 5. Impact of Immigration:
   
   An analysis of the role of immigration in shaping population growth trends in the Asian Tigers will be
   conducted. This component will compare historical immigration data against population changes to understand
   how immigration patterns affect overall demographic dynamics.

## Expected output

1. Population Prediction for the next 30 years using different mathematical models

![](https://github.com/11456320/COMP1012_Project_Azure/blob/main/plotted_images/population_prediction.jpg)

2. Sensitivity Analysis

![](https://github.com/11456320/COMP1012_Project_Azure/blob/main/plotted_images/sensitivity_analysis.jpg)

3. Economic Influences

![](https://github.com/11456320/COMP1012_Project_Azure/blob/main/plotted_images/economic_influence_analysis.jpg)

4. Environmental Factors

![](https://github.com/11456320/COMP1012_Project_Azure/blob/main/plotted_images/environmental_factors_analysis.jpg)

5. Stochastic Modeling Using Monte Carlo Simulations

Hong Kong:
![](https://github.com/11456320/COMP1012_Project_Azure/blob/main/plotted_images/Hong%20Kong_monte_carlo_simulation.jpg)

Singapore:
![](https://github.com/11456320/COMP1012_Project_Azure/blob/main/plotted_images/Singapore_monte_carlo_simulation.jpg)

South Korea:
![](https://github.com/11456320/COMP1012_Project_Azure/blob/main/plotted_images/South%20Korea_monte_carlo_simulation.jpg)

Taiwan:
![](https://github.com/11456320/COMP1012_Project_Azure/blob/main/plotted_images/Taiwan_monte_carlo_simulation.jpg)

6. Impact of Immigration

![](https://github.com/11456320/COMP1012_Project_Azure/blob/main/plotted_images/immigration_impact_analysis.jpg)
