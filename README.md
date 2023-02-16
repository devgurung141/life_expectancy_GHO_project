# Analyze Life Expectancy Data

# Project Description
The goal of the project is to analyze data, find key features affecting life expectancy and create a model to predict life expectancy

# Project Goals:
* find key features affecting life expectancy
* find features relation with life expectancy
* create a model to predict life expectancy
* provide recommendations to raise life expectancy

# Initial Questions
* Does countries income determine life expectancy?
* Does infant deaths occur more in low income countries?
* Immunization relations with death of under five old
* What is the relationship of percentage of expenditure and life expectancy?
* What is the relation of GDP and life expectancy?

# The Plan

* Acquire data
    * Acquired data from  databank.worldbank.org and WHO website

* Prepare data 
    * Rename column names
    * Drop unnecessary columns. 
    * Merge dataframes.
    * erge dataframes
    * Checked for Nulls, fill Nulls using forward fill and back fill.
    * split data into train, validate and test (approximatley 56/24/20)

* Explore Data
    * Use graph and hypothesis testing to find attributes relation with life expectancy, and answer the following initial questions
        * Does countries income determine life expectancy?
        * Does infant deaths occur more in low income countries?
        * Immunization relations with death of under five old
        * What is the relationship of percentage of expenditure and life expectancy?
        * What is the relation of GDP and life expectancy?

* Develop Model
    * Use MinMaxScaler to scale data
    * Set up baseline RMSE
    * Evaluate models on train data and validate data
    * Select the best model based on the RMSE
    * Evaluate the best model on test data to make predictions

* Draw Conclusions

# Data Dictionary
| Feature | Definition |
|:--------|:-----------|
| country | Country|
| year | Year|
| life_expectancy | Life Expectancy in age|
| life_expectancy_male | Male Life Expectancy in age|
| life_expectancy_Femle | Femael Life Expectancy in age|
| infant_deaths | Number of Infant Deaths per 1000 population|
| under_five_deaths | Mortality rate, under-5 (per 1,000 live births)|
| traffic_deaths | Mortality caused by road traffic injury (per 100,000 population)|
| hepatitis_B | Hepatitis B (HepB) immunization coverage among 1-year-olds (%)|
| diphtheria | Diphtheria tetanus toxoid and pertussis (DTP3) immunization coverage among 1-year-olds (%)|
| measles | Measles - number of reported cases per 1000 population|
| populaiton| Population|
| GDP | Gross Domestic Product per capita (in USD)|
| percentage_expenditure | Expenditure on health as a percentage of Gross Domestic Product per capita(%)|
| income_group| The World Bankd classification of the world's economies based on the GNI per capita of the previous year|

# Steps to Reproduce
1. Clone this repo 
2. Data can be also be acquired from [Databank](https://databank.worldbank.org/reports.aspx?source=world-development-indicators#), [WHO](https://www.who.int/data/gho/data/indicators)save a file as 'GHO.csv','CHE.csv','income.csv', and put the file into the cloned repo 
2. Run notebook

# Takeaways and Conclusions
* GDP, immunizations, percentage expendiutre on health have positive coorelation with life expectancy.
* Infant deaths, deaths of under five, traffice death have negative correlation with life expectancy.
* Poor countries have life expenctancy below average life expectancy of 71.71 whereas rich coutries have higher life expectancy and above average life expectancy.
* Poor countries have higher infant deaths.
* All models perfomed better than baseline on train and validat data.
* All models perfomed worse on validate to train
* Tweedie Regression with power 1 and alpha 0 performed better than other models and I am going to use it for test data.

# Recommendations 
* recommend low income countries to run program to decrease infant deaths. There is higher number of infant deaths.
* recommend to increase in expenses to health to rise life expectancy.



