# imports 

import pandas as pd
import numpy as np

import os

from sklearn.model_selection import train_test_split


def acquire_life_data():
    
    '''read CSV file from a local storage and retrun a dataframe'''
    
    # read data from csv file of local storage
    df = pd.read_csv('GHO.csv')
    
    # rename columns 
    df.rename(columns = {'Country Name': 'country', 
        'Time': 'year',
       'Life expectancy at birth, total (years) [SP.DYN.LE00.IN]': 'life_expectancy',
       'Life expectancy at birth, male (years) [SP.DYN.LE00.MA.IN]': 'life_expectancey_male',
       'Life expectancy at birth, female (years) [SP.DYN.LE00.FE.IN]': 'life_expectancey_female',
       'Mortality rate, infant (per 1,000 live births) [SP.DYN.IMRT.IN]': 'infant_deaths',
       'Mortality rate, under-5 (per 1,000 live births) [SH.DYN.MORT]': 'under_five_deaths',
       'Mortality rate attributed to unsafe water, unsafe sanitation and lack of hygiene (per 100,000 population) [SH.STA.WASH.P5]': 'death_unsafe_sanitation',
       'Mortality caused by road traffic injury (per 100,000 population) [SH.STA.TRAF.P5]': 'traffic_deaths',
       'Immunization, HepB3 (% of one-year-old children) [SH.IMM.HEPB]': 'hepatitis_B',
       'Immunization, measles (% of children ages 12-23 months) [SH.IMM.MEAS]': 'measles',
       'Immunization, DPT (% of children ages 12-23 months) [SH.IMM.IDPT]': 'diphtheria',
       'Population, total [SP.POP.TOTL]': 'population',
       'GDP per capita (current US$) [NY.GDP.PCAP.CD]':'GDP'},inplace=True)
    
    # drop unwanted columns
    df.drop(columns = ['Country Code', 'Time Code', 'death_unsafe_sanitation'],inplace = True )
    
    # sort values and index
    df = df.sort_values(by=['country', 'year'], ascending=[True, False], ignore_index=True)
    
    # rename country name
    df.country = df.country.replace({"Congo": "Congo, Rep."})
    
    return df


def acquire_expenditure_data():
    
    """read CSV file from a local and retrun a dataframe"""
    
    # read datat from csv file of local storage
    df = pd.read_csv('CHE.csv')
    
    # get only needed columns
    df = df[['Location','Period','Value'  ]]
    
    # get only needed years
    df = df[df['Period'] > 2010]
    
    # rename columns
    df.rename(columns = {'Location': 'country', 'Period':'year','Value':'percentage_expenditure'},inplace=True)
    
    # sort values and index
    df = df.sort_values(by=['country', 'year'], ascending=[True, False], ignore_index=True)
    
    # rename countries names so that data will not be lost on merge. different csv files have different names for a same country
    df.country = df.country.replace({"Bolivia (Plurinational State of)":"Bolivia",
                                     "Congo": "Congo, Rep.",
                                     "Democratic Republic of the Congo": "Congo, Dem. Rep.",
                                     "Egypt": "Egypt, Arab Rep.",
                                     "Gambia":"Gambia, The", 
                                     "Iran (Islamic Republic of)": "Iran, Islamic Rep.",
                                     "Kyrgyzstan": "Kyrgyz Republic",
                                     "Republic of Korea": "Korea, Rep.",
                                    "The former Yugoslav Republic of Macedonia":"North Macedonia" ,
                                    "Slovakia": "Slovak Republic",
                                     'Republic of Moldova': "Moldova",
                                    "United Republic of Tanzania":  "Tanzania", 
                                    "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
                                    "United States of America": "United States",
                                     "Viet Nam": "Vietnam",
                                    "Yemen": 'Yemen, Rep.',
                                    "Venezuela (Bolivarian Republic of)":"Venezuela, RB",
                                     'Türkiye': 'Turkiye', 
                                     "Lao People's Democratic Republic": 'Lao PDR'
                                        })
    
    return df  


def acquire_income_data(): 
    
    """read CSV file from a local and retrun a dataframe"""

    df = pd.read_csv('income.csv')
    
    # get only needed columns
    df = df[['Country', 'Income group']]
    
    # rename columns
    df.rename(columns = {'Country':'country', 'Income group': 'income_group'},inplace=True)
    
    # rename country name
    df.country = df.country.replace({
                                    'Czech Republic': 'Czechia',
                                    'Turkey': 'Turkiye'
                                     
                                     })
    
    return df   


def acquire_data():
    
#   ''' acquire data from csv file calling functions, merge datas and return a dataframe with merged data'''

    # acquire data

    df = acquire_life_data()
    df_1 = acquire_expenditure_data()
    df_2 = acquire_income_data()

    # merge dataframes
    df= pd.merge(df, df_1, how='inner', on=['country','year'])
    df= pd.merge(df, df_2, how='inner', on=['country'])

    return df


def clean_data(df):
    
    
    # fill missing values for venezuela
    df.loc[1763:1773,["income_group"]] = "Low income"
    
    # replace '..' with NAN 
    df = df.replace("..", np.NAN)
    
    # fill missing values with values avilable in front
    df.loc[:,['traffic_deaths']] = df.loc[:,['traffic_deaths']].bfill()
    
    # fill missing values with values avilable in front
    df.loc[:,['hepatitis_B']] = df.loc[:,['hepatitis_B']].ffill()
    
    # fill missing values with values avilable in back
    df.loc[:,['GDP']] = df.loc[:,['GDP']].bfill()
    
    # get countries which are missing values in almost all columns
    country_to_drop = df[df.life_expectancy.isnull()].country.unique().tolist()
    
    # drop countries which are missing values in almost all columns
    df=df[~df['country'].isin(country_to_drop)]
    
    # reset index
    df = df.reset_index(drop=True)
    
    # change data types
    df['year']= df['year'].astype('int')
    
     # change data types
    for col in df.drop(columns = ['country', 'year', 'income_group']).columns:
        df[col] = df[col].astype('float')
        df[col] = round(df[col], 2)
    
    # encode categorical variables
    dummies = pd.get_dummies(df['income_group'])
    
    # concat dataframes with dummy variables
    df = pd.concat([df, dummies], axis=1)
    
    return df



def train_val_test(df, seed=42):
    '''split data into train, validate and test data'''
    
    # split data into 80% train_validate, 20% test
    train_validate, test = train_test_split(df, test_size=0.2, 
                                            random_state=seed)
    
    # split train_validate data into 70% train, 30% validate
    train, validate = train_test_split(train_validate, test_size=0.3, 
                                       random_state=seed)
    
    # returns train, validate, test data
    return train, validate, test

def wrangle_data(): 
    '''acquire data from csv, clean data, split data and teturn train, validate and test data'''
    
    df = acquire_data()
    
    df = clean_data(df)
    
    train, validate, test = train_val_test(df, seed=42)
    
    return train, validate, test
    

def get_correlation(df):
    '''takes in a dataframe and print correlation of qulity with other features'''
    
    correlation =df.corr()
    
    print(correlation['life_expectancy'].sort_values(ascending = False),'\n')  
    
    
  