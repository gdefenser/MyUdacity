import numpy as np
import pandas as pd


folder_path = '/home/loveshadev/PycharmProjects/Udacity/ML_P4_DataAnalysis/'

#filter Data set
def filterDfByLastTenYears(df):
    return df[(df['yearID'] >= 2016) & (df['yearID'] <= 2017)]

#read data-set of All star
df_allstar = filterDfByLastTenYears(pd.read_csv(folder_path+'AllstarFull.csv'))
#read data-set of Batting
df_batting = filterDfByLastTenYears(pd.read_csv(folder_path+'Batting.csv'))
#read data-set of WSWin Teams
df_teams = filterDfByLastTenYears(pd.read_csv(folder_path+'Teams.csv'))
#df_teams = df_teams[df_teams['WSWin']=='Y']


df_merged_teams_batting = df_teams.merge(df_batting,on=['teamID'],how='inner')
df_merged_allstar_batting = df_allstar.merge(df_batting,on=['playerID'],how='left')
#print len(df_allstar)
#print len(df_batting)
#print len(df_teams)

#print len(df_merged_teams_batting)
#print len(df_merged_allstar_batting)