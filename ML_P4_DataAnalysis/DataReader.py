import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

folder_path = '/home/loveshadev/PycharmProjects/Udacity/ML_P4_DataAnalysis/'

#filter Data set
def filterDfByLastTenYears(df):
    return df[(df['yearID'] >= 2007) & (df['yearID'] <= 2017)]

#read data-set of All star
df_allstar = filterDfByLastTenYears(pd.read_csv(folder_path+'AllstarFull.csv'))
#read data-set of Batting
df_batting = filterDfByLastTenYears(pd.read_csv(folder_path+'Batting.csv'))
#read data-set of WSWin Teams
df_teams = filterDfByLastTenYears(pd.read_csv(folder_path+'Teams.csv'))
#df_teams = df_teams[df_teams['WSWin']=='Y']

#print len(df_allstar)
#print len(df_batting)
#print len(df_teams)

def printDiscrtption(df):
    print 'Minimum value by each year'
    print df.min()
    print '============================================'
    print 'Minimum value by each year'
    print df.max()
    print '============================================'
    print 'Mean value by each year'
    print df.mean()
    print '============================================'
    print 'Standard deviation by each year'
    print df.std()


def drawGrid(x1_set,y1_set,x2_set,y2_set,label_1,label_2):
    plt.plot(x1_set, y1_set, 'r', label=label_1)
    plt.plot(x2_set, y2_set, 'b', label=label_2)
    plt.legend(bbox_to_anchor=[0.4, 1])
    plt.grid()

def calculateWinRate(df_teams_row):
    return float(df_teams_row.W) / float(df_teams_row.W+df_teams_row.L)

df_analysis_allstar_batting = df_allstar.merge(df_batting,on=['yearID','teamID','playerID'],how='inner')

print len(df_analysis_allstar_batting)