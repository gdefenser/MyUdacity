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
# read data-set of Pitching
df_pitching = filterDfByLastTenYears(pd.read_csv(folder_path+'Pitching.csv'))
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


def drawGrid(set1,set2,x_key,y_key,label_1,label_2):
    plt.plot(set1[x_key], set1[y_key], 'r', label=label_1)
    plt.plot(set2[x_key], set2[y_key], 'b', label=label_2)
    plt.legend(bbox_to_anchor=[0.4, 1])
    ax = plt.gca()
    ax.set_ylabel(y_key)
    ax.set_xlabel(x_key)
    plt.grid()


def calculateWinRate(df_teams_row):
    return float(df_teams_row.W) / float(df_teams_row.W+df_teams_row.L)

def calculateHitRate(df_lan_batting_row):
    if df_lan_batting_row.AB == 0:
        return float(0)
    else:
        return float(df_lan_batting_row.H) / float(df_lan_batting_row.AB)

df_batting['HitRate'] = df_batting.apply(calculateHitRate,axis=1)

df_lan_b = df_batting[df_batting['teamID']=='LAN']
df_lan_p = df_pitching[df_pitching['teamID']=='LAN']
df_lan_bp = df_lan_b.merge(df_lan_p,on=['yearID','teamID','playerID'],how='left')
df_lan_bp.fillna(0)

df_lan_allstar_bp = df_allstar.merge(df_lan_bp,on=['yearID','teamID','playerID'],how='inner')


df_analysis_lan_bp = df_lan_bp.groupby(['yearID'])['yearID','H_x','AB','HitRate','HR_x','SO_x','H_y']
df_analysis_lan_allstar_bp = df_lan_allstar_bp.groupby(['yearID'])[['yearID','H_x','AB','HitRate','HR_x','SO_x','H_y']].mean()

df_lan_allstar_2008 = df_allstar[(df_allstar['yearID']==2008) & (df_allstar['teamID']=='LAN')]
print df_lan_allstar_2008
df_lan_p_2008 = df_lan_p[(df_lan_p['yearID']==2008) & (df_lan_p['playerID']=='martiru01')]
print len(df_lan_p_2008)
#print len(df_lan_allstar_batting_pitching)

