# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
from __future__ import print_function, division

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import tensorflow as tf

from glob import glob
import os, sys

# %matplotlib inline
# -

# ##### Some setup data...

# +
team_long_names = """Atlanta Hawks
New Jersey Nets
Brooklyn Nets
Boston Celtics
Charlotte Hornets
Chicago Bulls
Cleveland Cavaliers
Dallas Mavericks
Denver Nuggets
Detroit Pistons
Golden State Warriors
Houston Rockets
Indiana Pacers
Los Angeles Clippers
Los Angeles Lakers
Memphis Grizzlies
Miami Heat
Milwaukee Bucks
Minnesota Timberwolves
New Orleans Pelicans
New York Knicks
Oklahoma City Thunder
Orlando Magic
Philadelphia 76ers
Phoenix Suns
Portland Trail Blazers
Sacramento Kings
San Antonio Spurs
Toronto Raptors
Utah Jazz
Washington Bullets
Washington Wizards
Seattle SuperSonics
Baltimore Bullets""".split('\n')

team_short_names = """ATL  NJN  BKN  BOS  CHA  CHI  CLE  DAL  DEN  DET  GSW  HOU  IND  LAC  LAL  MEM  MIA  MIL  MIN  NOP  NYK  OKC  ORL  PHI  PHX  POR  SAC  SAS  TOR  UTA  WAS WAS SEA BAL""".split()


long_to_short = dict(zip(team_long_names, team_short_names))
short_to_long = dict(zip(team_short_names, team_long_names))

# +
## how do we weight stats when calculating a players value?  larger number = more weight
stat_weights = {'PTS': 2.0, 
                'AST': 1.5,
                'BLK': 1.25,
                'TRB': 1.0,
                'ORB': 0.5, 
                'STL': 1.25}
base_stat_keys = list(stat_weights.keys())

for k in base_stat_keys:
    stat_weights[k + '_PER36'] = stat_weights[k] * 1.25
    
stat_keys = stat_weights.keys()
# -

column_renamer = {'Pos':'Position', 
                 'Tm':'Team', 
                 'G':'GamesPlayed', 
                 'GS':'GamesStarted',
                 'MP':'MinutesPlayed',
                 'PF':'Fouls'}


def parse_bball_ref_common_cols(df):
    df['PlayerName'] = df['Player'].apply(lambda x:  x.split('\\')[0])
    df['PlayerID'] = df['Player'].apply(lambda x:  x.split('\\')[1])
    
    df.drop(columns=[k for k in ['Rk', 'Player'] if k in df.keys()], inplace=True)
    df.rename(columns=column_renamer, inplace=True)
    return df


# + {"heading_collapsed": true, "cell_type": "markdown"}
# #### Playground

# + {"hidden": true}
df = yearly_stats[-1]
# df['Player'].apply(lambda x: x.split('\\')[0])
# df['PlayerID'] = df.apply(lambda x:  x['Player'].split('/')[1], axis=1)

# + {"hidden": true}
df.drop?

# + {"hidden": true}
more finals_stats.csv

# + {"hidden": true}
pd.read_csv?
# -

# #### Read info about teams that made it to the finals, players that are in the HOF, and MVP winners...

finals_team_data = pd.read_csv('finals_stats.csv', index_col='Year')
finals_team_data.dropna(axis='index', inplace=True)
finals_team_data['Champion'] = finals_team_data['Champion'].apply(lambda x: long_to_short[x])
finals_team_data['Runner-Up'] = finals_team_data['Runner-Up'].apply(lambda x: long_to_short[x])
finals_team_data.drop(columns=['Lg'], inplace=True)


hof = pd.read_table('hof_players.txt', delim_whitespace=True)
hof['Name'] = [fn + ' ' + ln for (fn, ln) in zip(hof_players['FirstName'], hof_players['LastName'])]
hof.drop(columns=['FirstName', 'LastName', 'Height(M)'], inplace=True)
hof_names = np.array(hof['Name'].values)

mvps = pd.read_csv('mvp_stats.csv')
mvps = parse_bball_ref_common_cols(mvps)
mvps['Year'] = mvps['Season'].apply(lambda x: int(x.split('-')[0]) + 1)
mvps.drop(columns=['Season','Team'], inplace=True)
mvps.set_index('Year', inplace=True)


def add_per_stats(df):
    for key in base_stat_keys:
        df[key + '_PER36'] = 36.0 * df[key] / df['MinutesPlayed']
    return df


def read_and_clean_yearly_stats(fname, year, veteran_ids, previous_rookie_ids):
    """
    parse a single year's stats into those i'm looking for
    
    also indicate whether a player is a rookie (0), second year (1), or veteran player (2)
    """
    df = parse_bball_ref_common_cols(pd.read_csv(fname))
    df = add_per_stats(df)
    
    def get_leaders(msk):
        leader_values = {}
        for key in stat_keys:
             leader_values[key] = df[key].loc[msk].max()
        return leader_values
    
    if year < 2019:
        champ = finals_team_data['Champion'][year]
        runnerup = finals_team_data['Runner-Up'][year]

        champ_players = df['Team'] == champ
        ru_players = df['Team'] == runnerup    
  
        champ_leaders = get_leaders(champ_players)
        ru_leaders = get_leaders(ru_players)
        
        mvpid = mvps['PlayerID'][year]
    else:
        champ = None
        runnerup = None
        mvpid = None

    league_leaders = get_leaders(np.ones(df.shape[0], dtype=bool))

    def calculate_player_value(row):
        if row['Team'] in [champ, runnerup]:
            if row['Team'] == champ:
                multiplier = 1.0
                leader_values = champ_leaders
            else:
                multiplier = 0.5
                leader_values = ru_leaders

            champ_value = (row['MinutesPlayed']/3000 +  row['GamesStarted']/82 + 0.33 * row['GamesPlayed']/82) * 0.5
            champ_value += sum(stat_weights[key] * row[key] / leader_values[key] for key in stat_keys)
            champ_value *= multiplier
        else:
            champ_value = 0
        
        league_value = sum(stat_weights[key] * row[key] / league_leaders[key] for key in stat_keys)
        if row['PlayerID'] == mvpid:
            league_value += 5
        return champ_value + league_value
        
    def set_veteran_status(pid):
        if pid in previous_rookie_ids:
            return 1
        elif pid in veteran_ids:
            return 2
        else:
            return 0
    
    df['YearlyPlayerValue'] = df.apply(calculate_player_value, axis=1)
    df['VeteranStatus'] = df['PlayerID'].apply(set_veteran_status)
    df['YoungPlayer'] = df['Age'].apply(lambda x:  x <= 23)
    
    # everyone who was a rookie last year will be a veteran next year
    next_veteran_ids = np.union1d(veteran_ids, previous_rookie_ids)
    rookie_ids = np.array(df['PlayerID'].loc[df['VeteranStatus']==0].values)
    
    return df, rookie_ids, next_veteran_ids


yearly_files = sorted(glob('yearly_player_stats/*.csv'))

# +
### figure out who's a rookie etc at the beginning of my time....
year_one_df = parse_bball_ref_common_cols(pd.read_csv(yearly_files[0]))
year_two_df = parse_bball_ref_common_cols(pd.read_csv(yearly_files[1]))

year_one_ids = np.unique(year_one_df['PlayerID'].values)
year_two_ids = np.unique(year_two_df['PlayerID'].values)

## if you're in both year 1 and year 2, you're a veteran by year 3
veteran_ids = np.intersect1d(year_one_ids, year_two_ids)

## if you're only in year 2, you're a second year player in year 3
previous_rookie_ids = np.setdiff1d(year_two_ids, year_one_ids)
# -

dataframes = {}
for fname in yearly_files[2:]:
    year = int(fname.split('_')[-1].split('.')[0])
    df, previous_rookie_ids, veteran_ids = read_and_clean_yearly_stats(
        fname, year, veteran_ids, previous_rookie_ids)
    
    dataframes[year] = df


