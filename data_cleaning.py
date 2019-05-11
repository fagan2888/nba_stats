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

# + {"code_folding": [0, 35]}
team_long_names = [k.strip() for k in """Atlanta Hawks
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
                    Baltimore Bullets""".split('\n')]

team_short_names = """ATL  
                    NJN  
                    BKN  
                    BOS  
                    CHA  
                    CHI  
                    CLE  
                    DAL  
                    DEN  
                    DET  
                    GSW  
                    HOU  
                    IND  
                    LAC  
                    LAL  
                    MEM  
                    MIA  
                    MIL  
                    MIN  
                    NOP  
                    NYK  
                    OKC  
                    ORL  
                    PHI  
                    PHX  
                    POR  
                    SAC  
                    SAS  
                    TOR  
                    UTA  
                    WAS 
                    WAS
                    SEA 
                    BAL""".split()

long_to_short = dict(zip(team_long_names, team_short_names))
short_to_long = dict(zip(team_short_names, team_long_names))

# +
## how do we weight stats when calculating a players value?  larger number = more weight
stat_weights = {'PTS': 2.0, 
                'AST': 1.5,
                'BLK': 1.25,
                'TRB': 1.0,
                'ORB': 0.5, 
                'STL': 1.25,
                'TOV': -1.0}
base_stat_keys = list(stat_weights.keys())

for k in base_stat_keys:
    stat_weights[k + '_PER36'] = stat_weights[k] * 1.25
    
stat_keys = stat_weights.keys()

champ_multiplier = 0.66
ru_multiplier = 0.33
playoff_values = dict([(str(i), 0.1*np.sqrt(i)) for i in range(1, 5)])

mvp_value = 7.5
finals_mvp_value = 4
all_star_value = 1.5

# +
column_renamer = {'Pos':'Position', 
                 'Tm':'Team', 
                 'G':'GamesPlayed', 
                 'GS':'GamesStarted',
                 'MP':'MinutesPlayed',
                 'PF':'Fouls',
                 'Starters':'Player'}

def parse_bball_ref_common_cols(df):
    df.rename(columns=column_renamer, inplace=True)
    df['PlayerName'] = df['Player'].apply(lambda x:  x.split('\\')[0])
    df['PlayerID'] = df['Player'].apply(lambda x:  x.split('\\')[1])
    
    df.drop(columns=[k for k in ['Rk', 'Player'] if k in df.keys()], inplace=True)
    return df

def add_per_stats(df):
    for key in base_stat_keys:
        df[key + '_PER36'] = 36.0 * df[key] / df['MinutesPlayed']
    return df


# -

# #### Read info about teams that made it to the finals, players that are in the HOF, MVP winners, and players that made All-Star teams...

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

all_star_pids = {}
all_star_files = sorted(glob('all_stars/*.csv'))
for fname in all_star_files:
    year = int(fname.split('/')[-1].split('.')[0])
    adf = parse_bball_ref_common_cols(pd.read_csv(fname))
    ## really don't care about any of the stats in the all-star game
    ## I just want to know if they made it or not
    all_star_pids[year] = adf['PlayerID'].values


# #### Read and parse playoff stats for reference later:

# +
def read_and_clean_playoff_year(year):
    dataframes = {}
    for playoff_round in '1234':
        fname = 'playoff_player_stats/{}/round{}.csv'.format(year, playoff_round)
        dataframes[playoff_round] = read_and_clean_playoff_round_stats(fname)
    return dataframes

def read_and_clean_playoff_round_stats(fname):
    if not os.path.isfile(fname):
        return pd.DataFrame(columns=['PlayerID'])
    
    df = pd.read_csv(fname, header=[0, 1])
    new = [' '.join(col).strip() for col in pdf.columns.values]
    for ii, n in enumerate(new):
        if n.startswith('Unnamed'):
            new[ii] = n.split()[-1]
        elif n.startswith('Per Game'):
            new[ii] = n.replace(' ','')
        else:
            new[ii] = n.split()[-1]
    df.columns = new
    
    df = add_per_stats(parse_bball_ref_common_cols(df))
    return df     


# -

playoff_stats_by_year = {}
playoff_years = range(1988, 2019)
for year in playoff_years:
    playoff_stats_by_year[year] = read_and_clean_playoff_year(year)


# +
def calculate_playoff_value(row, year):
    playoff_stats_by_round = playoff_stats_by_year[year]
    pid = row['PlayerID']
    for playoff_round in '1234':
        # 1 = first round
        # 2 = conference semifinals
        # 3 = east/west finals
        # 4 = nba finals
        
        ##TODO continue here


# -

# #### Now read and parse yearly stats
#
# Also calculate player "values" based on both volume and PER stats in the regular season and in the playoffs, with bonuses for contributing to a team that makes the finals, being an all star, or being the MVP or finals MVP.  Also mark who's a young player and a rookie & second year player each year based on their presence in the stats the previous year -- these are going to the players that I look to predict their growth later.

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
        
        fmpv = finals_team_data['Finals MVP'][year]
        mvpid = mvps['PlayerID'][year]
    else:
        champ = None
        runnerup = None
        mvpid = None

    all_stars = all_star_pids[year]   
    league_leaders = get_leaders(np.ones(df.shape[0], dtype=bool))

    found_fmvp = False
    def calculate_player_value(row):          
        if row['Team'] in [champ, runnerup]:
            ## did you contribute to a team that made it to the finals?
            champ_value =  0.5 * (row['MinutesPlayed']/3000 +  
                           row['GamesStarted']/82 + 
                           0.33 * row['GamesPlayed']/82)
            
            if row['Team'] == champ:
                multiplier = champ_multiplier
                leader_values = champ_leaders                
            else:
                multiplier = ru_multiplier
                leader_values = ru_leaders

            pname = row['PlayerName'].rsplit(num=1)
            pname = pname[0][0]+'. '+pname[1]
            if pname == fmvp:
                if found_fmvp:
                    print("!! -- found two Finals MVPs in {}".format(year))
                champ_value += finals_mvp_value
                found_fmvp = True
                
            champ_value += sum(stat_weights[key] * row[key] / leader_values[key] for key in stat_keys)
            champ_value *= multiplier
        else:
            champ_value = 0
            
        playoff_value = calculate_playoff_value(row, year)
        
        league_value = sum(stat_weights[key] * row[key] / league_leaders[key] for key in stat_keys)
        if row['PlayerID'] == mvpid:
            league_value += mvp_value
        if row['PlayerID'] in all_stars:
            league_value += all_star_value
        return champ_value + league_value + playoff_value
        
    def set_veteran_status(pid):
        if pid in previous_rookie_ids:
            return 1
        elif pid in veteran_ids:
            return 2
        else:
            return 0
    
    ## drop the "total" values of players now (not earlier, since we want 
    ## to use total stats to normalize our value added above)
    ## will sum-up player values later, 
    ## but a player gets value from their contribution to each team
    df = df[df['Team'] != 'TOT']
    
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


