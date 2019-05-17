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
# %reload_ext autoreload
# %autoreload 2

from basketball_reference_web_scraper import client
from basketball_reference_web_scraper.data import Team


import csv
import os
import random
import time
import copy
import glob
import shutil

user_agent_list = [   #Chrome
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.113 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.90 Safari/537.36',
    'Mozilla/5.0 (Windows NT 5.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.90 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.2; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.90 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/44.0.2403.157 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.113 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.87 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.87 Safari/537.36',
    #Firefox
    'Mozilla/4.0 (compatible; MSIE 9.0; Windows NT 6.1)',
    'Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; rv:11.0) like Gecko',
    'Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; WOW64; Trident/5.0)',
    'Mozilla/5.0 (Windows NT 6.1; Trident/7.0; rv:11.0) like Gecko',
    'Mozilla/5.0 (Windows NT 6.2; WOW64; Trident/7.0; rv:11.0) like Gecko',
    'Mozilla/5.0 (Windows NT 10.0; WOW64; Trident/7.0; rv:11.0) like Gecko',
    'Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.0; Trident/5.0)',
    'Mozilla/5.0 (Windows NT 6.3; WOW64; Trident/7.0; rv:11.0) like Gecko',
    'Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Trident/5.0)',
    'Mozilla/5.0 (Windows NT 6.1; Win64; x64; Trident/7.0; rv:11.0) like Gecko',
    'Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.1; WOW64; Trident/6.0)',
    'Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.1; Trident/6.0)',
    'Mozilla/4.0 (compatible; MSIE 8.0; Windows NT 5.1; Trident/4.0; .NET CLR 2.0.50727; .NET CLR 3.0.4506.2152; .NET CLR 3.5.30729)'
]

output_base = 'scraped/'

def wait_random_time(max_wait_seconds=22.25):
    ### wait some period of time and set a new user agent string
    seconds_to_wait = random.random()*max_wait_seconds
    time.sleep(seconds_to_wait)
    client.http_client.USER_AGENT = random.choice(user_agent_list)


# + {"code_folding": []}
def download_yearly_stats(year, return_download_tables=False):
    downloaded_tables = {}
    
    output_file_path = output_base+f'/stats_by_year/{year}_totals.csv'
    if not os.path.isfile(output_file_path):
        downloaded_tables['totals'] = client.players_season_totals(year, output_type='csv', 
                                     output_file_path=output_base+f'/stats_by_year/{year}_totals.csv')
        wait_random_time()
        
    output_file_path = output_base+f'/stats_by_year/{year}_advanced.csv'
    if not os.path.isfile(output_file_path):
        downloaded_tables['advanced'] = client.players_advanced_stats(year, output_type='csv', 
                                     output_file_path=output_base+f'/stats_by_year/{year}_advanced.csv')
        wait_random_time()
    if return_download_tables:
        return downloaded_tables

for year in range(1950, 2019):
    download_yearly_stats(year)
    print(f"Done with stats for {year}")
# -

for year in range(1950, 2019):
    if year == 1954:
        print("Skipping 1954")
        continue
        
    print(f"Starting on the {year} playoffs")
    output_directory = output_base + f'playoffs_by_series/{year}/'
    os.makedirs(output_directory, exist_ok=True)
    client.playoffs_series_in_one_year(year, output_directory=output_directory, output_type='csv')
    wait_random_time(60)

# + {"code_folding": []}
from basketball_reference_web_scraper import output
from basketball_reference_web_scraper.data import OutputWriteOption
import csv

output_write_option = OutputWriteOption("w")
def combine_series_by_round(year):
    directory = output_base + f'playoffs_by_series/{year}/'
    for table in ['basic', 'advanced']:
        if year < 1984 and table == 'advanced':
            continue
            
        files = glob.glob(directory+f'*_{table}.csv')
        file_lists = {}

        for fname in files:
            pround = fname.split('/')[-1][:-len('.csv')].split('_')[0]
            if pround in file_lists:
                file_lists[pround].append(fname)
            else:
                file_lists[pround] = [fname]

        for pround, files in file_lists.items():
            if len(files) == 1:
                continue

            files = sorted(files)
            if '_0' not in files[0]:
                input_file_path = files[0][:-len('.csv')] + '_0.csv'
                shutil.move(files[0], input_file_path)
                files[0] = input_file_path
            else:
                input_file_path = files[0]

            output_file_path = input_file_path[:-len(f'0_{table}.csv')] + f'{table}.csv'
            print("Combining:\n\t"+'\n\t'.join(files))

            all_rows = []
            for file_path in files:
                with open(file_path, 'r') as input_file:
                    CSVfile = csv.reader(input_file)
                    header = None
                    for row in CSVfile:
                        if header is None:
                            header = row
                        else:
                            all_rows.append({k: row[ii] for ii, k in enumerate(header)})
            print(f"Writing {len(all_rows)} total lines to {output_file_path}")
            output.playoff_stats_writer(all_rows, output_file_path, output_write_option, table)


# -

for year in range(1950, 2019):
    combine_series_by_round(year)

## had to redo 2018:
combine_series_by_round(2018)

# +
### doing all the playoff series
## deprecated because I wrote a function in client to do it with less hassle (though with overwriting)

# def read_playoff_series_list(file_path):
#     series_list = []
#     with open(file_path, 'r') as input_file:
#         CSVfile = csv.reader(input_file)
#         header = None
#         for row in CSVfile:
#             if header is None:
#                 header = row
#             else:
#                 output_dictionary = {}
#                 for ii, k in enumerate(header):
#                     if header[ii] in ['winning_team', 'losing_team']:
#                         output_dictionary[k] = Team(row[ii])
#                     else:
#                         output_dictionary[k] = row[ii]
#                 series_list.append(output_dictionary)
#     return series_list
        

# def add_unique_name_and_round(series_list):
#     series_count = {}
#     for series in series_list:
#         round_name = series['series_name']
#         finals = 'semi' in round_name.lower()
#         if True in [round_name.startswith(f'{x} ') for x in ['Western', 'Central', 'Eastern']]:
#             finals = False

#         if round_name in series_count:
#             unique_series_name = round_name + '_{}'.format(series_count[round_name])
#             series_count[round_name] = series_count[round_name] + 1
#         elif finals:
#             unique_series_name = copy.deepcopy(round_name)
#         else:
#             unique_series_name = round_name + '_0'
#             series_count[round_name] = 1
        
#         series['round_name'] = round_name
#         series['unique_series_name'] = unique_series_name
#     return series_list


# def download_playoff_series_in_a_year(year, return_download_tables=False):
#     schedule_file_path = output_base + f'playoff_schedules/{year}_playoffs.csv'

#     if os.path.isfile(schedule_file_path):
#         print("Reading schedule from "+schedule_file_path)
#         playoff_series_list = read_playoff_series_list(schedule_file_path)
#     else:
#         playoff_series_list = client.playoff_series_list(year, output_type='csv',
#                                                         output_file_path=schedule_file_path)
#         wait_random_time()
    
#     playoff_series_list = add_unique_name_and_round(playoff_series_list)

#     output_directory = output_base + f'playoffs_by_series/{year}/'
    
#     downloaded_tables = []
#     for series in playoff_series_list:
#         unique_name = series['unique_series_name']
#         round_name = series['round_name']
        
#         output_file_path_base = output_directory + unique_name
#         if not os.path.isfile(output_file_path_base+'_basic.csv'):
#             tables = client.playoff_series_stats(series, output_type='csv', 
#                                                 output_file_path=output_file_path_base)
#             print("Downloaded tables for {}".format(series['unique_series_name']))
#             downloaded_tables.append(tables)
#             wait_random_time()

#     if return_download_tables:
#         return downloaded_tables

# for year in range(1950, 2019):
#     print(f"Beginning series from the {year} Playoffs")
#     os.makedirs(output_base+f'playoffs_by_series/{year}/', exist_ok=True)
#     download_playoff_series_in_a_year(year)
#     print(f"Finished with series in the {year} Playoffs")
# -


