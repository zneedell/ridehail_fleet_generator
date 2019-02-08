#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 09:48:16 2019

@author: zack
"""

import pandas as pd
import numpy as np
import datetime
import seaborn as sns

fieldsA = ['RIDE_ID','started_on','completed_on','distance_travelled',
           'end_location_lat','end_location_long','surge_factor','start_location_lat',
           'end_location_long','rider_id','driver_reached_on']
datesA = ['started_on','completed_on','driver_reached_on']
fieldsB = ['RIDE_ID','driver_accepted_on','dispatched_on','total_fare', 'driving_time_to_rider','driving_distance_to_rider','status','driver_id']
datesB = ['driver_accepted_on','dispatched_on']
dfA = pd.read_csv('../Data/Rides_DataA.csv',parse_dates=datesA,usecols=fieldsA)
dfB = pd.read_csv('../Data/Rides_DataB.csv',parse_dates=datesB,usecols=fieldsB)
dfB['status'] = dfB['status'].astype('category')
df = dfA.join(dfB, on='RIDE_ID',lsuffix='_A', rsuffix='_B')

hour_shift = 0.1

df['busy_duration_hours'] = (df['completed_on'] - df['dispatched_on']).dt.seconds/(60*60)
df['date_mod'] = (df['driver_reached_on']+ datetime.timedelta(hours=-hour_shift)).dt.date
df['day_of_week'] = (df['driver_reached_on']+ datetime.timedelta(hours=-hour_shift)).dt.dayofweek
df['hour_of_day'] = df['dispatched_on'].dt.hour + df['dispatched_on'].dt.minute/60
df.loc[df['hour_of_day'] < hour_shift,'hour_of_day'] = df['hour_of_day'] + 24

df = df.query('day_of_week < 4 and busy_duration_hours> 0.05 and busy_duration_hours < 5')

# %%
smaller = df[100000:] # to ignore weird stuff when it was just getting started

max_gap = 1.0 # in hours, gap between a trip end and new dispatch that defines a "break"

def count_trips(x):
    return np.sum(x.status == "b'DISPATCHED'")


def get_biggest_gap(driver_day):
    if np.size(driver_day,0) > 1:
        start_times = driver_day.hour_of_day.values
        end_times = start_times + driver_day.busy_duration_hours.values
        
        gaps = start_times[1:] - end_times[:-1]
        gap = np.max(gaps)
        if gap < 0:
            return np.nan
        else:
            return gap
    else:
        return np.nan

def get_shifts(driver_day):
    if np.size(driver_day,0) > 1:
        start_times = driver_day.hour_of_day.values
        end_times = start_times + driver_day.busy_duration_hours.values
        gaps = start_times[1:] - end_times[:-1]
        return np.sum(gaps > max_gap) + 1
    else:
        return 1
    
def get_shift_details(driver_day):
    driver_day = driver_day.sort_values(by='hour_of_day')
    start_times = driver_day.hour_of_day.values
    end_times = start_times + driver_day.busy_duration_hours.values
    if start_times[0] >= end_times[0]:
        return {'first_shift_start':np.nan,
                'first_shift_end':np.nan,
                'second_shift_start':np.nan,
                'second_shift_end':np.nan,
                'n_shifts':np.nan}
    elif np.size(driver_day,0) > 1:
        gaps = start_times[1:] - end_times[:-1]
        gap = np.max(gaps)
        gapind = np.argmax(gaps)
        first_shift_start = start_times[0]
        first_shift_end = end_times[gapind]
        if first_shift_end - first_shift_start > 12:
            return {'first_shift_start':np.nan,
                'first_shift_end':np.nan,
                'second_shift_start':np.nan,
                'second_shift_end':np.nan,
                'n_shifts':np.nan}
        elif gap > max_gap:
            first_shift_start = start_times[0]
            first_shift_end = end_times[gapind]
            second_shift_start = start_times[gapind+1]
            second_shift_end = end_times[-1]
            return {'first_shift_start':first_shift_start,
                'first_shift_end':first_shift_end,
                'second_shift_start':second_shift_start,
                'second_shift_end':second_shift_end,
                'n_shifts':np.sum(gaps > 1.5) + 1}
        else:
            return {'first_shift_start':start_times[0],
                'first_shift_end':end_times[-1],
                'second_shift_start':-1,'second_shift_end':-1,'n_shifts':1}    
    elif np.size(driver_day,0) == 1:
        return {'first_shift_start':driver_day.hour_of_day.values[0],
                'first_shift_end':driver_day.hour_of_day.values[0] + driver_day.busy_duration_hours.values[0],
                'second_shift_start':-1,'second_shift_end':-1,'n_shifts':1}
    else:
        return {'first_shift_start':np.nan,'first_shift_end':np.nan,'second_shift_start':np.nan,'second_shift_end':np.nan,'n_shifts':np.nan}

    
counts = smaller.groupby(['date_mod','driver_id']).apply(count_trips)
biggest_gap = smaller.groupby(['date_mod','driver_id']).apply(get_biggest_gap)
shift_info = smaller.groupby(['date_mod','driver_id']).apply(get_shift_details).apply(pd.Series)

combined = pd.concat([counts, shift_info.first_shift_start,shift_info.first_shift_end,shift_info.second_shift_start,shift_info.second_shift_end,shift_info.n_shifts],axis=1,
                     names=['n_trips','first start','first end','second start','second end', 'n shifts'], 
                     keys=['n_trips','first start','first end','second start','second end', 'n shifts']).dropna()

sns.jointplot("first start", "second start", data=combined[combined['n shifts']==2],kind="hex", space=0, color="g")

combined.to_pickle("../Data/austin_output.pkl")