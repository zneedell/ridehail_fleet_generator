#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 10:40:09 2019

@author: zack
"""

import pandas as pd


rideHailFleet = pd.read_csv("../Data/rideHailFleet.csv")
combined = pd.read_pickle("../Data/austin_output.pkl")


sampled = combined.sample(n=len(rideHailFleet.index),replace=True)
sampled['shifts'] = ""
hours_to_seconds = 60*60

def add_shift(row):
    out_str = "{" + str(int(row['first start']*hours_to_seconds))
    out_str += ":" + str(int(row['first end']*hours_to_seconds))
    if row['n shifts'] > 1:
        out_str += "};{" + str(int(row['second start']*hours_to_seconds))
        out_str += ":" + str(int(row['second end']*hours_to_seconds))
    return out_str + "}"

new_shifts = sampled.transform(add_shift,axis=1).tolist()

rideHailFleet[' shifts'] = new_shifts
rideHailFleet.to_csv("../Data/rideHailFleet_new.csv",index=False)