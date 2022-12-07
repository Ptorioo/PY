# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import time
import os
from os import system
from datetime import *
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

le = LabelEncoder()

'''
upload_1 = pd.read_csv(r'C:\PY\Data\91APP_MemberData.csv', low_memory = False)

df_1 = pd.DataFrame(upload_1)
df_1['UnifiedUserId'] = le.fit_transform(df_1['UnifiedUserId'])
df_1['MemberId'] = le.fit_transform(df_1['MemberId'])
df_1['Gender'] = le.fit_transform(df_1['Gender'])
df_1['RegisterSourceTypeDef'] = le.fit_transform(df_1['RegisterSourceTypeDef'])
df_1['APPRefereeId'] = le.fit_transform(df_1['APPRefereeId'])
df_1['APPRefereeLocationId'] = le.fit_transform(df_1['APPRefereeLocationId'])
df_1['CountryAliasCode'] = le.fit_transform(df_1['CountryAliasCode'])
'''

upload_2 = pd.read_csv(r'C:\PY\Data\91APP_OrderData.csv', low_memory = False)

upload_2 = upload_2.fillna('0')

statusDef = {'Return' : 0, 'Finish' : 1, 'New' : 1, 'Cancel' : 0, 'Fail' : 0, 'Overdue' : 0, 'Shipping' : 1}
upload_2.StatusDef = [statusDef[item] for item in upload_2.StatusDef]
df_2 = pd.DataFrame(upload_2)

print('Successfully read in file : C:\PY\Data\91APP_OrderData.csv ...')

df_2 = df_2.drop('UnifiedUserId', axis = 1)
df_2 = df_2.drop('MemberId', axis = 1)
df_2 = df_2.drop('TradesGroupCode', axis = 1)
df_2 = df_2.drop('OrderDateTime', axis = 1)

print('Successfully dropped columns ...')

df_2['ChannelType'] = le.fit_transform(df_2['ChannelType'])
df_2['ChannelDetail'] = le.fit_transform(df_2['ChannelDetail'])
df_2['PaymentType'] = le.fit_transform(df_2['PaymentType'])
df_2['ShippingType'] = le.fit_transform(df_2['ShippingType'])
df_2['StatusDef'] = le.fit_transform(df_2['StatusDef'])

print('Successfully encoded dataframe ...')

lens = len(df_2.index)

x_train = df_2.drop('StatusDef', axis = 1)
y_train = df_2['StatusDef']

x_train = x_train.drop(df_2.index[int(round(lens/5, 0)) : lens])
y_train = y_train.drop(df_2.index[int(round(lens/5, 0)) : lens])
x_test = df_2.drop(df_2.index[ : int(round(lens/5, 0))])
x_test = x_test.drop('StatusDef', axis = 1)

print('Successfully splitted train data & test data ...')

# The rest is garbage !!!
#---------------------------------------------------------------
#
# sc = StandardScaler()
# x_train = sc.fit_transform(x_train)
# x_test = sc.fit_transform(x_test)
# ^^ Maybe useful but I don't know... ?

# classifier = LinearSVC()
# classifier.fit(x_train, y_train)
#
# print('Model fitted ...')
#
# y_pred = classifier.predict(x_test)
#
# classifier.score(x_train, y_train)
# score = round(classifier.score(x_train, y_train) * 100, 2)
# print(score)
