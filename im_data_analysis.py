# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import time
import os
from os import system
from datetime import *
from sklearn.preprocessing import LabelEncoder
from sklearn import tree

file_path = 'dtree.png'
if os.path.isfile(file_path):
    os.remove(file_path)
    print('Removed existing output file "dtree.png"')

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

statusDef = {'Return' : 0, 'Finish' : 1, 'New' : 1, 'Cancel' : 0, 'Fail' : 0, 'Overdue' : 0, 'Shipping' : 1}
upload_2.StatusDef = [statusDef[item] for item in upload_2.StatusDef]
df_2 = pd.DataFrame(upload_2)

print('Successfully read in file : C:\PY\Data\91APP_OrderData.csv ...')

df_2['UnifiedUserId'] = le.fit_transform(df_2['UnifiedUserId'])
df_2['MemberId'] = le.fit_transform(df_2['MemberId'])
df_2['TradesGroupCode'] = le.fit_transform(df_2['TradesGroupCode'])
df_2['ChannelType'] = le.fit_transform(df_2['ChannelType'])
df_2['PaymentType'] = le.fit_transform(df_2['PaymentType'])
df_2['ShippingType'] = le.fit_transform(df_2['ShippingType'])
df_2['StatusDef'] = le.fit_transform(df_2['StatusDef'])

print('Successfully encoded dataframe ...')

df_2['TotalSalesAmount'],_ = pd.factorize(df_2['TotalSalesAmount'])
df_2['TotalPrice'],_ = pd.factorize(df_2['TotalPrice'])
df_2['TotalDiscount'],_ = pd.factorize(df_2['TotalDiscount'])
df_2['TotalPromotionDiscount'],_ = pd.factorize(df_2['TotalPromotionDiscount'])
df_2['StatusDef'],uniques = pd.factorize(df_2['StatusDef'])

x_train = df_2[['TotalSalesAmount', 'TotalPrice', 'TotalDiscount', 'TotalPromotionDiscount']]
y_train = df_2[['StatusDef']]

dtree = tree.DecisionTreeClassifier()
status_dtree = dtree.fit(x_train, y_train)

print('Successfully fits data into decision tree ...')

tree.export_graphviz(status_dtree, out_file = "tree.dot", max_depth = 6, feature_names = [str('TotalSalesAmount'), str('TotalPrice'), str('TotalDiscount'), str('TotalPromotionDiscount')], filled = True, proportion = True, rotate = True, rounded = True)
system("dot -Tpng tree.dot -o dtree.png")
os.remove('tree.dot')

print('Successfully exported dtree.png ...')