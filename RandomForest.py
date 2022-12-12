# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import os
from os import system
from datetime import datetime, date
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def rem_time(d):
    s = ''
    s = str(d.year) + '/' + str(d.month) + '/' + str(d.day)
    return s

def string_to_days_from_now(date_string) :
	try :
		date_time = datetime.strptime(date_string, '%Y/%m/%d %H:%M')
		# date_time = datetime.strptime(rem_time(date_time), '%Y/%m/%d')
		# now = datetime(2022, 12, 10)
		# subtract = now - date_time
		return date_time.hour
	except :
		return 0

def string_to_days_from_now2(date_string) :
	try : 
		date_time = datetime.strptime(date_string, '%Y/%m/%d')
		now = datetime(2022, 12, 10)
		subtract = now - date_time
		age_ = float(subtract.days / 365)
		return age_
	except : 
		return 0

def string_to_days_from_now3(date_string) :
	try : 
		date_time = datetime.strptime(date_string, '%Y/%m/%d')
		now = datetime(2022, 12, 10)
		subtract = now - date_time
		return subtract.days
	except : 
		return 0

def convert_channel_type(_string) : 
	if _string == 'OfficialECom' : 
		return 0
	elif _string == 'Mall': 
		return 1
	elif _string == 'Pos' : 
		return 2
	elif _string == 'LocationWizard' : 
		return 3
	else : 
		return 4

def convert_channel_detail(_string) : 
	if _string == 'DesktopOfficialWeb' : 
		return 0
	elif _string == 'MobileWeb': 
		return 1
	elif _string == 'iOSApp' : 
		return 2
	elif _string == 'AndroidApp' : 
		return 3
	else : 
		return 4

def convert_payment_type(_string) : 
		types = { 'CreditCardOnce' : 0, 'CreditCardInstallment' : 1, 'ATM' : 2, 'Family' : 3, 'HiLife' : 4, 'SevenEleven' : 5, 'ApplePay' : 6, 'LinePay' : 7, 'GooglePay' : 8, 'JKOPay' : 9, 'Aftee' : 10, 'EasyWallet' : 11, '0' : 12}
		for i in types : 
			if _string == i: 
				return types[i]

def convert_shipping_type(_string) : 
		types = { 'Family' : 0, 'FamilyPickup' : 1, 'HiLife' : 2, 'Family' : 3, 'HiLife' : 4, 'HiLifePickup' : 5, 'SevenEleven' : 6, 'SevenElevenPickup' : 7, 'Home' : 8, 'Oversea' : 9, '0' : 10}
		for i in types : 
			if _string == i: 
				return types[i]

def convert_register_source_type(_string) : 
		types = { 'AndroidApp' : 0, 'iOSApp' : 1, 'Web' : 2, 'Store' : 3, 'LocationWizard' : 4, '0' : 5}
		for i in types : 
			if _string == i: 
				return types[i]

upload = pd.read_csv(r'C:\PY\Data\91APP_CombinedData.csv', low_memory = False)
upload = upload.fillna('0')

statusDef = {'Return' : 0, 'Finish' : 1, 'New' : 1, 'Cancel' : 0, 'Fail' : 0, 'Overdue' : 0, 'Shipping' : 1}
upload.StatusDef = [statusDef[item] for item in upload.StatusDef]

df = pd.DataFrame(upload)

print('Successfully read in file : C:\PY\Data\91APP_CombinedData.csv ...')

df = df.drop('UnifiedUserId', axis = 1)
df = df.drop('MemberId', axis = 1)
df = df.drop('TradesGroupCode', axis = 1)
df = df.drop('APPRefereeId', axis = 1)
df = df.drop('APPRefereeLocationId', axis = 1)
df = df.drop('Gender', axis = 1)

print('Successfully dropped columns ...')

df['OrderDateTime'] = df['OrderDateTime'].map(lambda x : string_to_days_from_now(str(x)))
df['Birthday'] = df['Birthday'].map(lambda x : string_to_days_from_now2(str(x)))
df['FirstAppOpenDateTime'] = df['FirstAppOpenDateTime'].map(lambda x : string_to_days_from_now3(str(x)))
df['LastAppOpenDateTime'] = df['LastAppOpenDateTime'].map(lambda x : string_to_days_from_now3(str(x)))
df['RegisterDateTime'] = df['RegisterDateTime'].map(lambda x : string_to_days_from_now3(str(x)))
df['ChannelDetail'] = df['ChannelDetail'].map(lambda x : convert_channel_detail(str(x)))
df['PaymentType'] = df['PaymentType'].map(lambda x : convert_payment_type(str(x)))
df['ChannelType'] = df['ChannelType'].map(lambda x : convert_channel_type(str(x)))
df['ShippingType'] = df['ShippingType'].map(lambda x : convert_shipping_type(str(x)))
df['RegisterSourceTypeDef'] = df['RegisterSourceTypeDef'].map(lambda x : convert_register_source_type(str(x)))
df['AppUsingDuration'] = df['FirstAppOpenDateTime'] - df['LastAppOpenDateTime']

print('Successfully converted data ...')

le = LabelEncoder()

df['IsAppInstalled'] = le.fit_transform(df['IsAppInstalled'])
df['IsEnableEmail'] = le.fit_transform(df['IsEnableEmail'])
df['IsEnablePushNotification'] = le.fit_transform(df['IsEnablePushNotification'])
df['IsEnableShortMessage'] = le.fit_transform(df['IsEnableShortMessage'])
df['CountryAliasCode'] = le.fit_transform(df['CountryAliasCode'])

print('Successfully encoded dataframe ...')

df['Birthday'] = df['Birthday'].replace(0, np.NaN)
mean = df['Birthday'].mean()
std = df['Birthday'].std()
is_null = df['Birthday'].isnull().sum()
rand_age = np.random.uniform(mean - std, mean + std, size = is_null)
age_slice = df['Birthday'].copy()
age_slice[np.isnan(age_slice)] = rand_age
df['Birthday'] = age_slice

print(df['OrderDateTime'])

print('Successfully processed birthday')

lens = len(df.index)

x_train = df.drop('StatusDef', axis = 1)
y_train = df['StatusDef']

x_train = x_train.drop(df.index[int(round(2 * lens / 3, 0)) : lens])
y_train = y_train.drop(df.index[int(round(2 * lens / 3, 0)) : lens])
x_test = df.drop(df.index[ : int(round(2 * lens / 3, 0))])
y_test = x_test['StatusDef']
x_test = x_test.drop('StatusDef', axis = 1)

print('Successfully splitted train data & test data ...')

randomForestModel = RandomForestClassifier(n_estimators = 100, criterion = 'gini')
randomForestModel.fit(x_train, y_train)

print('Model Fitted ...')

pred = randomForestModel.predict(x_train)

print(round(randomForestModel.score(x_train, y_train) * 100, 2))
print(round(randomForestModel.score(x_test, y_test) * 100, 2))

features = list(df.columns.values)
features.remove("StatusDef")

importances = list(randomForestModel.feature_importances_)

print("Creating figure ...")
fig = plt.figure(figsize=(42,14))
plt.rcParams.update({"font.size":24})
plt.barh(features, importances, color='blue')
plt.xlabel("Features", fontsize = 32)
plt.ylabel("Importance", fontsize = 32)
plt.title("Feature importance", fontsize = 48)
plt.savefig("importance.png")