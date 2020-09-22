# coding: utf-8

from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score

import pandas as pd
import numpy as np
import time
pd.set_option('display.max_row', 5000)


total_data = 129

# not exist file
not_exist_file = [30, 44, 69, 97]


# OSA re-labelling
def relabelling(data):
    data = data.replace('3', 'apnea')
    data = data.replace('4', 'apnea')
    data = data.replace('0', 'normal')

    data = data.replace('hypopnea', 'apnea')
    data = data.replace('all apnea', 'apnea')
    data = data.replace('all hypopnea', 'apnea')
    data = data.replace('mix apnea', 'normal')
    data = data.replace('mix hypopnea', 'normal')
    return data


# sleep train
def sleep_train(target_list_):
    global not_exist_file
    global total_data
    sleep_ecg_train = pd.DataFrame()
    sleep_spo2_train = pd.DataFrame()
    sleep_train_ = pd.DataFrame()
    for x in range(totla_data):
        if x in not_exist_file:
            continue
        elif x in target_list_:
            continue
        else:
            sleep_ecg_train = pd.concat([sleep_ecg_train, pd.read_csv('30sec_ECG%d.csv' % x)])
            sleep_spo2_train = pd.concat([sleep_spo2_train, pd.read_csv('30sec_spo2%d.csv' % x)])
            sleep_ecg_train = sleep_ecg_train.iloc[:, :-2]
            sleep_spo2_train = sleep_spo2_train.iloc[:, :]
            sleep_train_ = pd.concat([sleep_ecg_train, sleep_spo2_train], axis=1, sort=False)

    sleep_train_['SLEEPSTAGE2'] = np.where(sleep_train_['SLEEP_STAGE'] == 'W', 'wake', 'sleep')

    sleep_train_x = sleep_train_.iloc[:, :-3]
    sleep_train_y = sleep_train_.iloc[:, -1]

    sleep_rf = RandomForestClassifier(n_estimators=500)
    sleep_rf.fit(sleep_train_x, sleep_train_y)

    return sleep_rf


# sleep test
def sleep_test(sleep_rf, target):
    sleep_ecg_test = pd.read_csv('30sec_ECG%s.csv' % target)
    sleep_spo2_test = pd.read_csv('30sec_spo2%s.csv' % target)
    sleep_ecg_test = sleep_ecg_test.iloc[:, :-2]
    sleep_spo2_test = sleep_spo2_test.iloc[:, :]
    sleep_test_ = pd.concat([sleep_ecg_test, sleep_spo2_test], axis=1, sort=False)
    sleep_test_['SLEEPSTAGE2'] = np.where(sleep_test_['SLEEP_STAGE'] == 'W', 'wake', 'sleep')

    sleep_x_test = sleep_test_.iloc[:, :-3]
    sleep_y_test = sleep_test_.iloc[:, -1]

    sleep_pred_test = sleep_rf.predict(sleep_x_test)

    print('sleep stage classification')
    print(confusion_matrix(sleep_y_test, sleep_pred_test))
    print(classification_report(sleep_y_test, sleep_pred_test))

    return Counter(sleep_pred_test)['sleep'] / 2 / 60


# osa train 60sec
def osa_train_clf60(target_list_):
    global not_exist_file
    global total_data
    osa_train = pd.DataFrame()
    for x in range(total_data):
        if x in not_exist_file:
            continue
        elif x in target_list_:
            continue
        else:
            osa_ecg_train = pd.read_csv('60sec_ECG%d.csv' % x)
            osa_spo2_train = pd.read_csv('60sec_spo2%d.csv' % x)
            osa_ecg_train = osa_ecg_train.iloc[:, :-2]
            osa_spo2_train = osa_spo2_train.iloc[:, :]
            osa_train_concat = pd.concat([osa_ecg_train, osa_spo2_train], axis=1, sort=False)
            osa_train_concat['num'] = x
            osa_train = pd.concat([osa_train, osa_train_concat])

    osa_train60_ = relabelling(osa_train)

    osa_train_x = osa_train60_.iloc[:, :-3]
    osa_train_y = osa_train60_.iloc[:, -3]

    osa_rf60 = RandomForestClassifier(n_estimators=500)
    osa_rf60.fit(osa_train_x, osa_train_y)

    return osa_rf60, osa_train60_


# osa train 20sec
def osa_train_clf20(osa_train60_):
    # 60 to 20
    osa_train2 = osa_train60_[osa_train60_['OSA'] == 'apnea']

    # osa 20sec train
    apnea_list = osa_train2[osa_train2['OSA'] == 'apnea'][['num']]
    file_num = list(apnea_list['num'])
    index_num = list(apnea_list.index)
    num = pd.DataFrame(columns=['file', 'index'])
    num['file'] = file_num
    num['index'] = index_num

    osa_train = pd.DataFrame()
    for x in (num['file'].unique()):
        osa_ecg_train = pd.read_csv('20sec_ECG%d.csv' % x)
        osa_spo2_train = pd.read_csv('20sec_spo2data%d.csv' % x)
        osa_ecg_train = osa_ecg_train.iloc[:, :-2]
        osa_spo2_train = osa_spo2_train.iloc[:, :]
        osa_concat = pd.concat([osa_ecg_train, osa_spo2_train], axis=1, sort=False)
        for y in (num[num['file'] == x]['index']):
            osa_train = pd.concat([osa_train, osa_concat.iloc[y * 3:y * 3 + 3, :]])

    osa_train = relabelling(osa_train)

    osa_train_x = osa_train.iloc[:, :-2]
    osa_train_y = osa_train.iloc[:, -2]

    osa_rf20 = RandomForestClassifier(n_estimators=500)
    osa_rf20.fit(osa_train_x, osa_train_y)

    return osa_rf20


# osa test 60sec
def osa_test60(osa_rf60, target):
    osa_ecg_test = pd.read_csv('60sec_ECG%d.csv' % target)
    osa_spo2_test = pd.read_csv('60sec_spo2%d.csv' % target)
    osa_ecg_test = osa_ecg_test.iloc[:, :-2]
    osa_spo2_test = osa_spo2_test.iloc[:, :]
    osa_test_ = pd.concat([osa_ecg_test, osa_spo2_test], axis=1, sort=False)

    osa_test_ = relabelling(osa_test_)

    osa_x_test = osa_test_.iloc[:, :-2]
    osa_y_test = osa_test_.iloc[:, -2]

    osa_pred_test_ = osa_rf60.predict(osa_x_test)

    print('osa classification 60sec')
    print(confusion_matrix(osa_y_test, osa_pred_test_))
    print(classification_report(osa_y_test, osa_pred_test_))

    return osa_test_, osa_pred_test_


# osa test 20sec & calculate osa count
def osa_test20(osa_rf20, osa_test_, osa_pred_test_, target):
    osa_df = pd.DataFrame(index=range(len(osa_test_) * 3), columns=['osa'])
    osa_df['osa'] = 'normal'

    osa_test_20 = osa_test_[osa_pred_test_ == 'apnea']
    index20 = osa_test_20.index
    ecg_20_x = pd.read_csv('20sec_ECG%d.csv' % target)
    spo2_20_x = pd.read_csv('20sec_spo2%d.csv' % target)
    osa_20_target = pd.concat([ecg_20_x.iloc[:, :-2], spo2_20_x], axis=1, sort=False)

    osa_test_20_x = pd.DataFrame()
    for k in index20:
        osa_test_20_x = pd.concat([osa_test_20_x, osa_20_target.iloc[k * 3:k * 3 + 3, :]])

    osa_test_20 = relabelling(osa_test_20_x)

    osa_x_test_20 = osa_test_20.iloc[:, :-2]
    osa_y_test_20 = osa_test_20.iloc[:, -2]

    osa_pred_test_20 = osa_rf20.predict(osa_x_test_20)

    print('osa classification 20sec')
    print(confusion_matrix(osa_y_test_20, osa_pred_test_20))
    print(classification_report(osa_y_test_20, osa_pred_test_20))

    # ahi count calculation
    pred_index = osa_test_20[osa_pred_test_20 == 'apnea'].index

    for z in pred_index:
        osa_df.iloc[z, 0] = 'apnea'

    if osa_df.iloc[-1, 0] == 'apnea':
        count = 1
    else:
        count = 0
    for z in range(len(osa_df) - 1):
        if osa_df.iloc[z, 0] == 'apnea' and osa_df.iloc[z + 1, 0] == 'normal':
            count += 1

    return count


# split by 10
target_list = []
for x in range(total_data // 10 + 1):
    term = []
    for y in range(10 * x, 10 * (x + 1)):
        term.append(y)
    target_list.append(term)


# sleep stage classification
for x in target_list:
    rf = sleep_train(x)
    for y in x:
        try:
            result.loc[y, 'pred_sleep'] = sleep_test(rf, y)
        except:
            continue


# osa classification and osa count
for x in target_list:
    rf60, osa_train60 = osa_train_clf60(x)
    rf20 = osa_train_clf20(osa_train60)
    for y in x:
        try:
            osa_test, osa_pred_test = osa_test60(rf60, y)
            result.loc[y, 'pred_cnt'] = osa_test20(rf20, osa_test, osa_pred_test, y)
        except:
            continue


# calculate ahi
result['pred_ahi'] = result['pred_cnt'] / result['pred_sleep']
