import numpy as np

def xgb_tree(x, num_booster):
    if num_booster == 0:
        state = 0
        if state == 0:
            state = (1 if x['Pclass']<3 or np.isnan(x['Pclass'])  else 2)
            if state == 1:
                state = (3 if x['Fare']<13.6458502 or np.isnan(x['Fare'])  else 4)
                if state == 3:
                    return -0.0693877563
                if state == 4:
                    state = (9 if x['Age']<42.5 else 10)
                    if state == 9:
                        return 0.0865979418
                    if state == 10:
                        return -0.00666666729
            if state == 2:
                state = (5 if x['Age']<6.5 else 6)
                if state == 5:
                    state = (11 if x['SibSp']<3 or np.isnan(x['SibSp'])  else 12)
                    if state == 11:
                        return 0.140000001
                    if state == 12:
                        return -0.111111119
                if state == 6:
                    state = (13 if x['Age']<32.5 or np.isnan(x['Age'])  else 14)
                    if state == 13:
                        return -0.0997389033
                    if state == 14:
                        return -0.153488383
    elif num_booster == 1:
        state = 0
        if state == 0:
            state = (1 if x['Pclass']<3 or np.isnan(x['Pclass'])  else 2)
            if state == 1:
                state = (3 if x['Fare']<52.2770996 or np.isnan(x['Fare'])  else 4)
                if state == 3:
                    state = (7 if x['Parch']<1 or np.isnan(x['Parch'])  else 8)
                    if state == 7:
                        return -0.0406082124
                    if state == 8:
                        return 0.093497619
                if state == 4:
                    state = (9 if x['Pclass']<2 or np.isnan(x['Pclass'])  else 10)
                    if state == 9:
                        return 0.094897598
                    if state == 10:
                        return -0.0592660122
            if state == 2:
                state = (5 if x['Age']<6.5 else 6)
                if state == 5:
                    state = (11 if x['SibSp']<3 or np.isnan(x['SibSp'])  else 12)
                    if state == 11:
                        return 0.129323557
                    if state == 12:
                        return -0.102724113
                if state == 6:
                    state = (13 if x['Age']<32.5 or np.isnan(x['Age'])  else 14)
                    if state == 13:
                        return -0.0900987387
                    if state == 14:
                        return -0.139663339
    elif num_booster == 2:
        state = 0
        if state == 0:
            state = (1 if x['Pclass']<3 or np.isnan(x['Pclass'])  else 2)
            if state == 1:
                state = (3 if x['Fare']<13.6458502 or np.isnan(x['Fare'])  else 4)
                if state == 3:
                    state = (7 if x['Fare']<7.75 or np.isnan(x['Fare'])  else 8)
                    if state == 7:
                        return -0.142080292
                    if state == 8:
                        return -0.0411125682
                if state == 4:
                    state = (9 if x['Age']<42.5 else 10)
                    if state == 9:
                        return 0.0736838281
                    if state == 10:
                        return -0.00772823254
            if state == 2:
                state = (5 if x['Age']<6.5 else 6)
                if state == 5:
                    state = (11 if x['SibSp']<3 or np.isnan(x['SibSp'])  else 12)
                    if state == 11:
                        return 0.12030784
                    if state == 12:
                        return -0.0953842178
                if state == 6:
                    return -0.090982236
    elif num_booster == 3:
        state = 0
        if state == 0:
            state = (1 if x['Pclass']<3 or np.isnan(x['Pclass'])  else 2)
            if state == 1:
                state = (3 if x['Fare']<52.2770996 or np.isnan(x['Fare'])  else 4)
                if state == 3:
                    state = (7 if x['Age']<15 else 8)
                    if state == 7:
                        return 0.146795467
                    if state == 8:
                        return -0.0269840453
                if state == 4:
                    state = (9 if x['Pclass']<2 or np.isnan(x['Pclass'])  else 10)
                    if state == 9:
                        return 0.0821756721
                    if state == 10:
                        return -0.0594713204
            if state == 2:
                state = (5 if x['Age']<6.5 else 6)
                if state == 5:
                    state = (11 if x['SibSp']<3 or np.isnan(x['SibSp'])  else 12)
                    if state == 11:
                        return 0.112550609
                    if state == 12:
                        return -0.0888769552
                if state == 6:
                    state = (13 if x['Age']<32.5 or np.isnan(x['Age'])  else 14)
                    if state == 13:
                        return -0.0735475942
                    if state == 14:
                        return -0.121475317
    elif num_booster == 4:
        state = 0
        if state == 0:
            state = (1 if x['Fare']<10.4812498 or np.isnan(x['Fare'])  else 2)
            if state == 1:
                state = (3 if x['Age']<16.5 else 4)
                if state == 3:
                    return 0.0184162073
                if state == 4:
                    return -0.0909106433
            if state == 2:
                state = (5 if x['Fare']<74.375 or np.isnan(x['Fare'])  else 6)
                if state == 5:
                    state = (11 if x['SibSp']<3 or np.isnan(x['SibSp'])  else 12)
                    if state == 11:
                        return -0.00273782061
                    if state == 12:
                        return -0.118794046
                if state == 6:
                    return 0.0763329566
    elif num_booster == 5:
        state = 0
        if state == 0:
            state = (1 if x['Fare']<10.8249998 or np.isnan(x['Fare'])  else 2)
            if state == 1:
                state = (3 if x['Fare']<7.1333499 or np.isnan(x['Fare'])  else 4)
                if state == 3:
                    return -0.129830331
                if state == 4:
                    state = (7 if x['Age']<19.5 else 8)
                    if state == 7:
                        return -0.0106215896
                    if state == 8:
                        return -0.076526165
            if state == 2:
                state = (5 if x['SibSp']<3 or np.isnan(x['SibSp'])  else 6)
                if state == 5:
                    state = (9 if x['Age']<15.5 else 10)
                    if state == 9:
                        return 0.119682573
                    if state == 10:
                        return 0.00138043496
                if state == 6:
                    state = (11 if x['Age']<17.5 or np.isnan(x['Age'])  else 12)
                    if state == 11:
                        return -0.126713961
                    if state == 12:
                        return 0.0615616553
    elif num_booster == 6:
        state = 0
        if state == 0:
            state = (1 if x['Pclass']<3 or np.isnan(x['Pclass'])  else 2)
            if state == 1:
                state = (3 if x['Fare']<13.6458502 or np.isnan(x['Fare'])  else 4)
                if state == 3:
                    return -0.0478180945
                if state == 4:
                    state = (9 if x['Age']<42.5 else 10)
                    if state == 9:
                        return 0.0603423826
                    if state == 10:
                        return -0.0106344288
            if state == 2:
                state = (5 if x['Age']<6.5 else 6)
                if state == 5:
                    state = (11 if x['SibSp']<3 or np.isnan(x['SibSp'])  else 12)
                    if state == 11:
                        return 0.0999216512
                    if state == 12:
                        return -0.0678185076
                if state == 6:
                    state = (13 if x['Age']<32.5 or np.isnan(x['Age'])  else 14)
                    if state == 13:
                        return -0.0564754978
                    if state == 14:
                        return -0.104635693
    elif num_booster == 7:
        state = 0
        if state == 0:
            state = (1 if x['Fare']<52.2770996 or np.isnan(x['Fare'])  else 2)
            if state == 1:
                state = (3 if x['Age']<6.5 else 4)
                if state == 3:
                    state = (7 if x['SibSp']<3 or np.isnan(x['SibSp'])  else 8)
                    if state == 7:
                        return 0.120081916
                    if state == 8:
                        return -0.0637827888
                if state == 4:
                    state = (9 if x['Fare']<10.4812498 or np.isnan(x['Fare'])  else 10)
                    if state == 9:
                        return -0.0682334229
                    if state == 10:
                        return -0.0263117608
            if state == 2:
                state = (5 if x['SibSp']<2 or np.isnan(x['SibSp'])  else 6)
                if state == 5:
                    return 0.0687555224
                if state == 6:
                    state = (13 if x['Fare']<111.824997 or np.isnan(x['Fare'])  else 14)
                    if state == 13:
                        return -0.126738429
                    if state == 14:
                        return 0.0544673391
    elif num_booster == 8:
        state = 0
        if state == 0:
            state = (1 if x['Fare']<52.2770996 or np.isnan(x['Fare'])  else 2)
            if state == 1:
                state = (3 if x['Age']<6.5 else 4)
                if state == 3:
                    state = (7 if x['SibSp']<3 or np.isnan(x['SibSp'])  else 8)
                    if state == 7:
                        return 0.113764763
                    if state == 8:
                        return -0.0600335076
                if state == 4:
                    state = (9 if x['Pclass']<3 or np.isnan(x['Pclass'])  else 10)
                    if state == 9:
                        return -0.0165072344
                    if state == 10:
                        return -0.0574411936
            if state == 2:
                state = (5 if x['SibSp']<2 or np.isnan(x['SibSp'])  else 6)
                if state == 5:
                    return 0.0628494471
                if state == 6:
                    state = (13 if x['Fare']<111.824997 or np.isnan(x['Fare'])  else 14)
                    if state == 13:
                        return -0.119837247
                    if state == 14:
                        return 0.0516858809
    elif num_booster == 9:
        state = 0
        if state == 0:
            state = (1 if x['Fare']<10.4812498 or np.isnan(x['Fare'])  else 2)
            if state == 1:
                state = (3 if x['Fare']<7.1333499 or np.isnan(x['Fare'])  else 4)
                if state == 3:
                    return -0.111991264
                if state == 4:
                    state = (7 if x['Fare']<7.8833499 or np.isnan(x['Fare'])  else 8)
                    if state == 7:
                        return -0.0179245993
                    if state == 8:
                        return -0.0776858404
            if state == 2:
                state = (5 if x['SibSp']<3 or np.isnan(x['SibSp'])  else 6)
                if state == 5:
                    state = (9 if x['Age']<15.5 else 10)
                    if state == 9:
                        return 0.102338471
                    if state == 10:
                        return 0.000354360003
                if state == 6:
                    state = (11 if x['Age']<17.5 or np.isnan(x['Age'])  else 12)
                    if state == 11:
                        return -0.108861342
                    if state == 12:
                        return 0.0559688099

def xgb_predict(x):
    predict = 0
# initialize prediction with base score
    for i in range(10):
        predict = predict + xgb_tree(x, i)
    return predict