# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 14:31:12 2019

@author: rhlei
"""

import re
import pandas as pd
import xgboost as xgb
import numpy as np

#train = pd.read_csv("Titanic/Titanic/train.csv")
#test  = pd.read_csv("Titanic/Titanic/test.csv")
#X_y_train = xgb.DMatrix(data=train[['Pclass', 'Age', 'Fare', 'SibSp', 'Parch']], 
#                        label=train['Survived'])
#X_test    = xgb.DMatrix(data=test[['Pclass', 'Age', 'Fare', 'SibSp', 'Parch']])


#print(train[['Pclass', 'Age', 'Fare', 'SibSp', 'Parch']])

#params = {
          #'base_score': np.mean(train['Survived']),
#          'eta':  0.1,
#          'max_depth': 3,
#          'gamma' :3,
#          'objective'   :'binary:logistic',
#          'eval_metric' :'auc'
#         }
#model = xgb.train(params=params, dtrain=X_y_train, num_boost_round=10)
#model.dump_model("model2.txt")

#ypred=model.predict(X_test)
#print(ypred)




def string_parser(s,numOfTree):
    if len(re.findall(r":leaf=", s)) == 0:
        out  = re.findall(r"[\w.-]+", s)
        tabs = re.findall(r"[\t]+", s)
        if (out[4] == out[8]):
         missing_value_handling = (" or np.isnan(x['" + out[1] + "']) ")
            

        else:
            missing_value_handling = ""
            
        if len(tabs) > 0:
            return (re.findall(r"[\t]+", s)[0].replace('\t', '    ') + 
                    '        if state == ' + out[0] + ':\n' +
                    re.findall(r"[\t]+", s)[0].replace('\t', '    ') +
                    '            state = (' + out[4] +  
                    ' if ' +  "x['" + out[1] +"']<" + out[2] + missing_value_handling + 
                    ' else ' + out[6] + ')\n' )
        
        else:
            return ('        if state == ' + out[0] + ':\n' +
                    '            state = (' + out[4] +  
                    ' if ' +  "x['" + out[1] +"']<" + out[2] + missing_value_handling +
                    ' else ' + out[6] + ')\n' )
    else:
        out = re.findall(r"[\d.-]+", s)
        return (re.findall(r"[\t]+", s)[0].replace('\t', '    ') + 
                '        if state == ' + out[0] + ':\n    ' +
                re.findall(r"[\t]+", s)[0].replace('\t', '    ') + 
                '        return "' + numOfTree+out[0] + '"\n')
      
def tree_parser(tree, i):
    if i == 0:
        return ('    if num_booster == 0:\n        state = 0\n'
             + "".join([string_parser(tree.split('\n')[i],'0') 
                        for i in range(len(tree.split('\n'))-1)]))
    else:
        numOfTree = str(i)
        return ('    elif num_booster == '+str(i)+':\n        state = 0\n'
             + "".join([string_parser(tree.split('\n')[i],numOfTree)
                        for i in range(len(tree.split('\n'))-1)])) 
    
def model_to_py(base_score, dump_file, out_file):
    resList=[]
    with open(dump_file,'r') as inputdump:
      resPart = ""
      for line in inputdump:

        if line is None:
          break
        if 'booster' in line and resPart is not "":
          resList.append(resPart)
          resPart=""
        elif 'booster' not in line:
          resPart = resPart + line
    if resPart is not "":
      resList.append(resPart)
    #trees = model.get_dump()
    result = ["import numpy as np\n\n" 
             +"def xgb_tree(x, num_booster):\n"]
    
    for i in range(len(resList)):
        result.append(tree_parser(resList[i], i))
    
    with open(out_file, 'w') as the_file:
        the_file.write("".join(result) + "\ndef xgb_predict(x):\n    predict = " 
                       + "[]" + "\n"
                       + "# initialize prediction with base score\n" 
                       + "    for i in range(" 
                       + str(len(resList))  
                       + "):\n        predict.append( xgb_tree(x, i) )"
                       + "\n    return predict\n\n\n\n"
                       +"import pandas as pd \n"
                       +"data = pd.read_csv('test_data1.csv').head(10).apply(lambda x:xgb_predict(x), axis=1)\n"
                       +"print(data.head(10))\n")




model_to_py(0, 'xgb_zzr.txt', 'xgb_model_zzr.py')


#import xgb_tree from xgb_model_zzr

#for i in  

#model.dump_model("model1.txt")

#import xgb_model
#print(1/float(1+np.exp(-xgb_model.xgb_predict({'Pclass':1,'Age':46,'SibSp':0,'Parch':0,'Fare':26}))))
#print(1/float(1+np.exp(-xgb_model.xgb_predict({'Pclass':3,'Age':np.nan,'SibSp':0,'Parch':0,'Fare':7.8958}))))
             
