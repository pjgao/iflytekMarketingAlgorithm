'''
@author: pipixiu
@time: 2018.9.18
@city: Nanjing
'''


import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import log_loss
import lightgbm as lgb
from scipy import sparse
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def count_corr(df):
    '''
    输入dataframe
    输出相关系数dataframe:col_1,col_2,cor(不包含同一特征且已去重复)
    '''
    x = df.corr().abs().unstack().sort_values(ascending=False).reset_index()
    x = x.loc[x.level_0!=x.level_1]
    x2 = pd.DataFrame([sorted(i) for i in x[['level_0','level_1']].values])
    x2['cor'] = x[0].values
    x2.columns = ['col_1','col_2','cor']
    return x2.drop_duplicates()

# 数据预处理
def get_feature(df,all_data,one_hot_col,vec_col):
    enc = OneHotEncoder()
    df_x=df[['user_tag_length']]
    for feature in one_hot_col:
        enc.fit(all_data[feature].values.reshape(-1, 1))
        df_a=enc.transform(df[feature].values.reshape(-1, 1))        
        df_x= sparse.hstack((df_x, df_a))
    print('one-hot prepared !')
    cv=CountVectorizer()
    for feature in vec_col:
        cv.fit(all_data[feature])
        df_a = cv.transform(df[feature])
        df_x = sparse.hstack((df_x, df_a))  
    print('cv prepared!')
    return df_x

def LGB_test(train_x,train_y,test_x,test_y):
    print("LGB test")
    weights = (len(train_y)/train_y.value_counts()).to_dict()
    clf = lgb.LGBMClassifier(
        
        boosting_type='gbdt', num_leaves=64, reg_alpha=0.0, reg_lambda=1,
        max_depth=-1, n_estimators=3000, objective='binary',
        subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
        learning_rate=0.05, min_child_weight=50,random_state=2018,n_jobs=-1
    )
    clf.fit(train_x, train_y,eval_set=[(train_x, train_y),(test_x,test_y)],early_stopping_rounds=100,verbose=10)
    # print(clf.feature_importances_)
    return clf

def load_data(path):
    test = pd.read_table(f'{path}/round1_iflyad_test_feature.txt',index_col='instance_id')
    train = pd.read_table(f'{path}/round1_iflyad_train.txt',index_col='instance_id')

    train.time = train.time.apply(lambda x:datetime.fromtimestamp(x))
    test.time = test.time.apply(lambda x:datetime.fromtimestamp(x))

    train.app_id = train.app_id.fillna(-1).astype(int)
    test.app_id = test.app_id.fillna(-1).astype(int)

    train.app_cate_id = train.app_cate_id.fillna(-1).astype(int)
    test.app_cate_id = test.app_cate_id.fillna(-1).astype(int)

    train.os_name = train.os_name.map({'android':2,'ios':1,'unknown':0})
    test.os_name = test.os_name.map({'android':2,'ios':1,'unknown':0})

    col_bool = train.select_dtypes(bool).columns.values.tolist()

    for i in col_bool:
        train[i] = train[i].astype(int)
        test[i] = test[i].astype(int)

    test['advert_industry_inner_0'] = test.advert_industry_inner.apply(lambda x:x.split('_')[0]).apply(int)
    test['advert_industry_inner_1'] = test.advert_industry_inner.apply(lambda x:x.split('_')[1]).apply(int)

    train['advert_industry_inner_0'] = train.advert_industry_inner.apply(lambda x:x.split('_')[0]).apply(int)
    train['advert_industry_inner_1'] = train.advert_industry_inner.apply(lambda x:x.split('_')[1]).apply(int)

    train.advert_industry_inner = train.advert_industry_inner.apply(lambda x:int(''.join(x.split('_'))))
    test.advert_industry_inner = test.advert_industry_inner.apply(lambda x:int(''.join(x.split('_'))))

    v = train.var()
    constant_feature = v[v==0].index.values.tolist()

    for i in constant_feature:
        _ = train.pop(i)
        _ = test.pop(i)

    train_corr_col = count_corr(train)
    corr_col = train_corr_col[train_corr_col.cor>0.99].col_2.values.tolist()

    for i in corr_col:
        _ = train.pop(i)
        _ = test.pop(i)

    train.fillna('-1',inplace=True)
    test.fillna('-1',inplace=True)

    train['user_tag_length'] = train.user_tags.apply(lambda x:len(x.split(',')))
    test['user_tag_length'] = test.user_tags.apply(lambda x:len(x.split(',')))
    return train,test

print('load data...')

train,test = load_data('data')

print('load data ok!')

test_nunique = test.nunique()
test_nunique.sort_values(inplace=True)

one_hot_col = test_nunique[test_nunique<50].index.values.tolist()
object_col = test.select_dtypes('object').columns
one_hot_col = list(set(one_hot_col)|set(object_col))

vec_col = [i for i in test.columns if i not in one_hot_col+['user_tag_length']]

test['click']=-1

data = pd.concat([train,test])

for i in vec_col:
    data[i] = data[i].astype(str)

for feature in one_hot_col:
    try:
        data[feature] = LabelEncoder().fit_transform(data[feature].apply(int))
    except:
        data[feature] = LabelEncoder().fit_transform(data[feature])



X=data[data.click!=-1]
y=X.pop('click')
X_test =data[data.click==-1]
X_test=X_test.drop(['click'],axis=1)

x1,x2,y1,y2 = train_test_split(X,y)

x1= get_feature(x1,data,one_hot_col,vec_col)
x2 = get_feature(x2,data,one_hot_col,vec_col)

test_sparse = get_feature(X_test,data,one_hot_col,vec_col)

clf = LGB_test(x1,y1,x2,y2)

prob = clf.predict_proba(x2,num_iteration=clf.best_iteration_)

print('log loss in valid:',log_loss(y2,prob[:,1]))

prob_submit = clf.predict_proba(test_sparse,num_iteration=clf.best_iteration_)[:,1]

sub = pd.DataFrame({'instance_id':X_test.index.values.tolist(),'predicted_score':prob_submit.tolist()})

sub.to_csv('submit/baseline_test.csv',index=False)
print('submit data has been saved')