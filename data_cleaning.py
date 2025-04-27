import pandas as pd 
import numpy as np 
from matplotlib import pyplot as plt 
import matplotlib 
#import matplotlib.pyplot as plt
#import matplotlib
matplotlib.rcParams["figure.figsize"] = (20,10)

df1 = pd.read_csv("Bengaluru_House_Data.csv")
#print(df1.head())
print(df1.shape)


df1.groupby('area_type')['area_type'].agg('count')

df2= df1.drop(['area_type','society','balcony','availability'],axis='columns')
#print(df2.head())

#print(df2.isnull().sum())

df3= df2.dropna()
df3.isnull().sum()

#print(df3.isnull().sum())


#print(df3.shape)

#print(df3['size'].unique())



df3['bhk']= df3['size'].apply(lambda x: x.split(' ')[0])
#print(df3.head())
#print(df3['bhk'].unique())

#print(df3[df3.bhk > '20'])


print(df3.total_sqft.unique())


def is_float(x):
    try:
        float(x)
    except:
        return False
    return True


#print(df3[~df3['total_sqft'].apply(is_float)].head())

def convert_sqft_to_num(x):
    tokens = x.split(' - ')
    if len(tokens)== 2:
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None


print(convert_sqft_to_num('2100 - 2850'))

df4 = df3.copy()

df4['total_sqft'] = df4['total_sqft'].apply(convert_sqft_to_num)
#print(df4.head())

#print(df4.loc[30])



#3rd lecture----------------------------------------------------------------------

df5 = df4.copy()

df5['price_pre_sqft']= df5['price']*100000/df5['total_sqft']
#print(df5.head())

#print(df5.location.unique())

df5.location =  df5.location.apply(lambda x : x.strip())
location_stats = df5.groupby('location')['location'].agg('count').sort_values(ascending=False)
#print(location_stats)

#print(len(location_stats[location_stats<=10]))

#print(location_stats[location_stats<=10])
df5.location.unique()
location_stats_less_than_10=location_stats[location_stats<=10]

df5.location = df5.location.apply(lambda x : 'other' if x in location_stats_less_than_10 else x)
#print(len(df5.location.unique()))

#print(df5.head())
import pandas as pd

# Convert total_sqft to numeric, forcing errors to NaN
df5['total_sqft'] = pd.to_numeric(df5['total_sqft'], errors='coerce')

# Convert bhk to numeric, if it's not already
df5['bhk'] = pd.to_numeric(df5['bhk'], errors='coerce')

# Drop rows with NaN values in total_sqft or bhk if necessary
df5 = df5.dropna(subset=['total_sqft', 'bhk'])

# Now apply your filter
import pandas as pd

# Convert total_sqft to numeric, forcing errors to NaN
df5['total_sqft'] = pd.to_numeric(df5['total_sqft'], errors='coerce')

# Convert bhk to numeric, if it's not already
df5['bhk'] = pd.to_numeric(df5['bhk'], errors='coerce')

# Drop rows with NaN values in total_sqft or bhk if necessary
df5 = df5.dropna(subset=['total_sqft', 'bhk'])

# Now apply your filter


# Convert total_sqft to numeric, forcing errors to NaN
df5['total_sqft'] = pd.to_numeric(df5['total_sqft'], errors='coerce')

# Convert bhk to numeric, if it's not already
df5['bhk'] = pd.to_numeric(df5['bhk'], errors='coerce')

# Drop rows with NaN values in total_sqft or bhk if necessary
df5 = df5.dropna(subset=['total_sqft', 'bhk'])

# Now apply your filter
#print(df5[~(df5.total_sqft / df5.bhk < 300)].head())

df5.shape

df6 = df5[~(df5.total_sqft / df5.bhk < 300)]
df6.shape

#print(df6.price_pre_sqft.describe())


def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key , subdf in df.groupby('location'):
        m=np.mean(subdf.price_pre_sqft)
        st = np.std(subdf.price_pre_sqft)
        reduce_df = subdf[(subdf.price_pre_sqft > (m-st)) & (subdf.price_pre_sqft <= (m+st))]
        df_out = pd.concat([df_out,reduce_df],ignore_index= True)
    return df_out


df7 = remove_pps_outliers(df6)

print(df7.shape)
'''
def plot_scatter_chat(df,location):
    bhk2= df[(df.location== location) & (df.bhk==2)]
    bhk3= df[(df.location== location) & (df.bhk==3)]
    matplotlib.rcParams['figure.figsize'] =(15,10)
    plt.scatter(bhk2.total_sqft,bhk2.price_pre_sqft,color='blue', lebel='2 BHK',s=50)
    plt.scatter(bhk3.total_sqft,bhk3.price_pre_sqft, marker='+',color='green', lebel='3 BHK',s=50)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price Per Square Feet")
    plt.title(location)
    plt.legend()
print(plot_scatter_chat(df7,"Rajaji Nagar"))
'''


def plot_scatter_chat(df, location):
    bhk2 = df[(df.location == location) & (df.bhk == 2)]
    bhk3 = df[(df.location == location) & (df.bhk == 3)]
    matplotlib.rcParams['figure.figsize'] = (15, 10)
    
    # Corrected 'lebel' to 'label'
    plt.scatter(bhk2.total_sqft, bhk2.price_pre_sqft, color='blue', label='2 BHK', s=50)
    plt.scatter(bhk3.total_sqft, bhk3.price_pre_sqft, marker='+', color='green', label='3 BHK', s=50)
    
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price Per Square Feet")
    plt.title(location)
    plt.legend()
    plt.show()

# Call the function with your DataFrame
#plot_scatter_chat(df7, "Rajaji Nagar")

def remove_bhk_outlaiers(df):
    exclude_indices = np.array([])
    for location,location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk,bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] ={
                'mean' : np.mean(bhk_df.price_pre_sqft),
                'std' : np.std(bhk_df.price_pre_sqft),
                'count' : bhk_df.shape[0]

            }
        for bhk,bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk - 1)
            if stats and stats['count']> 5:
                exclude_indices = np.append(exclude_indices,bhk_df[ bhk_df.price_pre_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')

df8 = remove_bhk_outlaiers(df7)
#print(df8.shape)

#plot_scatter_chat(df8,"Hebbal")


#histogram---------
matplotlib.rcParams["figure.figsize"] = (20,10)
plt.hist(df8.price_pre_sqft,rwidth = 0.8)
plt.xlabel("Price Per Square Feet")
plt.ylabel("Count")
#plt.show()


#print(df8.bath.unique())

df8[df8.bath>df8.bhk+2]

df9 = df8[df8.bath<df8.bhk+2]

#print(df9.shape)

df10 = df9.drop(['size', 'price_pre_sqft'], axis='columns')
#print(df10.head())



#lecture 5--------------------------------------------------------------------
dummies = pd.get_dummies(df10.location)
#print(dummies.head(3))

df11 = pd.concat([df10,dummies.drop('other',axis='columns')],axis='columns')
#print(df11.head(3))



df12 = df11.drop('location',axis='columns')
#print(df12.head(2))

 
df12.shape 
x = df12.drop('price',axis='columns')
#print(x.head())

y = df12.price
#print(y.head())

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=10)



from sklearn.linear_model import LinearRegression 
lr_clf = LinearRegression()
lr_clf.fit(X_train,y_train)
lr_clf.score(X_test,y_test)
#print(lr_clf.score(X_test,y_test))


from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

cross_val_score(LinearRegression(), x,y, cv=cv)
#print(cross_val_score(LinearRegression(), x,y, cv=cv))

'''
from sklearn.model_selection import GridSearchCV  # Correct capitalization
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor  # Correct class name


def find_best_model_using_gridsearchcv(x,y):
    algos= {
        'linear_regression':{
            'model': LinearRegression(),
            'params':{
                'normalize': [True,False]
            }
        },
        'lasso':{
            'model': Lasso(),
            'params':{
                'alpha': [1,2],
                'selection': ['random','cyclic'] 
                   
            }
         },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params':{
                'criterion':['mse','friedman_mse'],
                'splitter':['best','random'] 
            }
        }    
    }
    scores= []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        gs = GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(x,y)
        scores.append({
            'model': algo-name,
            'best_score': gs.best_score_,
            'best_params': gs.best_paramas_
        })
    return pd.DataFrame(score,columns=['model','best_score','best_params'])
find_best_model_using_gridsearchcv(x,y)'''

from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
from sklearn.preprocessing import StandardScaler

def find_best_model_using_gridsearchcv(X, y):
    algos = {
        'linear_regression': {
            'model': LinearRegression(),
            'params': {
                # Removed 'normalize' as it's deprecated
                'fit_intercept': [True, False]
            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1, 2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion': ['squared_error', 'friedman_mse'],  # Replaced 'mse' with 'squared_error'
                'splitter': ['best', 'random']
            }
        }
    }

    scores = []  # Correct initialization of an empty list
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

    for algo_name, config in algos.items():
        gs = GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(X, y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })
    
    return pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])



# Example call (assuming X and y are defined)
# find_best_model_using_gridsearchcv(X, y)

#print(find_best_model_using_gridsearchcv(x,y))


import numpy as np

# def predict_price(location, sqft, bath, bhk):
#     # Initialize a zero vector with the same length as the number of columns in x
#     loc_index = np.where(x.columns == location)[0][0] if location in x.columns else -1

#     # Create a zero array for input features
#     x_input = np.zeros(len(x.columns))
#     x_input[0] = sqft
#     x_input[1] = bath
#     x_input[2] = bhk

#     # Set location index to 1 if location exists
#     if loc_index >= 0:
#         x_input[loc_index] = 1

#     # Return the predicted price using the trained model
#     return lr_clf.predict([x_input])[0]



def predict_price(location, sqft, bath, bhk):
    loc_index = np.where(x.columns == location)[0][0] if location in x.columns else -1
    x_input = np.zeros(len(x.columns))
    x_input[0] = sqft
    x_input[1] = bath
    x_input[2] = bhk
    if loc_index >= 0:
        x_input[loc_index] = 1
    
    x_input_df = pd.DataFrame([x_input], columns=x.columns)  # Wrap in DataFrame
    return lr_clf.predict(x_input_df)[0]




print(predict_price('Indira Nagar', 1000, 3, 3))


print(predict_price('1st phase JP Nagar',1000,2, 3))
print(predict_price('',1000,2, 2))
print(predict_price('Indira Nagar',1500,3, 3))

'''

import pickle
with open('banglore_home_price_model.pickle','wb') as f:
    pickle.dump(lr_clf,f)

import json
columns = {
    'data_columns': [col.lower() for col in x.columns]
}
with open("columns.json","w") as f:
    f.write(json.dumps(columns)) 
    '''