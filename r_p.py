import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import pandas_datareader.data as web
import numpy as np


start = dt.datetime(2019,1,1)
end = dt.datetime(2019,1,11)


# Read file and Get Tickers:
                       ######change this!!!!!!
tickers = pd.read_csv('companylist.csv',header = 0)

# empty dataframe
main_df = pd.DataFrame()

###TRAVERSE && STORE TICKER IN A VARIABLE
###get data from YAHOO
### add new column to dataframe
###everything stored in: main_df

for index, row in tickers.iterrows():
    ticker = str((row['Symbol']))
    #print(ticker)
    df = web.DataReader(ticker, 'yahoo', start, end)
    close_price = df['Close'].to_frame()
    close_price.columns = [ticker]
    if main_df.empty:
        main_df = close_price
        #print(main_df)
    else:
        main_df = main_df.join(close_price)
        #print(main_df)
  
#print(main_df)


#get relative prices and concate into a matrix

relative_price1 = main_df.loc['2019-01-08'] / main_df.loc['2019-01-07']
relative_price1 = relative_price1.to_frame()
relative_price1.columns = ['R_P1']
matrix = pd.DataFrame()
matrix = relative_price1
relative_price2 = main_df.loc['2019-01-09'] / main_df.loc['2019-01-07']
relative_price2 = relative_price2.to_frame()
relative_price2.columns = ['R_P2']
matrix = matrix.join(relative_price2)
relative_price3 = main_df.loc['2019-01-10'] / main_df.loc['2019-01-07']
relative_price3 = relative_price3.to_frame()
relative_price3.columns = ['R_P3']
matrix = matrix.join(relative_price3)
#print(matrix)


#create a empty matrix(covariance) with columns and rows labeled
cols = []
for index, row in tickers.iterrows():
    t = str((row['Symbol']))
    cols.append(t)

covariance = pd.DataFrame(columns = cols, index = cols)

#calculate covariance between rows of matrix(W), append result to matrix covariance
for col in covariance:
    for i, row in covariance.iterrows():
        covariance[col][i] = matrix.loc[str(col)].corr(matrix.loc[str(i)])
        
#print(covariance)

# construct U
    
u1 = pd.DataFrame(columns = cols, index = cols)
#print(u)
for col in u1:
    for i, row in u1.iterrows():
        u1[col][i] = 0

#print(u1)

################
left = [u1, covariance]
u2 = pd.concat(left)
right = [covariance, u1]
u3 = pd.concat(right)

u = pd.concat([u2,u3], axis =1)

################
u_copy = pd.DataFrame()
u_copy = u
################

u.index = range(112)
u.columns = range(112)
#print(u)


#construct D
h = w = 112
d = pd.DataFrame(0, index = range(h), columns = range(w))
for i in range(len(d.index)):
    for j in range(len(d.columns)):
        if(j == i):
            data = pd.DataFrame()
            data = u.iloc[i]
            num = data.sum()
            d.loc[i,j] = num
            
        
#print(d)


#construct L
l = pd.DataFrame()
l = u + d
#print(l)


#construct I
h = w = 112
I = pd.DataFrame(0, index = range(h), columns = range(w))
for i in range(len(I.index)):
    for j in range(len(I.columns)):
        if(j == i):
            I.loc[i,j] = 1
        else:
            I.loc[i,j] = 0

#print(I)

#getting user input          
eta = input("Give me a constant(eta)")
eta = int(eta)
beta = input("Give me another constant(beta)")
beta = int(beta)
lambda_ = 2 * eta * beta

# 2 lambda L
l_ = pd.DataFrame()
l_ = l * lambda_

#print(l_)

# 2 lambda L + I
before_inverse = pd.DataFrame()
before_inverse = l_ + I
#print(before_inverse)



#convert dataframe to numeric matrix
before_inverse = before_inverse.values
before_inverse = np.float64(before_inverse)
#######################
inverse = np.linalg.inv(before_inverse)
inverse = pd.DataFrame(inverse)
#print("inverse:")
#print(inverse)


rp = main_df.loc['2019-01-10'] / main_df.loc['2019-01-07']
rp = rp.to_frame()
# xt
concate = [rp,rp]
xt = pd.concat(concate)
#print(xt)

# xt+1
rp2 = main_df.loc['2019-01-11'] / main_df.loc['2019-01-07']
rp2 = rp2.to_frame()
concate2 = [rp2,rp2]
xt_plus1 = pd.concat(concate2)

#print(xt_plus1)

# bs，bl
min_value = rp.min()
max_value = rp.max()
bl = 1 - min_value + 0.5
bs = max_value - 1 + 60

#print(bl)
#print(bs)

#calculate coefficient
coefficient = ((1 + 0.02) / (bl + 0.02)) / (1 + ((bs + 0.02) / (bl + 0.02)))
#print(coefficient)

# construct pt
h = 56
w = 1
pt_ = pd.DataFrame(index = range(h), columns = range(w))
pt_.fillna(coefficient, inplace = True) 
concatpt = [pt_,-pt_]
pt = pd.concat(concatpt)
#print("pt")
print(pt)

# Dl
h = w = 112
Dl = pd.DataFrame(0, index = range(h), columns = range(w))
for i in range(len(Dl.index)):
    for j in range(len(Dl.columns)):        
        if(i < 56 and j == i):
            Dl.loc[i,j] = 1
#print(Dl)


# Ds
h = w = 112
Ds = pd.DataFrame(0, index = range(h), columns = range(w))
for i in range(len(Ds.index)):
    for j in range(len(Ds.columns)):        
        if(i >= 56 and j == i):
            Ds.loc[i,j] = 1
        
#print(Ds)

############################
# data type of Ds, Dl, xt converted
Dl = Dl.values
xt = xt.values
Ds = Ds.values

# Dlxt
Dlxt = np.dot(Dl, xt)
Dlxt = pd.DataFrame(Dlxt)
#print(Dlxt)

# Ds(Xt - 1 + r)
xt_ = xt - 1 + 0.02
Dsxt = np.dot(Ds, xt_)
Dsxt = pd.DataFrame(Dsxt)
#print(Dsxt)

# Dl1(1+r)2
acol = pd.DataFrame(1.02, index = range(112), columns = range(1))
acol = acol.values
Dlr = np.dot(Dl,acol)
Dlr = pd.DataFrame(Dlr)
#print(Dlr)

# numerator
numerator = Dlxt + Dsxt - Dlr
#print(numerator)


'''

arr = np.array([1,2,3,4])
arr2 = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
print(np.dot(arr,arr2))
print("/")
print(np.dot(arr2,arr))

'''

# ptDlxt
'''
apt = pd.DataFrame(index = range(1), columns = range(112))
for i in range(len(apt.index)):
    for j in range(len(apt.columns)):
        apt.loc[i][j] = pt.loc[j][i]
Dl = pd.DataFrame(Dl)        
ptDl = apt.dot(Dl)
xt = pd.DataFrame(xt)
ptDlxt = ptDl.dot(xt)
#print(ptDlxt)
'''
p_t = pt.values
p_t = p_t.T
ptDl = np.dot(p_t, Dl)
ptDlxt = np.dot(ptDl, xt)

# ptDs(xt - 1 + r)
_xt = xt - 1 + 0.02
pt = pt.values
pt = pt.T
ptDs = np.dot(pt, Ds)
ptDsxt = np.dot(ptDs, _xt)
ptDsxt = pd.DataFrame(ptDsxt)
#print(ptDsxt)

# (1-ptDl1)(1+r)
ccol = pd.DataFrame(1, index = range(112), columns = range(1))
ptDl1 = ptDl.dot(ccol)
ptDlr = (1 - ptDl1) * (1.02)

# Denomerator
Denomerator = ptDlxt + ptDsxt + ptDlr
Denomerator = Denomerator.loc[0][0]
#print(Denomerator)

# Result
for i in range(len(numerator.index)):
    for j in range(len(numerator.columns)):
        numerator.loc[i][j] = numerator.loc[i][j] / Denomerator
#ltpt = pd.DataFrame(ltpt)
#print(" ltpt: ")
#print(numerator)

# numerator + pt
pt = pt.T
pt = pd.DataFrame(pt)
adding = numerator + pt
#print(adding)

# adding * inverse (Pt+1)
adding = adding.values
adding = adding.T
inverse_ = inverse.values
result = np.dot(adding, inverse_)
result = result.T
result = pd.DataFrame(result)
#print(result)
result = result.values
result = result.T


# xt+1 * pt+1(result)
xt_plus1 = xt_plus1.values
final_result = np.dot(result, xt_plus1)
print(final_result)

'''
pt = pt.values
pt = pt.T
xt = xt.values
test = np.dot(pt, xt)
print(test)
'''

    
## # TICKER, SOURCE, START, END
##df2 = web.DataReader('BABA','yahoo', start, end)
##df = web.DataReader('AAPL','yahoo', start, end)
##close_price = df['Close'].to_frame()
##close_price.columns = ['AAPL']
##print(close_price)
###GET a column
##
##close_price2 = df2['Close'].to_frame()
##print(close_price2)
##main_df = pd.DataFrame()
##
##
####
##
##
##print(main_df.head(15))
##
##



##combined_columns = close_price.append(close_price2)
##
##
###c = np.array(df[['AAPL', 'BABA']])
###matrix = pd.DataFrame(combined_columns);
###matrix.set_index(
##
##print(combined_columns)
