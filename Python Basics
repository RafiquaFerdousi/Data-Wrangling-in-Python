import pandas as pd
bank = pd.read_csv('Data/Bank.csv')
bank['Salary'].mean()
sal = bank['Salary']
sal.min(), sal.mean(), sal.median(), sal.max() bank.describe()
import seaborn as sns
sns.histplot(x=bank['Salary'])
sns.histplot(x=bank['Salary'], bins=10, kde=True);
sns.histplot(x=bank['Salary'], 
             bins=10, kde=False,
             stat="probability",
             color='green' 
            );
sns.boxplot(x=bank['Salary']);
sns.boxplot(y=bank['Salary'],  color='lightgreen', showmeans=True);
bank['Gender'] == 'Female'
FemaleEmployees = bank[bank['Gender'] == "Female"]
type(FemaleEmployees)
FemaleEmployees['Salary'].mean()
round(FemaleEmployees['Salary'].mean(),2)
(bank['Gender'] == 'Female') & (bank['JobGrade'] == 1)
bank[(bank['Gender'] == 'Female') & (bank['JobGrade'] == 1)].shape 
bank[bank['JobGrade'] >= 4]
mgmt = [4,5,6]
bank[bank['JobGrade'].isin(mgmt)]
bank['Dummy'] = 0
bank.drop('Dummy', axis=1, inplace=True)
bank.head()
1 if bank['Gender'] == "Female" else 0


import numpy as np
bank['GenderDummy_F'] = np.where(bank['Gender'] == "Female", 1, 0)
bank.head()

def my_recode(gender):
    if gender == "Female":
        return 1
    else:
        return 0
    
my_recode("Female"), my_recode("Male")
bank['GenderDummy_F'] = bank['Gender'].apply(my_recode)
bank['GenderDummy_F'] = bank['Gender'].apply(lambda x: 1 if x == "Female" else 0)

grades = [1,2,3,4,5,6]
status = ["non-mgmt", "non-mgmt", "non-mgmt", "non-mgmt", "mgmt", "mgmt"]

bank['Manager'] = bank['JobGrade'].replace(grades, status)
bank[170:175]

genders=["Female", "Male"]
dummy_vars=[1,0]

bank['GenderDummy_F'] = bank['Gender'].replace(genders, dummy_vars)
bank.head()

bank['logSalary'] = np.log(bank['Salary'])

import seaborn as sns
sns.kdeplot(x=bank['logSalary'], shade=True, linewidth=2);

sns.boxplot(x=bank['Salary'], y=bank['Gender'], showmeans=True);
sns.boxplot(x=bank['Salary'], y=bank['JobGrade'].astype('category'), showmeans=True);


sns.displot(x='Salary', row='Gender', data=bank, linewidth=0, kde=True);

#Overlaying kernel density plots
sns.histplot(x='Salary', hue='Gender', data=bank, linewidth=0);
sns.kdeplot(x='Salary', hue='Gender', data=bank, shade=True);

import statsmodels.stats.api as sms
model = sms.CompareMeans.from_data(bank[bank['Gender'] == "Female"]['Salary'], bank[bank['Gender'] == "Male"]['Salary'])
model.summary( usevar='unequal')

import pandas as pd
from scipy import stats
bank = pd.read_csv('Data/Bank.csv')

# Recode JobGrade to Manager
grades = [1,2,3,4,5,6]
status = ["non-mgmt", "non-mgmt", "non-mgmt", "non-mgmt", "mgmt", "mgmt"]
bank['Manager'] = bank['JobGrade'].replace(grades, status)

contab_freq = pd.crosstab(
    bank['Gender'],
    bank['Manager'],
    margins = True
   )
contab_freq


conttab_relfreq = pd.crosstab(
    bank['Gender'],
    bank['Manager'],
    margins = True,
    normalize='index'
   )
conttab_relfreq

import seaborn as sns
sns.scatterplot(x="FlyAsh", y="Strength", data=con);
ax = sns.scatterplot(x="FlyAsh", y="Strength", data=con)
ax.set_title("Concrete Strength vs. Fly ash")
ax.set_xlabel("Fly ash");
sns.lmplot(x="FlyAsh", y="Strength", data=con);
sns.lmplot(x="FlyAsh", y="Strength", hue="AirEntrain", data=con);

from scipy import stats
stats.pearsonr(con['Strength'], con['FlyAsh'])
cormat = con.corr()
round(cormat,2)
sns.heatmap(cormat);


#Regression
import pandas as pd
con = pd.read_csv('Data/ConcreteStrength.csv')
con.rename(columns={'Fly ash': 'FlyAsh', 'Coarse Aggr.': "CoarseAgg",
                    'Fine Aggr.': 'FineAgg', 'Air Entrainment': 'AirEntrain', 
                    'Compressive Strength (28-day)(Mpa)': 'Strength'}, inplace=True)
con['AirEntrain'] = con['AirEntrain'].astype('category')
con.head()

import statsmodels.api as sm
Y = con['Strength']
X = con['FlyAsh']
X.head()
X = sm.add_constant(X)
X.head()
model = sm.OLS(Y, X, missing='drop')
model_result = model.fit()
model_result.summary()

import seaborn as sns
sns.histplot(model_result.resid);
from scipy import stats
mu, std = stats.norm.fit(model_result.resid)
mu, std
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()
# plot the residuals
sns.histplot(x=model_result.resid, ax=ax, stat="density", linewidth=0, kde=True)
ax.set(title="Distribution of residuals", xlabel="residual")

# plot corresponding normal curve
xmin, xmax = plt.xlim() # the maximum x values from the histogram above
x = np.linspace(xmin, xmax, 100) # generate some x values
p = stats.norm.pdf(x, mu, std) # calculate the y values for the normal curve
sns.lineplot(x=x, y=p, color="orange", ax=ax)
plt.show()

sns.boxplot(x=model_result.resid, showmeans=True);
sm.qqplot(model_result.resid, line='s');
sm.graphics.plot_fit(model_result,1, vlines=False);
model_result.fittedvalues

Y = con['Strength']
X = con[['No',
 'Cement',
 'Slag',
 'FlyAsh',
 'Water',
 'SP',
 'CoarseAgg',
 'FineAgg']]
X = sm.add_constant(X)

ks = sm.OLS(Y, X)
ks_res =ks.fit()
ks_res.summary()

import statsmodels.formula.api as smf
ksf =  smf.ols(' Strength ~ No + Cement + Slag + Water + CoarseAgg + FlyAsh + SP + FineAgg + AirEntrain', data=con)
ksf_res = ksf.fit()
ksf_res.summary()

X1 = fullX.drop(columns='FineAgg', inplace=False)
mod1 = sm.OLS(Y, X1)
mod1_res = mod1.fit()
mod1_res.summary()