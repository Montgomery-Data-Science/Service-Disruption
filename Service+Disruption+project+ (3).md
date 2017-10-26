
# Cell Tower Outage Prediction Algorithm 

## Steps in CRISP-DM (Cross Industry Standard Process for Data Minning 

Step 1: Understand the Business Objective/Question <br>
Step 2: Data Understanding <br>
Step 3: Data Preperation <br>
Step 4: Modeling <br>
Step 5: Evaluation <br>
Step 6: Deployment <br>

# Step 1: Understand the Business Question

We have data about the times when individual cell towers experienced technical difficulties. Our job is to use the provided data
to predict the probablity of each cell tower going down in the future, and predict how severe the outage will be.

# Step 2: Data Understanding

The following are the variables we have been given and will be using in our classifcation model

## **Data Dictionary**
<br>
### **Dependent Variable**
<font color="blue">**Fault_severity:**</font>**Categorical variable**<br> Categorical variable used to represent the severity of the faults 0-being no fault 2-being many faults 
<br>
### **Independent Variables**
<font color="blue">**Log_feature:**</font> **Categorical variable**<br>The Types of features logged for that each ID <br>
<font color="blue">**Volume:**</font> **Numeric variable**<br> The amount of logged features <br>
<font color="blue">**Resource_type:**</font> **Categorical variable**<br>The type of resource provided by that specific ID<br>
<font color="blue">**Severity_type:**</font> **Categorical variable**<br>The type of severity level logged for that specific ID<br>
<br>
### **Descriptive Variables**
<font color="blue">**ID:**</font> Used to identify a specific event at a specific location <br>
<font color="blue">**Location:**</font> Used to identify the location of the cell tower <br>

# Step 3: Data Preperation 

## Begin by Downloading necessary libraries 


```python
import pandas as pd 
import numpy as np
```

## Read the necessary files into Pandas Data Frames


```python
resources = pd.read_csv("resource_type.csv")
severity = pd.read_csv("severity_type.csv")
log = pd.read_csv("log_feature.csv")
event = pd.read_csv("event_type.csv")
train = pd.read_csv("train.csv")
```


```python

```

## Step 3a Clean the Data 

Create a combine data frame by merging each of the individual data frames


```python
# Merge all the data frames into one data frame
df=log.merge(event)
df=df.merge(resources)
df=df.merge(severity)
data=train.merge(df)

```


```python
data.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>location</th>
      <th>fault_severity</th>
      <th>log_feature</th>
      <th>volume</th>
      <th>event_type</th>
      <th>resource_type</th>
      <th>severity_type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>14121</td>
      <td>location 118</td>
      <td>1</td>
      <td>feature 312</td>
      <td>19</td>
      <td>event_type 34</td>
      <td>resource_type 2</td>
      <td>severity_type 2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>14121</td>
      <td>location 118</td>
      <td>1</td>
      <td>feature 312</td>
      <td>19</td>
      <td>event_type 35</td>
      <td>resource_type 2</td>
      <td>severity_type 2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>14121</td>
      <td>location 118</td>
      <td>1</td>
      <td>feature 232</td>
      <td>19</td>
      <td>event_type 34</td>
      <td>resource_type 2</td>
      <td>severity_type 2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>14121</td>
      <td>location 118</td>
      <td>1</td>
      <td>feature 232</td>
      <td>19</td>
      <td>event_type 35</td>
      <td>resource_type 2</td>
      <td>severity_type 2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9320</td>
      <td>location 91</td>
      <td>0</td>
      <td>feature 315</td>
      <td>200</td>
      <td>event_type 34</td>
      <td>resource_type 2</td>
      <td>severity_type 2</td>
    </tr>
  </tbody>
</table>
</div>



Strip out the words from the Location, Log_feature, Event_type, Severity_type and resource_type columns 


```python
# Loops through each column splitting the strings into a list then appending the numbers to a separate list
loc=[]
for row in data['location']:
    bers=row.split()
    bers=bers[1]
    loc.append(bers)
    
log_f=[]
for row in data['log_feature']:
    bers=row.split()
    bers=bers[1]
    log_f.append(bers)
    
event_t=[]
for row in data['event_type']:
    bers=row.split()
    bers=bers[1]
    event_t.append(bers)
    
resource_t=[]
for row in data['resource_type']:
    bers=row.split()
    bers=bers[1]
    resource_t.append(bers)
    
severity_t=[]
for row in data['severity_type']:
    bers=row.split()
    bers=bers[1]
    severity_t.append(bers)

# Lists with numbers assigned back to respective columns in combined data frame 
data['location']=loc
data['log_feature']=log_f
data['event_type']= event_t
data['resource_type']= resource_t 
data['severity_type']=severity_t
```


```python
data.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>location</th>
      <th>fault_severity</th>
      <th>log_feature</th>
      <th>volume</th>
      <th>event_type</th>
      <th>resource_type</th>
      <th>severity_type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>14121</td>
      <td>118</td>
      <td>1</td>
      <td>312</td>
      <td>19</td>
      <td>34</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>14121</td>
      <td>118</td>
      <td>1</td>
      <td>312</td>
      <td>19</td>
      <td>35</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>14121</td>
      <td>118</td>
      <td>1</td>
      <td>232</td>
      <td>19</td>
      <td>34</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>14121</td>
      <td>118</td>
      <td>1</td>
      <td>232</td>
      <td>19</td>
      <td>35</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9320</td>
      <td>91</td>
      <td>0</td>
      <td>315</td>
      <td>200</td>
      <td>34</td>
      <td>2</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python

```

## Step 3b Creating Dummy Variables 

delete all the non categorical variables from a copy of your dataset


```python
categories = data.copy()
del categories['fault_severity']
del categories['id']
del categories['volume']

```

Store the information from the categorical columns in a list


```python
# create list of names of the categorical columns you want to turn into dummy variables 
dummy_cats = categories.columns
dummies = []

# run the list through loop that appends values from each categorical column from the orginal data set to a new list (dummies)
for i in range(len(dummy_cats)):
    dummies.append(data[dummy_cats[i]])
    
# The newly created list and the list of column names should be the same length 
print len(dummy_cats)
print len(dummies)


```

    5
    5
    

Create the dummy variable columns 


```python
# Create prefixes to identify which category dummy variables belong to (used because we removed string in front of nums above)
prefixes = ['loc', 'logf', 'e_t', 'r_t', 's_t']
for i in range(len(prefixes)):
    # creates dummy variables for each column in the categorical dataframe and adds prefexies to the newly created colums
    dummycreation = pd.get_dummies(categories[dummy_cats[i]], prefix = prefixes[i])
    # joins/adds the columns to the to the categories dataframe
    categories = categories.join(dummycreation)
    #Deltetes orginal categorical column
    del categories[dummy_cats[i]]
```


```python
# updated data frame with dummy variable columns 
categories.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>loc_1</th>
      <th>loc_10</th>
      <th>loc_100</th>
      <th>loc_1000</th>
      <th>loc_1002</th>
      <th>loc_1005</th>
      <th>loc_1006</th>
      <th>loc_1007</th>
      <th>loc_1008</th>
      <th>loc_1009</th>
      <th>...</th>
      <th>r_t_5</th>
      <th>r_t_6</th>
      <th>r_t_7</th>
      <th>r_t_8</th>
      <th>r_t_9</th>
      <th>s_t_1</th>
      <th>s_t_2</th>
      <th>s_t_3</th>
      <th>s_t_4</th>
      <th>s_t_5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 1324 columns</p>
</div>



## Step 3c Compacting the data 

Add the location column back into the dummy_variable dataframe and compress data by location


```python
# add the location column back into the dummy_variable dataframe
df=categories.copy()
df['location']=data['location']
```


```python
# combine all the location instances by summing up the columns where the location is similar 
df_sum = df.groupby('location').sum()
```


```python
# flag each of the dummy variable columns (these coulmns should onlt be 1 or 0)
col_names=df_sum.columns
for cols in col_names:
    df_sum[cols]=df_sum[cols].apply(lambda x: 1 if x > 0 else 0)
```


```python
# Check data Frame to ensure flags correctly applied
df_sum.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>loc_1</th>
      <th>loc_10</th>
      <th>loc_100</th>
      <th>loc_1000</th>
      <th>loc_1002</th>
      <th>loc_1005</th>
      <th>loc_1006</th>
      <th>loc_1007</th>
      <th>loc_1008</th>
      <th>loc_1009</th>
      <th>...</th>
      <th>r_t_5</th>
      <th>r_t_6</th>
      <th>r_t_7</th>
      <th>r_t_8</th>
      <th>r_t_9</th>
      <th>s_t_1</th>
      <th>s_t_2</th>
      <th>s_t_3</th>
      <th>s_t_4</th>
      <th>s_t_5</th>
    </tr>
    <tr>
      <th>location</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>100</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1000</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1002</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 1324 columns</p>
</div>



Create another data frame for the numeric variables and compress the data by location


```python
# Create a data frame for the numerical variables
df_extra=pd.DataFrame()
df_extra['location']=data['location']
df_extra['volume']=data['volume']
# Sum up all numeric variables by location
df_extra = df_extra.groupby(by='location').sum()
```


```python
# Check to ensure data is summed and grouped by location
df_extra.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>volume</th>
    </tr>
    <tr>
      <th>location</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>664</td>
    </tr>
    <tr>
      <th>10</th>
      <td>20</td>
    </tr>
    <tr>
      <th>100</th>
      <td>246</td>
    </tr>
    <tr>
      <th>1000</th>
      <td>29</td>
    </tr>
    <tr>
      <th>1002</th>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



Create another data frame for the dependent variable and compress the data by location


```python
# create new dataframe with independent variable (fault severity) Group by location and take the mean of the fault_severities

df_independent= data.copy()

del df_independent['id']
del df_independent['log_feature']
del df_independent['volume']
del df_independent['event_type']
del df_independent['resource_type']
del df_independent['severity_type']

df_independent['fault_severity']=df_independent['fault_severity'].apply(lambda x: float(x))

df_independent = df_independent.groupby(by='location').mean()
df_independent['fault_severity'] = df_independent['fault_severity'].apply(lambda x: round(x, 0))

```

# Step 4: Modeling 

Combine the independent data frames and store the independent and dependent data frames in variables for the modeling process



```python
# combine the categorical and numerical dataframes and store in X
x=df_extra.join(df_sum)
# store the independnt data frame in y
y=df_independent['fault_severity']
```

Import train test split and split the data


```python
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
```

    C:\Users\smont\Anaconda2\lib\site-packages\sklearn\cross_validation.py:44: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. Also note that the interface of the new CV iterators are different from that of this module. This module will be removed in 0.20.
      "This module will be removed in 0.20.", DeprecationWarning)
    

Import the necessary modeling libraries and fit the model


```python
from sklearn.ensemble import GradientBoostingClassifier 

# create the model
gradient = GradientBoostingClassifier()
# fit the model 
g = gradient.fit(X_train, y_train)
```

Predict using the model and store in a list for later use


```python
predicted=g.predict(x)
```

# Step 5 Evaluation


```python
# Get the Accuracy score (percent of times algorthim correctly predicted the dependent variable (fault_severity))

g.score(X_test, y_test)
```




    0.70967741935483875



# Step 6 Deployment

Create Prediction/Probability DataFrame


```python
# Create data frame with location and columns for each possible predicted outcome
df_prob=pd.DataFrame(columns = ['location','Predicted','Probablity 0','Probablity 1', 'Probablity 2'])
df_prob['location']=data['location'].unique()
# Store the predicted outcome in the predicted outcome column 
df_prob['Predicted']=predicted

```


```python
# Store the probablities for 0, 1 ,2 fault severity in a list
probablities=g.predict_proba(x)
```


```python
# Convert list to an array 
probablities=np.array(probablities)
```

Put the probablities of each outcome into the created data frame


```python
# define lists to store the probabilities for each column in the data frame
rows=range(0,929)
cols=[0,1,2]
list_1=[]
list_2=[]
list_3=[]

# Create loop that stores the probabilities for each column into a list
for row in rows:
    for col in cols:
        if col==0:
            num=probablities[row][col]
            list_1.append(num)
        if col==1:
            num=probablities[row][col]
            list_2.append(num)
        if col==2:
            num=probablities[row][col]
            list_3.append(num)

```


```python
# Assign lists to the probability columns

df_prob['Probablity 0']=list_1
df_prob['Probablity 1']=list_2
df_prob['Probablity 2']=list_3
```

Check the data frame


```python
df_prob.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>location</th>
      <th>Predicted</th>
      <th>Probablity 0</th>
      <th>Probablity 1</th>
      <th>Probablity 2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>118</td>
      <td>1.0</td>
      <td>0.076023</td>
      <td>0.906411</td>
      <td>0.017566</td>
    </tr>
    <tr>
      <th>1</th>
      <td>91</td>
      <td>0.0</td>
      <td>0.599183</td>
      <td>0.382706</td>
      <td>0.018111</td>
    </tr>
    <tr>
      <th>2</th>
      <td>152</td>
      <td>0.0</td>
      <td>0.876537</td>
      <td>0.113166</td>
      <td>0.010297</td>
    </tr>
    <tr>
      <th>3</th>
      <td>931</td>
      <td>0.0</td>
      <td>0.625127</td>
      <td>0.357935</td>
      <td>0.016938</td>
    </tr>
    <tr>
      <th>4</th>
      <td>120</td>
      <td>0.0</td>
      <td>0.709597</td>
      <td>0.277281</td>
      <td>0.013122</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
