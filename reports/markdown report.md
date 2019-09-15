

```python
# Imports
import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import sem
```


```python
# Raw data import
ct_df = pd.read_csv("data/clinical_trial_data.csv")
md_df = pd.read_csv("data/mouse_drug_data.csv")

merged_df = pd.merge(ct_df, md_df, on="Mouse ID")
merged_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Mouse ID</th>
      <th>Timepoint</th>
      <th>Tumor Volume (mm3)</th>
      <th>Metastatic Sites</th>
      <th>Drug</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>b128</td>
      <td>0</td>
      <td>45.000000</td>
      <td>0</td>
      <td>Capomulin</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b128</td>
      <td>5</td>
      <td>45.651331</td>
      <td>0</td>
      <td>Capomulin</td>
    </tr>
    <tr>
      <th>2</th>
      <td>b128</td>
      <td>10</td>
      <td>43.270852</td>
      <td>0</td>
      <td>Capomulin</td>
    </tr>
    <tr>
      <th>3</th>
      <td>b128</td>
      <td>15</td>
      <td>43.784893</td>
      <td>0</td>
      <td>Capomulin</td>
    </tr>
    <tr>
      <th>4</th>
      <td>b128</td>
      <td>20</td>
      <td>42.731552</td>
      <td>0</td>
      <td>Capomulin</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Data cleaning and engineering/grouping pre-process before visualization

merged_df.loc[(merged_df['Drug'] != 'Ceftamin') & \
              (merged_df['Drug'] !='Naftisol') & \
              (merged_df['Drug'] !='Naftisol') & \
              (merged_df['Drug'] != 'Propriva') & \
              (merged_df['Drug'] !='Ramicane') & \
              (merged_df['Drug'] !='Stelasyn') & \
              (merged_df['Drug'] !='Zoniferol') ,:]

group_drug_df = merged_df.groupby(['Drug','Timepoint'])
TumorVolumeMean_df = group_drug_df.mean()['Tumor Volume (mm3)']
TumorVolumeMean_df.head()

TVM_indexreset_df = TumorVolumeMean_df.reset_index()
TVM_indexreset_df.head()

CapomulinMeanVolume = TVM_indexreset_df.loc[TVM_indexreset_df['Drug'] == 'Capomulin']
InfubinolMeanVolume = TVM_indexreset_df.loc[TVM_indexreset_df['Drug'] == 'Infubinol']
KetaprilMeanVolume = TVM_indexreset_df.loc[TVM_indexreset_df['Drug'] == 'Ketapril']
PlaceboMeanVolume = TVM_indexreset_df.loc[TVM_indexreset_df['Drug'] == 'Placebo']
```

# Tumor volume change over time per treatment


```python
fig, ax = plt.subplots()

plt.scatter(CapomulinMeanVolume['Timepoint'],CapomulinMeanVolume['Tumor Volume (mm3)'], label = 'Capomulin', marker="o", facecolors="red", edgecolors="black",alpha=0.75)
plt.scatter(InfubinolMeanVolume['Timepoint'],InfubinolMeanVolume['Tumor Volume (mm3)'], label = 'Infubinol',marker="^", facecolors="blue", edgecolors="black",alpha=0.75)
plt.scatter(KetaprilMeanVolume['Timepoint'],KetaprilMeanVolume['Tumor Volume (mm3)'], label = 'Ketapril',marker="s", facecolors="green", edgecolors="black",alpha=0.75)
plt.scatter(PlaceboMeanVolume['Timepoint'],PlaceboMeanVolume['Tumor Volume (mm3)'], label = 'Placebo',marker="D", facecolors="black", edgecolors="black",alpha=0.75)

plt.xlim(0, max(PlaceboMeanVolume['Timepoint']+1))
plt.title('Tumor Response To Treatment')
plt.xlabel("Timepoint (Days)")
plt.ylabel("Tumor Volume (mm3)")
plt.legend(loc="best")
plt.grid(b=True,axis='both')

plt.plot(CapomulinMeanVolume['Timepoint'],CapomulinMeanVolume['Tumor Volume (mm3)'], linestyle='--',linewidth=0.7, color="red")
plt.plot(InfubinolMeanVolume['Timepoint'],InfubinolMeanVolume['Tumor Volume (mm3)'], linestyle='--',linewidth=0.7,color="blue")
plt.plot(KetaprilMeanVolume['Timepoint'],KetaprilMeanVolume['Tumor Volume (mm3)'], linestyle='--',linewidth=0.7,color="green")
plt.plot(PlaceboMeanVolume['Timepoint'],PlaceboMeanVolume['Tumor Volume (mm3)'], linestyle='--',linewidth=0.7,color="black")

plt.errorbar(CapomulinMeanVolume['Timepoint'], CapomulinMeanVolume['Tumor Volume (mm3)'], yerr = sem(CapomulinMeanVolume['Tumor Volume (mm3)']),linestyle ="--",fmt = 'o',capsize=3,capthick=1,markeredgecolor='red')
plt.errorbar(InfubinolMeanVolume['Timepoint'],InfubinolMeanVolume['Tumor Volume (mm3)'], yerr = sem(InfubinolMeanVolume['Tumor Volume (mm3)']),linestyle ="--",fmt = 'o',capsize=3,capthick=1,markeredgecolor='blue')
plt.errorbar(KetaprilMeanVolume['Timepoint'],KetaprilMeanVolume['Tumor Volume (mm3)'],  yerr = sem(KetaprilMeanVolume['Tumor Volume (mm3)']),linestyle ="--",fmt = 'o',capsize=3,capthick=1,markeredgecolor='green')
plt.errorbar(PlaceboMeanVolume['Timepoint'],PlaceboMeanVolume['Tumor Volume (mm3)'],  yerr = sem(PlaceboMeanVolume['Tumor Volume (mm3)']),linestyle ="--",fmt = 'o',capsize=3,capthick=1,markeredgecolor='black')

plt.savefig('reports/figures/tumor_response.png')
plt.show()
```

![png](C:\Users\tyu-dev\repo_folder\pymaceuticals-inc\reports\figures\tumor_response.png)


# Metastatic site change over time 


```python
fig, ax = plt.subplots()
metaStaticMean_df = group_drug_df.mean()['Metastatic Sites']
MSS_indexreset_df = metaStaticMean_df.reset_index()
MSS_indexreset_df.head()

CapomulinMeanVolumeMSS = MSS_indexreset_df.loc[MSS_indexreset_df['Drug'] == 'Capomulin']
InfubinolMeanVolumeMSS = MSS_indexreset_df.loc[MSS_indexreset_df['Drug'] == 'Infubinol']
KetaprilMeanVolumeMSS = MSS_indexreset_df.loc[MSS_indexreset_df['Drug'] == 'Ketapril']
PlaceboMeanVolumeMSS = MSS_indexreset_df.loc[MSS_indexreset_df['Drug'] == 'Placebo']

plt.scatter(CapomulinMeanVolumeMSS['Timepoint'],CapomulinMeanVolumeMSS['Metastatic Sites'], label = 'Capomulin', marker="o", facecolors="red", edgecolors="black",alpha=0.75)
plt.scatter(InfubinolMeanVolumeMSS['Timepoint'],InfubinolMeanVolumeMSS['Metastatic Sites'], label = 'Infubinol',marker="^", facecolors="blue", edgecolors="black",alpha=0.75)
plt.scatter(KetaprilMeanVolumeMSS['Timepoint'],KetaprilMeanVolumeMSS['Metastatic Sites'], label = 'Ketapril',marker="s", facecolors="green", edgecolors="black",alpha=0.75)
plt.scatter(PlaceboMeanVolumeMSS['Timepoint'],PlaceboMeanVolumeMSS['Metastatic Sites'], label = 'Placebo',marker="D", facecolors="black", edgecolors="black",alpha=0.75)

plt.xlim(0, max(PlaceboMeanVolumeMSS['Timepoint']+1))
plt.title('Metastatic Spread During Treatment')
plt.xlabel("Timepoint/Treatment Duration (Days)")
plt.ylabel("Metastatic Sites")
plt.legend(loc="best")
plt.grid(b=True,axis='both')

plt.plot(CapomulinMeanVolumeMSS['Timepoint'],CapomulinMeanVolumeMSS['Metastatic Sites'], linestyle='--',linewidth=0.7, color="red")
plt.plot(InfubinolMeanVolumeMSS['Timepoint'],InfubinolMeanVolumeMSS['Metastatic Sites'], linestyle='--',linewidth=0.7,color="blue")
plt.plot(KetaprilMeanVolumeMSS['Timepoint'],KetaprilMeanVolumeMSS['Metastatic Sites'], linestyle='--',linewidth=0.7,color="green")
plt.plot(PlaceboMeanVolumeMSS['Timepoint'],PlaceboMeanVolumeMSS['Metastatic Sites'], linestyle='--',linewidth=0.7,color="black")

plt.errorbar(CapomulinMeanVolumeMSS['Timepoint'], CapomulinMeanVolumeMSS['Metastatic Sites'], yerr = sem(CapomulinMeanVolumeMSS['Metastatic Sites']),linestyle ="--",fmt = 'o',capsize=3,capthick=1,markeredgecolor='red')
plt.errorbar(InfubinolMeanVolumeMSS['Timepoint'],InfubinolMeanVolumeMSS['Metastatic Sites'], yerr = sem(InfubinolMeanVolumeMSS['Metastatic Sites']),linestyle ="--",fmt = 'o',capsize=3,capthick=1,markeredgecolor='blue')
plt.errorbar(KetaprilMeanVolumeMSS['Timepoint'],KetaprilMeanVolumeMSS['Metastatic Sites'],  yerr = sem(KetaprilMeanVolumeMSS['Metastatic Sites']),linestyle ="--",fmt = 'o',capsize=3,capthick=1,markeredgecolor='green')
plt.errorbar(PlaceboMeanVolumeMSS['Timepoint'],PlaceboMeanVolumeMSS['Metastatic Sites'],  yerr = sem(PlaceboMeanVolumeMSS['Metastatic Sites']),linestyle ="--",fmt = 'o',capsize=3,capthick=1,markeredgecolor='black')

plt.savefig('reports/figures/metastatic_spread.png')
plt.show()
```


![png](C:\Users\tyu-dev\repo_folder\pymaceuticals-inc\reports\figures\metastatic_spread.png)


# Survival rate over time


```python
fig, ax = plt.subplots()

SR_df = merged_df.groupby(['Drug', 'Timepoint']).count()['Mouse ID']
SR_indexreset_df = SR_df.reset_index()
SR_indexreset_df.head()

CapomulinMeanVolumeSR = SR_indexreset_df.loc[SR_indexreset_df['Drug'] == 'Capomulin']
InfubinolMeanVolumeSR = SR_indexreset_df.loc[SR_indexreset_df['Drug'] == 'Infubinol']
KetaprilMeanVolumeSR = SR_indexreset_df.loc[SR_indexreset_df['Drug'] == 'Ketapril']
PlaceboMeanVolumeSR = SR_indexreset_df.loc[SR_indexreset_df['Drug'] == 'Placebo']

def testfunc(num1):
    num1 = float(num1)
    percentage = num1/26
    return percentage

SR_indexreset_df= pd.pivot_table(SR_indexreset_df, index='Timepoint', columns='Drug', values='Mouse ID', aggfunc = testfunc)
SR_indexreset_df= SR_indexreset_df

plt.scatter(CapomulinMeanVolumeSR['Timepoint'],CapomulinMeanVolumeSR['Mouse ID'], label = 'Capomulin', marker="o", facecolors="red", edgecolors="black",alpha=0.75)
plt.scatter(InfubinolMeanVolumeSR['Timepoint'],InfubinolMeanVolumeSR['Mouse ID'], label = 'Infubinol',marker="^", facecolors="blue", edgecolors="black",alpha=0.75)
plt.scatter(KetaprilMeanVolumeSR['Timepoint'],KetaprilMeanVolumeSR['Mouse ID'], label = 'Ketapril',marker="s", facecolors="green", edgecolors="black",alpha=0.75)
plt.scatter(PlaceboMeanVolumeSR['Timepoint'],PlaceboMeanVolumeSR['Mouse ID'], label = 'Placebo',marker="D", facecolors="black", edgecolors="black",alpha=0.75)

plt.xlim(0, max(PlaceboMeanVolumeSR['Timepoint']+1))      
plt.title('Survival Rate During Treatment')
plt.xlabel('Timepoint/Treatment Duration (Days)')
plt.ylabel("Survival Rate")
plt.legend(loc="best")


plt.plot(CapomulinMeanVolumeSR['Timepoint'],CapomulinMeanVolumeSR['Mouse ID'], linestyle='--',linewidth=0.7, color="red")
plt.plot(InfubinolMeanVolumeSR['Timepoint'],InfubinolMeanVolumeSR['Mouse ID'], linestyle='--',linewidth=0.7,color="blue")
plt.plot(KetaprilMeanVolumeSR['Timepoint'],KetaprilMeanVolumeSR['Mouse ID'], linestyle='--',linewidth=0.7,color="green")
plt.plot(PlaceboMeanVolumeSR['Timepoint'],PlaceboMeanVolumeSR['Mouse ID'], linestyle='--',linewidth=0.7,color="black")

vals = ax.get_yticks()
ax.set_yticklabels(['{:3.2f}%'.format(x*4) for x in vals])
SR_indexreset_df.head()

plt.savefig('reports/figures/survival_rate.png')
plt.show()
```


![png](C:\Users\tyu-dev\repo_folder\pymaceuticals-inc\reports\figures\survival_rate.png)



```python

```


```python
#Calcuate percentage change for tumor volume chart
def return_pc_45(df):
    cmvc0 =df.loc[df['Timepoint'] == 0,'Tumor Volume (mm3)']
    cmvc45=df.loc[df['Timepoint'] == 45,'Tumor Volume (mm3)']
    cmvc_percentchange=(cmvc0.values[0] - cmvc45.values[0])/cmvc0.values[0]*100
    return np.round(cmvc_percentchange,decimals=2)

print(
return_pc_45(CapomulinMeanVolume),
return_pc_45(PlaceboMeanVolume),
return_pc_45(InfubinolMeanVolume),
return_pc_45(KetaprilMeanVolume)
)
```

    19.48 -51.3 -46.12 -57.03



```python
pc_45_list = [return_pc_45(CapomulinMeanVolume),return_pc_45(PlaceboMeanVolume),\
              return_pc_45(InfubinolMeanVolume),return_pc_45(KetaprilMeanVolume)]
print(pc_45_list)

#Switch negative and positive for chart
pc_45_list=np.negative(pc_45_list)
print(pc_45_list)

#Color list based upon value
colors = []
for value in pc_45_list:
    if value < 0:
        colors.append('red')
    else:
        colors.append('green')
        
print(colors)
```

    [19.48, -51.3, -46.12, -57.03]
    [-19.48  51.3   46.12  57.03]
    ['red', 'green', 'green', 'green']


# Tumor Change Over 45 Day Treatment


```python
# Bar graph comparing total % tumor volume change for each drug across the full 45 days
x=['Capomulin','Infubinol','Ketapril','Placebo']
y=pc_45_list
fig, ax = plt.subplots()
sns.set(rc={'figure.figsize':(6,6)})
sns.barplot(x,y,order=x, palette=colors, hue = colors)
ax.set_title("Tumor Change Over 45 Day Treatment")
ax.legend_.remove()
plt.grid(b=True,axis='both')
plt.axhline(y=0, color='b', linestyle='-')
plt.ylabel("% Tumor Volume Change")

plt.savefig('reports/figures/tumor_change.png')
plt.show()
```


![png](C:\Users\tyu-dev\repo_folder\pymaceuticals-inc\reports\figures\tumor_change.png)



```python

```


```python
#Debugging
#https://pandas.pydata.org/pandas-docs/stable/reshaping.html
#https://pandas.pydata.org/pandas-docs/stable/generated/pandas.pivot_table.html
def testfunc(num1):
    num1 = float(num1)
    percentage = num1/26
    return percentage

SR_df = merged_df.groupby(['Drug', 'Timepoint']).count()['Mouse ID']
SR_indexreset_df = SR_df.reset_index()

SR_indexreset_df= pd.pivot_table(SR_indexreset_df, index='Timepoint', columns='Drug', values='Mouse ID', aggfunc = testfunc)

SR_indexreset_df= SR_indexreset_df
SR_indexreset_df.head(45)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Drug</th>
      <th>Capomulin</th>
      <th>Ceftamin</th>
      <th>Infubinol</th>
      <th>Ketapril</th>
      <th>Naftisol</th>
      <th>Placebo</th>
      <th>Propriva</th>
      <th>Ramicane</th>
      <th>Stelasyn</th>
      <th>Zoniferol</th>
    </tr>
    <tr>
      <th>Timepoint</th>
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
      <th>0</th>
      <td>0.961538</td>
      <td>0.961538</td>
      <td>0.961538</td>
      <td>0.961538</td>
      <td>0.961538</td>
      <td>0.961538</td>
      <td>1.000000</td>
      <td>0.961538</td>
      <td>1.000000</td>
      <td>0.961538</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.961538</td>
      <td>0.807692</td>
      <td>0.961538</td>
      <td>0.884615</td>
      <td>0.884615</td>
      <td>0.923077</td>
      <td>0.961538</td>
      <td>0.961538</td>
      <td>0.961538</td>
      <td>0.923077</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.961538</td>
      <td>0.769231</td>
      <td>0.807692</td>
      <td>0.846154</td>
      <td>0.807692</td>
      <td>0.923077</td>
      <td>0.884615</td>
      <td>0.923077</td>
      <td>0.884615</td>
      <td>0.846154</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.923077</td>
      <td>0.730769</td>
      <td>0.807692</td>
      <td>0.730769</td>
      <td>0.807692</td>
      <td>0.769231</td>
      <td>0.653846</td>
      <td>0.923077</td>
      <td>0.884615</td>
      <td>0.807692</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.884615</td>
      <td>0.692308</td>
      <td>0.769231</td>
      <td>0.730769</td>
      <td>0.769231</td>
      <td>0.730769</td>
      <td>0.653846</td>
      <td>0.884615</td>
      <td>0.807692</td>
      <td>0.653846</td>
    </tr>
    <tr>
      <th>25</th>
      <td>0.846154</td>
      <td>0.692308</td>
      <td>0.692308</td>
      <td>0.730769</td>
      <td>0.692308</td>
      <td>0.653846</td>
      <td>0.538462</td>
      <td>0.884615</td>
      <td>0.730769</td>
      <td>0.615385</td>
    </tr>
    <tr>
      <th>30</th>
      <td>0.846154</td>
      <td>0.615385</td>
      <td>0.653846</td>
      <td>0.692308</td>
      <td>0.576923</td>
      <td>0.576923</td>
      <td>0.500000</td>
      <td>0.884615</td>
      <td>0.692308</td>
      <td>0.576923</td>
    </tr>
    <tr>
      <th>35</th>
      <td>0.846154</td>
      <td>0.538462</td>
      <td>0.461538</td>
      <td>0.653846</td>
      <td>0.576923</td>
      <td>0.538462</td>
      <td>0.384615</td>
      <td>0.807692</td>
      <td>0.615385</td>
      <td>0.538462</td>
    </tr>
    <tr>
      <th>40</th>
      <td>0.807692</td>
      <td>0.538462</td>
      <td>0.384615</td>
      <td>0.576923</td>
      <td>0.576923</td>
      <td>0.461538</td>
      <td>0.346154</td>
      <td>0.769231</td>
      <td>0.461538</td>
      <td>0.538462</td>
    </tr>
    <tr>
      <th>45</th>
      <td>0.807692</td>
      <td>0.500000</td>
      <td>0.346154</td>
      <td>0.423077</td>
      <td>0.500000</td>
      <td>0.423077</td>
      <td>0.269231</td>
      <td>0.769231</td>
      <td>0.423077</td>
      <td>0.538462</td>
    </tr>
  </tbody>
</table>
</div>




```python
# #https://matplotlib.org/1.2.1/examples/pylab_examples/errorbar_demo.html
# plt.scatter(CapomulinMeanVolume['Timepoint'],CapomulinMeanVolume['Tumor Volume (mm3)'], label = 'Capomulin', marker="o", facecolors="red", edgecolors="black",alpha=0.75)
# plt.scatter(InfubinolMeanVolume['Timepoint'],InfubinolMeanVolume['Tumor Volume (mm3)'], label = 'Infubinol',marker="^", facecolors="blue", edgecolors="black",alpha=0.75)
# plt.scatter(KetaprilMeanVolume['Timepoint'],KetaprilMeanVolume['Tumor Volume (mm3)'], label = 'Ketapril',marker="s", facecolors="green", edgecolors="black",alpha=0.75)
# plt.scatter(PlaceboMeanVolume['Timepoint'],PlaceboMeanVolume['Tumor Volume (mm3)'], label = 'Placebo',marker="D", facecolors="black", edgecolors="black",alpha=0.75)


# group_drug_df = merged_df.groupby(['Drug','Timepoint'])
# TumorVolumeMean_df = group_drug_df.mean()['Tumor Volume (mm3)']
# TumorVolumeMean_df.head()

# TVM_indexreset_df = TumorVolumeMean_df.reset_index()
# TVM_indexreset_df.head()

# CapomulinMeanVolume = TVM_indexreset_df.loc[TVM_indexreset_df['Drug'] == 'Capomulin']
# InfubinolMeanVolume = TVM_indexreset_df.loc[TVM_indexreset_df['Drug'] == 'Infubinol']
# KetaprilMeanVolume = TVM_indexreset_df.loc[TVM_indexreset_df['Drug'] == 'Ketapril']
# PlaceboMeanVolume = TVM_indexreset_df.loc[TVM_indexreset_df['Drug'] == 'Placebo']

# plt.grid(b=True,axis='both')
# plt.show()
```


```python

```
