#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing relevant libraries
import pandas as pd
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt


# In[2]:


# Loading the dataset
data = pd.read_csv('UCDavisData.csv')

# Test data loaded
print(data)


# In[3]:


# Create a pandas dataframe
df = pd.DataFrame(data)

# Test dataframe
print(type(df))


# In[4]:


# 1.
# Central Tendency
# Mean values in the distribution
mean = df.mean()
print("Mean \n", mean)

# Median values in the distribution
median = df.median()
print("Median \n", median)

# Mode in the distribution
mode = df.mode()
print("Mode \n", mode)


# In[75]:


# # 2.
# # Histogram

# Height histogram plot
height = df['Height']
plt.figure()
plt.title("Height Histogram plot")
plt.xlabel("Height")
plt.ylabel("Frequency")
heightPlot=height.hist(grid=False,bins=10)

# Dad height histogram plot
dadHeight = df['dadheight']
plt.figure()
plt.title("Dad Height Histogram plot")
plt.xlabel("Height")
plt.ylabel("Frequency")
dadHeight.hist(grid=False,bins=10)

# Mom height histogram plot
momHeight = df['momheight']
plt.figure()
plt.title("Mom Height Histogram plot")
plt.xlabel("Height")
plt.ylabel("Frequency")
momHeight.hist(grid=False,bins=10)

# Creating a dataframe consisting of heights
dfheight = pd.DataFrame(data, columns=['Height', 'dadheight', 'momheight'])
print(dfheight)

# Histogram for heights
dfheight.hist()


# In[76]:


# Box Plot
# Box plot
# All data plot
# df.boxplot()  # box plot
dfheight.boxplot()


# In[ ]:


# 3.
"""
They indicate how much the observations are spread out around, let’s say, “a center”.
"""


# In[80]:


# 4
# Frequency tables
# height
height_frequency = dfheight['Height'].value_counts(normalize=True) * 100
print(height_frequency)

# momheight
momheight_frequency = dfheight['momheight'].value_counts(normalize=True) * 100
print(momheight_frequency)

# dadheight
dadheight_frequency = dfheight['dadheight'].value_counts(normalize=True) * 100
print(dadheight_frequency)

# All data
frequency_count = dfheight.value_counts(normalize=True) * 100
print(frequency_count)


# In[90]:


# 5.
# Bar plot
dfHeightBar=dfheight.head()
dfHeightBar.plot.bar()


# In[92]:


# Pie Plot
dfHeightPie=dfheight.head()
dfHeightPie.plot.pie(subplots=True)


# In[94]:


dfheight['Height'].head().plot.pie(subplots=True)


# In[95]:


dfheight['dadheight'].head().plot.pie(subplots=True)


# In[98]:


df['momheight'].tail().plot.pie(subplots=True)


# In[ ]:


# 6.
"""
The frequency distribution of the height, dadheight and momheight are normally distributed
"""


# In[101]:


# 7.
# Contingency table

pivot_table = pd.pivot_table(df, values='TV', columns=['computer'])
print(pivot_table)

"""
  The frequency distribution between TV and computer is inversly proportional.
  I.e the higher the computer frequency distribution, the lower the TV frequency distribution and vice versa.
"""


# In[102]:


# 8.
# TV & computer frequency
TV_frequency = df['TV'].value_counts(normalize=True) * 100
print(TV_frequency)

Computer_frequency = df['computer'].value_counts(normalize=True) * 100
print(Computer_frequency)

"""
The distribution is inversly proportional.
"""


# In[106]:


#  9.
# Grouped bar chart
groups = [df.TV, df.computer]
group_labels = ['TV', 'Computer']

# Convert data to pandas DataFrame.
Tv_Computer_df = pd.DataFrame(groups, index=group_labels).T

# Plot.
pd.concat(
    [Tv_Computer_df.mean().rename('average'), Tv_Computer_df.min().rename('min'), Tv_Computer_df.max().rename('max')],
    axis=1).plot.bar()

# Stacked bar chart
pd.concat(
    [Tv_Computer_df.mean().rename('average'), Tv_Computer_df.min().rename('min'), Tv_Computer_df.max().rename('max')],
    axis=1).plot.bar(stacked=True)

#   100% Stacked bar chart
# stacked_bar = pd.DataFrame({'TV': df.TV, 'Computer': df.computer}, index=['TV', 'Computer'])
stacked_bar = pd.DataFrame(df.TV, df.computer).head()

stacked_bar.plot.bar(stacked=True)


# In[ ]:


# 10.
"""The stacked bar chart is a better visualization as it gives clear differences between the data compared. The other
two comparisons may be hard to read for very large datasets. """


# In[107]:


# 11.
Seat = pd.DataFrame(data, columns=['TV', 'computer', 'Sleep', 'alcohol', 'exercise', 'GPA'])
MeanSeat = Seat.mean()

print(MeanSeat)


# In[109]:


# 12.
# alcohol,GPA
# box plot
alcohol_GPA = pd.DataFrame(data, columns=['alcohol', 'GPA'])
print(alcohol_GPA)
alcohol_GPA.boxplot()


# In[110]:


# 13.
# Median
TV_median = df.TV.median()
computer_median = df.computer.median()
Sleep_median = df.Sleep.median()
alcohol_median = df.alcohol.median()
Height_median = df.Height.median()
momheight_median = df.momheight.median()
dadheight_median = df.dadheight.median()
exercise_median = df.exercise.median()
GPA_median = df.GPA.median()

print(
    f'TV:{TV_median}, Computer: {computer_median},Sleep:{Sleep_median},Alcohol:{alcohol_median},Height:{Height_median},MomHeight:{momheight_median},DadHeight:{dadheight_median},Exercise:{exercise_median},GPA:{GPA_median}')

# IQR

df_np = df._get_numeric_data().to_numpy()
print(df_np)
labels = [label for label in df]

for label, var in zip(labels, df_np):
    q3, q1 = np.percentile(var, [75, 25])
    iqr = q3 - q1
    print(f'{label}:{iqr}')

# Range
for label, var in zip(labels, df_np):
    data_range = np.max(var) - np.min(var)
    print(f'{label}:{data_range}')

# Quartiles upper
for label, var in zip(labels, df_np):
    q3 = np.percentile(var, 75)
    print(f'{label}:{q3}')

for label, var in zip(labels, df_np):
    q1 = np.percentile(var, 25)
    print(f'{label}:{q1}')


# In[111]:


# 14.
# scatter plot
df.plot.scatter(x="Height", y="dadheight")
"""
The height of the dad explains the height of the student.
"""


# In[112]:


# 15.
"""
Large number of students vs dad height comparison lies in the center to mean normal relation.
"""


# In[118]:


# 16.
# Sleep data
sleep_data = np.array(df.Sleep)

print(sleep_data)

# create 95% confidence interval for sleep data
sleep_confidence_90 = st.norm.interval(alpha=0.90, loc=np.mean(sleep_data), scale=st.sem(sleep_data))
sleep_confidence_95 = st.norm.interval(alpha=0.95, loc=np.mean(sleep_data), scale=st.sem(sleep_data))
sleep_confidence_99 = st.norm.interval(alpha=0.99, loc=np.mean(sleep_data), scale=st.sem(sleep_data))

print(f'Sleep confidence 90%: {sleep_confidence_90},\n Sleep confidence 95%: {sleep_confidence_95},\n Sleep confidence 99%: {sleep_confidence_99}')


# In[119]:


# 17.
"""
The confidence interval increases with increase in levels ie.
90%: 0.4520779082934858
95%: 0.5386840530629016
99%: 0.7079508502088796
"""


# In[120]:


# 18.
# avg momheight dadheight
df_new = df.copy()
df_new['avgMomDadHeight'] = (df.momheight + df.dadheight) / 2
print(df_new)


# In[121]:


# 19.
"""
I chose the mean of dad + mom height to help relate to the Height column.
I get the average data by adding the momheight column to the dadheight column and dividing the sum by two.
"""


# In[122]:


# 20.

avgHeights = pd.DataFrame([df_new.Height, df_new.avgMomDadHeight])
avgMeanHeight = avgHeights.mean()

"""
the overal mean range is small ie 0-4
The dad and mom height determines the Student height
"""


# In[123]:


# 21.
ucdavis_sample1 = df.sample(30)
print(ucdavis_sample1)


# In[124]:


# 22.
ucdavis_sample1.to_csv('ucdavis_sample1')

