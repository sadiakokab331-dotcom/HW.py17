import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = sns.load_dataset('tips')

print(df.head())

sns.set_theme(style='whitegrid')

plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df, x='total_bill', y='tip', hue='time', style='sex', s=80
)
plt.title('Total Bill vs Tip Amount')
plt.xlabel('Total Bill ($)')
plt.ylabel('Tip ($)')
plt.show()

sns.pairplot(df, hue='smoker', diag_kind='kde')
plt.suptitle('Pairwise Relationships Across Variables', y=1.02)
plt.show()

plt.figure(figsize=(7, 5))
numeric_df = df.select_dtypes(
    include=['float64', 'int64']
) 
correlation_matrix = numeric_df.corr()

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap of Numerical Variables')
plt.show()

sns.jointplot(data=df, x='total_bill', y='tip', kind='reg', color='teal')
plt.show()