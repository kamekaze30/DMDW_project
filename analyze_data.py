import pandas as pd

# Load the dataset
df = pd.read_csv('diabetes.csv')

print('=== DEMOGRAPHIC STATISTICS ===')
print(f'Total Samples: {len(df)}')
print(f'Diabetes Cases: {df.Outcome.sum()} ({df.Outcome.mean()*100:.1f}%)')
print(f'Non-diabetes: {len(df)-df.Outcome.sum()} ({(1-df.Outcome.mean())*100:.1f}%)')
print()

print('=== AGE DISTRIBUTION ===')
print(df.Age.describe())
print()

print('=== BMI STATISTICS ===')
print(df.BMI.describe())
print()

print('=== GLUCOSE LEVELS ===')
print(df.Glucose.describe())
print()

print('=== PREGNANCIES ===')
print(df.Pregnancies.value_counts().head())
print()

print('=== OUTCOME BY AGE GROUP ===')
df['AgeGroup'] = pd.cut(df['Age'], bins=[0,30,40,50,100], labels=['<30','30-40','40-50','50+'])
print(pd.crosstab(df['AgeGroup'], df['Outcome'], normalize='index')*100)
