import pandas as pd
import seaborn as sea
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as LR
from sklearn.metrics import mean_squared_error as mse,r2_score as r2
from sklearn.model_selection import train_test_split as tts
from scipy.stats import ttest_ind

# 1. Load Dataset
main_file = pd.read_csv("../data/Sleep_issues_dataset.csv")

# 2. Basic Info
print(main_file.info())
print(main_file.describe())
print(main_file.shape())

# 2. Check missing values
print("Missing Values:\n", main_file.isnull().sum())#No empty values

# -------- EDA --------

# 3. Correlation Heatmap
plt.figure(figsize=(10,6))
corr = main_file.corr(numeric_only=True)
sea.heatmap(corr, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# 4. Distribution Check (Histogram)(Skewwness)
# sleep quality Score
plt.figure()
sea.histplot(main_file['sleep_quality_score'], kde=True)
plt.title("Sleep Quality Distribution")
plt.show()

#Cognoitive performance score
plt.figure()
sea.histplot(main_file['cognitive_performance_score'], kde=True)
plt.title("Cognitive Performance Distribution")
plt.show()

#Stress score
plt.figure()
sea.histplot(main_file['stress_score'], kde=True)
plt.title("Stress Score Distribution")
plt.show()

plt.figure()
sea.histplot(main_file['screen_time_before_bed_mins'], kde=True)
plt.title("Screen time")
plt.show()

# Outliers Visualisation

plt.figure()
sea.boxplot(x=main_file['sleep_quality_score'])
plt.title("Outliers in Sleep Quality")#Outliers found
plt.show()

plt.figure()
sea.boxplot(x=main_file['cognitive_performance_score'])
plt.title("Outliers in Cognitive Performance")#No outliers
plt.show()

plt.figure()
sea.boxplot(x=main_file['stress_score'])
plt.title("Outliers in Stress Score")#Outliers found
plt.show()

plt.figure()
sea.boxplot(x=main_file['screen_time_before_bed_mins'])
plt.title("Outliers in Screen Time")#outliers found
plt.show()


#TIme to use IQR as no normal data

df_clean = main_file.copy()
#stress_score
Q1 = df_clean['stress_score'].quantile(0.25)
Q3 = df_clean['stress_score'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
df_clean = df_clean[(df_clean['stress_score'] >= lower) & (df_clean['stress_score'] <= upper)]
#sleep_quality_score
Q1 = df_clean['sleep_quality_score'].quantile(0.25)
Q3 = df_clean['sleep_quality_score'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
df_clean = df_clean[(df_clean['sleep_quality_score'] >= lower) & (df_clean['sleep_quality_score'] <= upper)]
#Now data is cleaned to use for objectives

# OBJECTIVE 1: Lifestyle Impact
# 1. Stress vs Sleep Quality
plt.figure()
sea.scatterplot(
    data=main_file,
    x='stress_score',
    y='sleep_quality_score',
    hue='sleep_disorder_risk'
)
plt.title("Stress vs Sleep Quality (by Disorder Risk)")
plt.show()


# 2. Screen Time vs Sleep Quality (hue = stress → shows interaction)
plt.figure()
sea.scatterplot(
    data=main_file,
    x='screen_time_before_bed_mins',
    y='sleep_quality_score',
    hue='stress_score'
)
plt.title("Screen Time vs Sleep Quality (colored by Stress)")
plt.show()


# 3. Caffeine vs Sleep Quality (hue = disorder risk again for clarity)
plt.figure()
sea.scatterplot(
    data=main_file,
    x='caffeine_mg_before_bed',
    y='sleep_quality_score',
    hue='sleep_disorder_risk'
)
plt.title("Caffeine vs Sleep Quality (by Disorder Risk)")
plt.show()


# 4. Alcohol vs Sleep Quality (hue = stress → combined effect)
plt.figure()
sea.scatterplot(
    data=main_file,
    x='alcohol_units_before_bed',
    y='sleep_quality_score',
    hue='stress_score'
)
plt.title("Alcohol vs Sleep Quality (colored by Stress)")
plt.show()


#Objective 2: Linear regression to predict Sleep Quality on Cognitive Performance
X = df_clean[['sleep_quality_score']]
y = df_clean['cognitive_performance_score']
X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2, random_state=42)
model = LR()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
ms_error = mse(y_test, y_pred)
r_sq = r2(y_test, y_pred)
print("Objective 2->Result:")
print("Mean Squared Error:", ms_error)
print("R² Score:", r_sq)
print(f"Intercept: {model.intercept_:.2f}")
print(f"Coefficient: {model.coef_[0]:.2f}")
# Interpretation: Higher sleep quality is associated with better cognitive performance


# Visualization (Line Plot)
plot_df = pd.DataFrame({
    'Sleep Quality': X_test.values.flatten(),
    'Actual Cognitive Score': y_test.values,
    'Predicted Cognitive Score': y_pred
})
plot_df = plot_df.sort_values(by='Sleep Quality')
plt.figure(figsize=(10,5))

# Scatter plot (Actual values)
sea.scatterplot(
    x='Sleep Quality',
    y='Actual Cognitive Score',
    data=plot_df,
    color='green',   # change this
    label='Actual'
)

# Regression line (Predicted)
sea.lineplot(
    x='Sleep Quality',
    y='Predicted Cognitive Score',
    data=plot_df,
    color='red',    # change this
    label='Regression Line'
)

plt.title("Regression: Sleep Quality vs Cognitive Performance")
plt.xlabel("Sleep Quality Score")
plt.ylabel("Cognitive Performance Score")

plt.grid(True)
plt.show()

# OBJECTIVE 3: Effect of Stress on Cognitive Performance(Hypothesis)
# To statistically analyze whether stress levels have a significant impact
# on cognitive performance.

# Hypotheses:
# H0 (Null): No significant effect
# H1 (Alternative): Significant effect
# Split data based on median stress
low_stress = df_clean[
    df_clean['stress_score'] <= df_clean['stress_score'].median()
]['cognitive_performance_score']

high_stress = df_clean[
    df_clean['stress_score'] > df_clean['stress_score'].median()
]['cognitive_performance_score']


# Perform Independent T-Test
t_stat, p_value = ttest_ind(low_stress, high_stress)

print("\n--- OBJECTIVE 3 RESULTS ---")
print("T-statistic:", t_stat)
print("P-value:", p_value)


# Decision Rule
alpha = 0.05

if p_value < alpha:
    print("Reject H0: Stress significantly affects cognitive performance")
else:
    print("Fail to reject H0: No significant effect")



#Objective 4:
# a)country wise chart->Average sleep per country
plt.figure(figsize=(10,5))
sea.barplot(
    data=main_file,
    x='country',
    y='sleep_duration_hrs',
    estimator='mean'
)
plt.title("Average Sleep Duration by Country")
plt.xticks(rotation=45)
plt.show()

#b)Occupation wise
plt.figure(figsize=(10,5))
sea.boxplot(
    data=main_file,
    x='occupation',
    y='sleep_duration_hrs'
)
plt.title("Sleep Duration Distribution by Occupation")
plt.xticks(rotation=45)
plt.show()

# c)Combined
pivot = main_file.pivot_table(
    values='sleep_duration_hrs',
    index='country',
    columns='occupation',
    aggfunc='mean'
)

plt.figure(figsize=(10,6))

sea.heatmap(pivot, annot=True, cmap='coolwarm')

plt.title("Sleep Duration by Country and Occupation")
plt.show()

# Objective 5:Effect of Stress on Sleep Quality
# To determine whether stress levels have a significant impact on sleep quality.
# Hypotheses:
# H0 (Null Hypothesis): Stress has no significant effect on sleep quality.
#                      Mean sleep quality of low-stress and high-stress groups is equal.
# H1 (Alternative Hypothesis): Stress has a significant effect on sleep quality.
#                             Mean sleep quality differs between low-stress and high-stress groups.

# Split data based on median stress
low_sleep = df_clean[
    df_clean['stress_score'] <= df_clean['stress_score'].median()
]['sleep_quality_score']

high_sleep = df_clean[
    df_clean['stress_score'] > df_clean['stress_score'].median()
]['sleep_quality_score']


# Perform T-test
t_stat, p_value = ttest_ind(low_sleep, high_sleep)

print("\n--- OBJECTIVE 5 RESULTS ---")
print("P-value:", p_value)


# Decision Rule
alpha = 0.05

if p_value < alpha:
    print("Reject H0: Stress significantly affects sleep quality")
else:
    print("Fail to reject H0: No significant effect")
