import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv('medical_examination.csv')

# 2
df['bmi'] = df['weight'] / (df['height'] / 100) ** 2
df['overweight'] = df['bmi'].apply(lambda x: 1 if x > 25 else 0)

# 3
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)

# 4
def draw_cat_plot():
    # 5
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])
    df_cat = df_cat.rename(columns={'variable': 'variable', 'value': 'value'})

    # 6
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')

    # 7
    g = sns.catplot(x="variable", hue="value", col="cardio", data=df_cat, kind="count", height=5, aspect=1.5)

    # 8
    fig = g.fig
    g.set_axis_labels("variable", "total")  

    # 9
    fig.savefig('catplot.png')
    return fig



# 10
def draw_heat_map():
    # 11
    df_heat = df[(df['ap_lo'] <= df['ap_hi']) &
                 (df['height'] >= df['height'].quantile(0.025)) &
                 (df['height'] <= df['height'].quantile(0.975)) &
                 (df['weight'] >= df['weight'].quantile(0.025)) &
                 (df['weight'] <= df['weight'].quantile(0.975))]

    df_heat = df_heat.drop(columns=['bmi'])  # Remove 'bmi' from heatmap data

    # 12
    corr = df_heat.corr()

    # 13
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14
    fig, ax = plt.subplots(figsize=(10, 8))

    # 15
    sns.heatmap(corr, mask=mask, annot=True, fmt=".1f", cmap="coolwarm", linewidths=0.5)

    # 16
    fig.savefig('heatmap.png')
    return fig
