import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_style('whitegrid')
cp = sns.color_palette()

# aux
def load_data(path):
    data = pd.read_csv(path, sep=';')
    data.columns = data.columns.str.replace('.', '_')
    data['y_numeric'] = data['y'].map({'no': 0, 'yes': 1})
    return data

def get_categorical_fields(data, *extra):
    data_cat = data.select_dtypes(include='object')
    for field in extra:
        data_cat[field] = data[field]
    return data_cat

def print_cardinalities(data_cat, n_target=1):
    print(f'There are {data_cat.shape[1]-n_target} categorical features in the data set:')
    for cat_feature in data_cat.columns[:-n_target]:
        print(f'  - {cat_feature:15}(cardinality: {len(data_cat[cat_feature].unique())})')

# plots
def plot_target(data, **kwargs):
    fig, ax = plt.subplots(figsize=(4,4))
    ax = sns.countplot(x="y", data=data, ax=ax, color=cp[0]);

    ax.set_title("'Has the client\nsubscribed a term deposit?'", size=16)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.tick_params(labelsize=14)

    ax.text(
        1.75,
        ax.get_ylim()[1]/2, 
        f'NO  : {(1-(data["y_numeric"]).mean())*100:.2f}%\nYES: {((data["y_numeric"]).mean())*100:.2f}%',
        fontsize=20,
        va='center'
    )

    if kwargs.get('savefig', False):
        fig.savefig(kwargs.get('saveto', 'plots/01_target.png'), bbox_inches='tight')

    return ax

def plot_cat_prevalence(data_cat, cat_feature, **kwargs):
    
    # data manipulation
    d = data_cat.groupby(cat_feature).agg({'y_numeric': [np.mean, 'count']})
    d.columns = ["yes", "freq"]
    d['freq'] = d['freq']/d['freq'].sum()
    d.sort_values(by='freq', inplace=True)
    d *= 100

    #plotting
    ax = d.plot.barh(figsize=kwargs.get('figsize', (5,7)))
    if kwargs.get('xlim100', False):
        ax.set_xlim(0,100)
    
    # text
    ax.set_title(f'Category Prevalence: {cat_feature}', size=kwargs.get('title_size', 16))
    ax.set_xlabel("%", size=kwargs.get('xlabel_size', 14))
    ax.set_ylabel("")
    ax.tick_params(labelsize=12)
    
    # aditional info
    if kwargs.get('target_line', True):
        ax.axvline((data_cat["y_numeric"]).mean()*100, color='r', linestyle='--', label='%yes average')
    if kwargs.get('legend', True):
        ax.legend(loc=kwargs.get('legend_loc', (1.1,0)), fontsize=kwargs.get('legend_loc_size', 12))
    
    if kwargs.get('savefig', False):
        ax.get_figure().savefig(kwargs.get('saveto', f'plots/02_{cat_feature}.png'), bbox_inches='tight')

    return ax

