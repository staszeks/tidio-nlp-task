import pandas as pd
import numpy as np

# Charts
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

from transformers import BertTokenizer
from langdetect import detect

# Set the style for all charts
plt.style.use('seaborn-whitegrid')

# Create tailoerd color palette for graphs
colors_palette = [
    '#2C82FF', 
    '#15C2FF',
    '#344C9F',
    '#FFC859',
    '#6500E6',
]

def text_statistics(df_input, text_column_name):
    
    # Tokenize pandas.series using default BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    tokenized = []
    for idx, val in enumerate(df_input[text_column_name]):
        tokenized.append(tokenizer.tokenize(str(val)))
    
    df_output = pd.DataFrame(zip(tokenized), columns = [text_column_name])
    
    # Calculate statistics
    over_128 = sum(df_output[text_column_name].str.len() > 128)/df_output.shape[0]*100
    over_256 = sum(df_output[text_column_name].str.len() > 256)/df_output.shape[0]*100
    over_512 = sum(df_output[text_column_name].str.len() > 512)/df_output.shape[0]*100
    mean_len = df_output[text_column_name].str.len().mean()
    median_len = df_output[text_column_name].str.len().median()
    min_len = df_output[text_column_name].str.len().min()
    max_len = df_output[text_column_name].str.len().max()
    
    # Print output
    print(f"Number of rows in input dataframe: {df_output.shape[0]} rows\n")
    
    print(f"Percent of records with tokens over 128: {round(over_128,3)}%")
    print(f"Percent of records with tokens over 256: {round(over_256,3)}%")
    print(f"Percent of records with tokens over 512: {round(over_512,3)}%\n")
    
    print(f"Mean length of record: {round(mean_len, 1)} tokens")
    print(f"Median length of record: {median_len} tokens")
    print(f"Minimum length of record: {min_len} tokens")
    print(f"Maximum length of record: {max_len} tokens")
    
    # Create histogram
    sns.distplot(df_output[text_column_name].str.len(), bins=[x * 25 for x in range(20)])
    plt.yticks([])
    plt.xlim([0, 1000])
    plt.xlabel('Number of tokens')

def plot_sentiment_ratio(df, column_name, negative_ratio):
    
    df_pivot = df.pivot_table(
        values='conversation_id',
        index=column_name,
        columns='sentiment',
        aggfunc='nunique',
        fill_value=0,
    )

    df_pivot = df_pivot.reset_index()
    df_pivot.columns = [column_name, 'negative', 'positive']
    df_pivot.columns.name = ''
    
    if column_name == 'operator_id': # delete 0 as operator = sender
        df_pivot = df_pivot[df_pivot[column_name]!=str(0)]
        
    df_pivot = df_pivot[(df_pivot['negative'] != 0) & (df_pivot['positive'] != 0)]
    df_pivot['ratio'] = df_pivot['negative'] / df_pivot['positive']
    df_pivot = df_pivot.sort_values(by='ratio', ascending=False).reset_index(drop=True)

    df_pivot['ratio'] = df_pivot['ratio'].round(3) * 100
    
    top_5_worst = df_pivot.iloc[:5,:]
    top_5_best = df_pivot.iloc[-5:,:]
    
    fig = px.scatter(df_pivot, 
                 x="negative", 
                 y="positive",
                 hover_name=column_name, 
                 hover_data=['ratio'],
                 color_discrete_sequence = ["yellow"],
                 log_y=True,
                 log_x=True,
                )

    fig.update_layout(
        plot_bgcolor='rgb(0,0,0)', 
        showlegend=False, 
        margin=dict(pad=8),
        title={
            'text': f'negative vs positve for {column_name}<br> red line is global ratio={negative_ratio*100}%',
            'y': 0.97,
            'x': 0.5,
            'font': {
                'family': 'Calibri',
                'size': 24,
                'color': 'black',
            }
        },
    )

    fig.add_trace(go.Scatter(x=[0, df_pivot.negative.max()],
                             y=[0,df_pivot.negative.max() / negative_ratio],
                             line=dict(color='firebrick', width=4), hoverinfo='skip'
                            )
                 )

    fig.update_layout(legend=dict(
        y=1.25,
        x=0.01
    ))

    # Change hover descriptipn
    fig.data[0].hovertemplate = fig.data[0].hovertemplate.replace("%{hovertext}", f"{column_name}: " + "%{hovertext}")

    fig.update_xaxes(showgrid=False, zeroline=False, visible=False)
    fig.update_yaxes(showgrid=False, zeroline=False, visible=False)
    
    return fig, top_5_worst, top_5_best