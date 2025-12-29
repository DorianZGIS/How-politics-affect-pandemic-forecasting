
import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.signal import argrelextrema
from datetime import timedelta 
#import seaborn as sns
from datetime import datetime
from scipy.stats import pearsonr, spearmanr, kendalltau 
from matplotlib.lines import Line2D  
import os
import geopandas as gpd
from tqdm import tqdm

def correlation_topics_by_category_pie_charts(correlation_df, state_data, color_dict, threshold=0.5, layout='vertical', pie_radius=4, legend_size=32):
    """
    Analyzes the correlation topics used and their frequencies for each category (Republican, Democrat, Swing).

    Parameters:
    - correlation_df: DataFrame containing correlation data with columns ['State', 'Highest Correlation Topic'].
    - state_data: DataFrame containing state data with columns ['State', 'Category'].
    - color_dict: Dictionary containing custom colors for each unique correlation topic.
    - threshold: Threshold value for adjusting Highest Correlation Topic1.
    - layout: Can be of type: 'vertical' or 'horizontal'
    - pie_radius: Radius of the pie charts.
    - legend_size: Size of the legend.

    Returns:
    - result: DataFrame containing the count of each correlation topic for each category.
    """
    # Adjust Highest Correlation Topic1 based on the threshold
    if threshold is not None:
        correlation_df.loc[correlation_df['Correlation Value1'] < threshold, 'Highest Correlation Topic1'] = 'Correlation<0.5'

    # Group by Category and Highest Correlation Topic, and count occurrences
    result = correlation_df.groupby(['political_cluster_2020', 'Highest Correlation Topic1']).size().unstack(fill_value=0)

    # Get all unique topics from correlation_df
    all_topics = set(correlation_df['Highest Correlation Topic1'].unique())

    # Plotting the result as a pie chart with custom colors
    if layout == 'vertical':
        fig, axs = plt.subplots(len(result), figsize=(8, 6 * len(result)))
    elif layout == 'horizontal':
        num_clusters = len(result)
        fig, axs = plt.subplots(1, num_clusters, figsize=(14 * num_clusters, 20))

    for i, (cluster, data) in enumerate(result.iterrows()):
        ax = axs[i]
        data = data[data > 0]  # Exclude topics with 0 occurrences
        wedges, _, autotexts = ax.pie(data, autopct='%1.1f%%', colors=[color_dict[col] for col in data.index], radius=pie_radius, textprops={'fontweight': 'bold'})

        # Increase percentage size and remove labels
        for autotext in autotexts:
            autotext.set_fontsize(40)

        ax.set_title(f'{cluster} States Topic Distribution', size=44, fontweight='bold')
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        ax.set_xticks([])
        ax.set_yticks([])

    # Add legend for all topics
    handles = [plt.Rectangle((0, 0), 1, 1, color=color_dict[col], label=col) for col in color_dict.keys()]
    plt.subplots_adjust(right=0.8)  # Adjust subplot layout to make space for legend
    fig.legend(handles=handles, loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=legend_size)

    # Set the facecolor of the figure to white
    fig.patch.set_facecolor('white')

    plt.tight_layout()
    plt.show()

    #return result

def plot_map_corr(gdf_map_filtered, tf, color_dict, min_lag, temp_lag_intervall, states_to_exclude=['ny'], threshold=None, method='OLS', comparer='R^2'):
    # Plot the map
    fig, ax = plt.subplots(1, 1, figsize=(32, 12))
    
    # Track plotted positions to avoid overlap
    plotted_positions = set()

    # Iterate over topics and plot
    data_to_plot_in_table = pd.DataFrame()
    for topic, data in gdf_map_filtered.groupby('Highest Correlation Topic1'):
        color = color_dict[topic]  # Get color from color_dict
        if threshold is not None:
            data_below_threshold = data[data['Correlation Value1'] < threshold]
            if len(data_below_threshold) > 0:
                data_below_threshold.plot(ax=ax, color=color_dict['Correlation<0.5'], linewidth=0.8, edgecolor='0.8')
            data = data[data['Correlation Value1'] >= threshold]  # Filter data based on threshold
        
        # Exclude specified states from plotting correlation values directly on the map
        data.plot(ax=ax, color=color, linewidth=0.8, edgecolor='0.8')

        # Add lines and annotate correlation values for states not in excluded list
        data_to_plot_on_map = data[~data['state_shor'].isin(states_to_exclude)]
        for idx, row in data_to_plot_on_map.iterrows():
            centroid = row.geometry.centroid
            state_name = (row.state_shor).upper()
            correlation = int(row['Correlation Value1'])
            ax.annotate(f'{state_name}: \n {correlation:.0f}', (centroid.x, centroid.y), fontsize=12, ha='center', va='center')
        
        # Plot correlation values in an extra table for states in the excluded list
        data_to_plot_in_table = pd.concat([data_to_plot_in_table, data[data['state_shor'].isin(states_to_exclude)]], ignore_index=True)


        
    
    # Define custom sorting order based on the list of states
    sorting_order = {state: order for order, state in enumerate(states_to_exclude)}

    # Sort DataFrame based on custom sorting order
    data_to_plot_in_table['sorting_order'] = data_to_plot_in_table['state_shor'].map(sorting_order)
    data_to_plot_in_table = data_to_plot_in_table.sort_values(by='sorting_order').drop(columns='sorting_order')

    #data_to_plot_in_table = data_to_plot_in_table.reset_index()
    c=0
    c2=0
    for idx, row in data_to_plot_in_table.iterrows():
        centroid = row.geometry.centroid
        correlation = int(row['Correlation Value1'])
        state_name = (row.state_shor).upper()
        text_x, text_y = 0.9, 0.5 - c  # Text annotation coordinates in axes coordinates
        ax.text(text_x, text_y, f'{state_name}: {correlation:.0f}', transform=ax.transAxes, fontsize=10, verticalalignment='center')
        # Transform axes coordinates to data coordinates for the text annotation
        text_x_data, text_y_data = ax.transAxes.transform([text_x, text_y])
        # Draw line from text annotation to centroid
        p1 = -71
        p2 = 37 #37 crazy.... most common random number!
        ax.plot([p1, centroid.x], [p2-c2, centroid.y], color='black', linewidth=0.5, transform=ax.transData)
        c += 0.02
        c2+= 0.55 # Increment c for the next iteration
        


    # Add title and labels
    plt.title(method + ' for Social Media Posts and COVID-19 Cases \n Temporal Lag between ' + str(min_lag) + '-' + str(min_lag + temp_lag_intervall) + ' days \n Time Frame ' + str(tf + 1), size=24)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    # Create a legend for the colors
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', label=topic, markerfacecolor=color_dict.get(topic, 'grey'), markersize=10) for topic in color_dict.keys()]
    plt.legend(handles=legend_handles, title='Topic with the highest ' + comparer, loc='lower right')

    # Show the plot
    plt.show()


def plot_map_corr(gdf_map_filtered, tf, color_dict, min_lag, temp_lag_intervall, threshold=None, method='OLS',comparer='R^2' ):
    # Plot the map
    fig, ax = plt.subplots(1, 1, figsize=(32, 12))
    
    # Iterate over topics and plot
    for topic, data in gdf_map_filtered.groupby('Highest Correlation Topic1'):
        color = color_dict[topic]  # Get color from color_dict
        if threshold is not None:
            data_below_threshold = data[data['Correlation Value1'] < threshold]
            if len(data_below_threshold)>0:
                data_below_threshold.plot(ax=ax, color=color_dict['Correlation<0.5'], linewidth=0.8, edgecolor='0.8')
            data = data[data['Correlation Value1'] >= threshold]  # Filter data based on threshold
        data.plot(ax=ax, color=color, linewidth=0.8, edgecolor='0.8')

    # Add title and labels
    plt.title( method+' for Social Media Posts and COVID-19 Cases \n Temporal Lag between ' + str(min_lag) + '-' + str(min_lag + temp_lag_intervall) + ' days \n Time Frame ' + str(tf + 1), size=24)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    # Annotate correlation values
    for idx, row in gdf_map_filtered.iterrows():
        centroid = row.geometry.centroid
        correlation = row['Correlation Value1']
        ax.annotate(f'{correlation:.2f}', (centroid.x, centroid.y), fontsize=12, ha='center', va='center')

    # Create a legend for the colors
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', label=topic, markerfacecolor=color_dict.get(topic, 'grey'), markersize=10) for topic in color_dict.keys()]
    plt.legend(handles=legend_handles, title='Topic with the highest '+comparer, loc='lower right')

    # Show the plot
    plt.show()

# Best color scheme so far!
color_dict = {
    'COVID-19 BaseTopic': 'darkorange',
    'Virus': '#FFBB78',  # Light Orange
    'Symptoms': '#fd7b7b',
    'Vaccination': '#BD0303',
    'Testing': '#F03042',
    'Preventive Measures': '#BCBD22',
    'Quarantine': '#85BB65',
    'Travel Restrictions': '#3D9970',
    'Health Experts': '#E377C2',
    'Health & Care': '#9467BD',
    'Anti Narrative': '#C5FFEC',
    'Alternative Treatments': '#BDD7EE',
    'Anti Vaccines': '#00B0F0',
    'Big Pharma': '#0074D9',
    'Correlation<0.5': 'grey'
}

# # Example usage:
# tf = 3
# plot_map_corr(gdf_map_mainland[gdf_map_mainland['Time Frame'] == tf], tf, color_dict, min_lag, temp_lag_intervall, threshold=0.5)





def plot_crosscorrelation_only_positive_custom_local_minima2(df_corr, covid_id, kw, cluster, lag_max=50, custom_timeframes=['2020-05-25', '2020-08-28', '2021-01-03', '2021-03-23',
                                                                                                                                                       '2021-06-27', '2021-11-07', '2022-04-19', '2022-08-24',
                                                                                                                                                       '2022-10-28', '2022-12-03'], plotting=True,start_lag = 7):
    covid_id_col = covid_id + '_' + str(cluster)
    kw_col = kw + '_' + str(cluster)

    local_minima = [datetime.strptime(date_str, "%Y-%m-%d") for date_str in custom_timeframes]

    rss = []  # List to store cross-correlation values for each timeframe
    start_date = df_corr.index[0]
    for minima_date in local_minima:
        end_date = minima_date

        # Extract data for the specific time frame
        df_subset_covid = df_corr[covid_id_col].loc[(df_corr.index >= start_date) & (df_corr.index <= end_date)]
        df_subset_kw = df_corr[kw_col].loc[(df_corr.index >= start_date) & (df_corr.index <= end_date)]

        # Calculate and store cross-correlation values for each lag
        rs = []
        for lag in range(0, lag_max + 1):
            cor_val = df_subset_covid.corr(df_subset_kw.shift(lag))
            rs.append(cor_val)

        rss.append(rs)
        start_date = minima_date
    rss = pd.DataFrame(rss)
    rss = rss.fillna(0)
    # Plot the cross-correlation heatmap with 'RdBu_r' colormap (reversed)
    if plotting:
        fig, ax = plt.subplots(figsize=(16, 10))
        

        sns.heatmap(rss, cmap='RdBu_r', ax=ax, center=0)  # Set center to 0 for white to be at zero

        # Set y-axis ticks and labels starting from 1
        ax.set_yticks(np.arange(0.5, len(local_minima), 1))
        ax.set_yticklabels(range(1, len(local_minima) + 1))

        ax.set(
               xlim=[start_lag, (rss.shape[1])+start_lag], xlabel='Number of days the Tweets time series were shifted into the future',
               ylabel='Time frames defined by local minima in COVID-19 Cases')

        # Increase font size for headline, x-axis label, and y-axis label
        ax.title.set_fontsize(20)
        ax.xaxis.label.set_fontsize(16)
        ax.yaxis.label.set_fontsize(16)

        ax.set_xticklabels([int(int(item)) for item in ax.get_xticks()])

        # Add zero offset markers
        for i in range(len(local_minima)):
            ax.add_patch(plt.Rectangle((0+ start_lag, i), 1, 1, fill=False, edgecolor='black', lw=2))
            max_value = rss.loc[i, 0 + start_lag]
            ax.text(0 + start_lag + 0.5, i + 0.5, f"{max_value:.2f}", ha='center', va='center', color='black', fontsize=15,
                    rotation=90, weight='bold')
        
        # Add max offset markers
        for i in range(len(local_minima)):
            ax.add_patch(plt.Rectangle((lag_max, i), 1, 1, fill=False, edgecolor='black', lw=2))
            max_value = rss.loc[i, lag_max]
            ax.text(lag_max + 0.5, i + 0.5, f"{max_value:.2f}", ha='center', va='center', color='black', fontsize=15,
                    rotation=90, weight='bold')



        # Add best correlation markers
        best_lags = []
        best_lags_corr = []
        zero_lags_corr = []
        rss = rss.iloc[:,start_lag:]
        for i in range(len(local_minima)):
            max_pos = rss.loc[i].idxmax()
            max_value = rss.loc[i, max_pos]
            zero_lag_value = rss.loc[i, 0+ start_lag]
            zero_lags_corr.append(zero_lag_value)
            best_lags.append(max_pos)
            best_lags_corr.append(max_value)
            ax.add_patch(plt.Rectangle((max_pos, i), 1, 1, fill=False, edgecolor='white', lw=2))
            ax.text(max_pos + 0.5, i + 0.5, f"{max_value:.2f}", ha='center', va='center', color='white', fontsize=15,
                    rotation=90, weight='bold')

        # Increase font size for x-axis and y-axis labels
        ax.set_xlabel(ax.get_xlabel(), fontsize=20)
        ax.set_ylabel(ax.get_ylabel(), fontsize=20)

        # Show the plot
        plt.show()
        print(best_lags)
    
    else:
        # print('before: ' + str(rss))
        rss = rss.iloc[:,start_lag:]
        # print('after: ' + str(rss))
         # Add best correlation markers
        best_lags = []
        best_lags_corr = []
        zero_lags_corr = []
        for i in range(len(local_minima)):
            max_pos = rss.loc[i].idxmax()
            max_value = rss.loc[i, max_pos]
            zero_lag_value = rss.loc[i, 0+ start_lag]
            zero_lags_corr.append(zero_lag_value)
            best_lags.append(max_pos)
            best_lags_corr.append(max_value)
        
        # print('best_lags_corr: ' + str(best_lags_corr))
        # print('best_lags: ' + str(best_lags))
    return best_lags, zero_lags_corr, best_lags_corr

import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
#import seaborn as sns
import statsmodels.api as sm

def plot_OLS_only_positive_custom_local_minima(df_corr, covid_id, kw, cluster, lag_max=50, custom_timeframes=['2020-05-25', '2020-08-28', '2021-01-03', '2021-03-23',
                                                                                                                                                       '2021-06-27', '2021-11-07', '2022-04-19', '2022-08-24',
                                                                                                                                                       '2022-10-28', '2022-12-03'], plotting=True,start_lag = 7):
    covid_id_col = covid_id + '_' + str(cluster)
    kw_col = kw + '_' + str(cluster)

    local_minima = [datetime.strptime(date_str, "%Y-%m-%d") for date_str in custom_timeframes]

    rss = []  # List to store cross-correlation values for each timeframe
    start_date = df_corr.index[0]
    for minima_date in local_minima:
        end_date = minima_date

        # Extract data for the specific time frame
        df_subset_covid = df_corr[covid_id_col].loc[(df_corr.index >= start_date) & (df_corr.index <= end_date)]
        df_subset_kw = df_corr[kw_col].loc[(df_corr.index >= start_date) & (df_corr.index <= end_date)]
        rs = []
        for lag in range(0, lag_max + 1):
            X = sm.add_constant(df_subset_kw.shift(lag).fillna(0))
            model = sm.OLS(df_subset_covid.values, X)
            results = model.fit()

            # Check if the p-value for the beta coefficient is less than the significance level
            if results.pvalues[1] < 0.05:
                # If significant, store the regression coefficient as the correlation value
                cor_val = results.rsquared #params[1]
            else:
                # If not significant, set the correlation value to zero
                cor_val = 0

            rs.append(cor_val)

        rss.append(rs)
        start_date = minima_date
    rss = pd.DataFrame(rss)
    rss = rss.fillna(0)
    # Plot the cross-correlation heatmap with 'RdBu_r' colormap (reversed)
    if plotting:
        fig, ax = plt.subplots(figsize=(16, 10))
        

        sns.heatmap(rss, cmap='RdBu_r', ax=ax, center=0)  # Set center to 0 for white to be at zero

        # Set y-axis ticks and labels starting from 1
        ax.set_yticks(np.arange(0.5, len(local_minima), 1))
        ax.set_yticklabels(range(1, len(local_minima) + 1))

        ax.set(
               xlim=[start_lag, (rss.shape[1])+start_lag], xlabel='Number of days the Tweets time series were shifted into the future',
               ylabel='Time frames defined by local minima in COVID-19 Cases')

        # Increase font size for headline, x-axis label, and y-axis label
        ax.title.set_fontsize(20)
        ax.xaxis.label.set_fontsize(16)
        ax.yaxis.label.set_fontsize(16)

        ax.set_xticklabels([int(int(item)) for item in ax.get_xticks()])

        # Add zero offset markers
        for i in range(len(local_minima)):
            ax.add_patch(plt.Rectangle((0+ start_lag, i), 1, 1, fill=False, edgecolor='black', lw=2))
            max_value = rss.loc[i, 0 + start_lag]
            ax.text(0 + start_lag + 0.5, i + 0.5, f"{max_value:.2f}", ha='center', va='center', color='black', fontsize=15,
                    rotation=90, weight='bold')
        
        # Add max offset markers
        for i in range(len(local_minima)):
            ax.add_patch(plt.Rectangle((lag_max, i), 1, 1, fill=False, edgecolor='black', lw=2))
            max_value = rss.loc[i, lag_max]
            ax.text(lag_max + 0.5, i + 0.5, f"{max_value:.2f}", ha='center', va='center', color='black', fontsize=15,
                    rotation=90, weight='bold')



        # Add best correlation markers
        best_lags = []
        best_lags_corr = []
        zero_lags_corr = []
        rss = rss.iloc[:,start_lag:]
        for i in range(len(local_minima)):
            max_pos = rss.loc[i].idxmax()
            max_value = rss.loc[i, max_pos]
            zero_lag_value = rss.loc[i, 0+ start_lag]
            zero_lags_corr.append(zero_lag_value)
            best_lags.append(max_pos)
            best_lags_corr.append(max_value)
            ax.add_patch(plt.Rectangle((max_pos, i), 1, 1, fill=False, edgecolor='white', lw=2))
            ax.text(max_pos + 0.5, i + 0.5, f"{max_value:.2f}", ha='center', va='center', color='white', fontsize=15,
                    rotation=90, weight='bold')

        # Increase font size for x-axis and y-axis labels
        ax.set_xlabel(ax.get_xlabel(), fontsize=20)
        ax.set_ylabel(ax.get_ylabel(), fontsize=20)

        # Show the plot
        plt.show()
        print(best_lags)
    
    else:
        # print('before: ' + str(rss))
        rss = rss.iloc[:,start_lag:]
        # print('after: ' + str(rss))
         # Add best correlation markers
        best_lags = []
        best_lags_corr = []
        zero_lags_corr = []
        for i in range(len(local_minima)):
            max_pos = rss.loc[i].idxmax()
            max_value = rss.loc[i, max_pos]
            zero_lag_value = rss.loc[i, 0+ start_lag]
            zero_lags_corr.append(zero_lag_value)
            best_lags.append(max_pos)
            best_lags_corr.append(max_value)
        
        # print('best_lags_corr: ' + str(best_lags_corr))
        # print('best_lags: ' + str(best_lags))
    return best_lags, zero_lags_corr, best_lags_corr




def plot_local_minima_entire_us2(df_cases, mva, color_tf, color_leftout, exclude_dates=None):
    days_to_shift = mva
    df_cases.index = pd.to_datetime(df_cases.index)
    df_cases['cases'] = df_cases['cases'].rolling(window=days_to_shift).mean()

    # Create a figure with a single subplot
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot the "cases" column
    cases_line, = ax1.plot(df_cases.index, df_cases['cases'], label='Covid-19 Cases', color='#840924')
    ax1.set_ylabel('Covid-19 Cases normalized over HSA population', color='#840924')
    ax1.tick_params(axis='y', labelcolor='#840924')

    # Find local minima for the cases time series
    local_minima = find_local_extrema(df_cases['cases'])

    if exclude_dates is None:
        exclude_dates = ['2021-01-13', '2022-09-06']

    # Highlight specific dates with brighter and more transparent markers
    highlight_dates = {date: color_leftout for date in exclude_dates}
    for date, color in highlight_dates.items():
        date = pd.to_datetime(date)
        if date in df_cases.index:
            scatter_highlight = ax1.scatter(date, df_cases.loc[date, 'cases'], color=color, marker='o', edgecolors=color_leftout, alpha=0.7, s=100, label='Left out Local Minima')

    custom_timeframes = []
    for idx in range(len(local_minima)):
        if local_minima.index[idx] in pd.to_datetime(exclude_dates):
            continue
        else:
            ax1.axvline(x=local_minima.index[idx], color=color_tf, linestyle='--', alpha=0.7)
            # Plot local minima with vertical lines
            scatter_minima = ax1.scatter(local_minima.index[idx], local_minima.values[idx], color=color_tf, label='Local Minima', s=100)
            custom_timeframes.append(str(local_minima.index[idx])[:10])

    # Add title
    ax1.set_title('Epidemic waves defined through local minima in Covid-19 cases for the entire US')

    # Add grid lines
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Add legend with marker size scaling
    ax1.legend(loc='upper left', handles=[cases_line, scatter_minima, scatter_highlight], labels=['Covid-19 Cases', 'Local Minima', 'Left out Local Minima'], markerfirst=False, markerscale=0.5)

    # Set the x-axis label
    ax1.set_xlabel('Date')

    # Show the plot
    plt.tight_layout()
    plt.show()
    print(custom_timeframes)
    return custom_timeframes

# Example usage
# plot_local_minima_entire_us2(df_cases_whole_us.copy(), mva=28, color_tf='your_color', color_leftout='your_color', exclude_dates=['date1', 'date2'])



def plot_rolling_corr_given_min(df, covid_id, kw, cluster, custom_timeframes, shift=0):
    # build the columns
    tweets_id = kw + '_' + str(cluster)
    covid_id_col = covid_id + '_' + str(cluster)

    # Create a figure with subplots
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot the "cases" and "tweets" columns in the top graph
    ax1.plot(df.index, df[covid_id_col], label=covid_id, color='#840924')
    ax1.set_ylabel('Covid-19 Cases normalized over HSA population', color='#840924')
    ax1.tick_params(axis='y', labelcolor='#840924')

    ax1twin = ax1.twinx()

    # Plot the original tweet time series
    ax1twin.plot(df.index, df[tweets_id], label='Virus Topic', color='#1d5fa2')

    # Shift the tweet time series into the future by the specified amount
    if shift > 0:
        ax1twin.plot(df.index + pd.DateOffset(days=shift), df[tweets_id],
                     label=f'Virus Topic (Shifted by {shift} days)', color='gold', alpha=0.8)

    ax1twin.set_ylabel('Covid-19 related Tweets normalized over all Tweets per HSA', color='#1d5fa2')
    ax1twin.tick_params(axis='y', labelcolor='#1d5fa2')

    # Find local minima and maxima for the first time series
    local_minima = [datetime.strptime(date_str, "%Y-%m-%d") for date_str in custom_timeframes]

    # Plot local minima and maxima with vertical lines and points
    ax1.scatter(local_minima, df[df.index.isin(local_minima)][covid_id_col], color='black', label='Local Minima')

    for idx in local_minima:
        ax1.axvline(x=idx, color='black', linestyle='--', alpha=0.7)

    # Add titles
    ax1.set_title(f'Cases and Tweets Cluster {cluster} with Original and Shifted Tweet Time Series')

    # Add grid lines
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1twin.get_legend_handles_labels()
    ax1twin.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    # Set the common x-axis label
    ax1twin.set_xlabel('Date')

    # Show the plot
    plt.tight_layout()
    plt.show()

    # fig.savefig(path + '_' + str(cluster) + '_keywords_' + kw + '.png', dpi=800)

def plot_time_frames_for_given_minimas_with_shift(df, covid_id, kw, cluster, custom_timeframes=None, shifts=None, num_dates=5, title_fontsize=20, individual_title_fontsize=18, legend_fontsize=14):
    covid_id_col = covid_id + '_' + str(cluster)
    kw_col = kw + '_' + str(cluster)

    local_minima = [datetime.strptime(date_str, "%Y-%m-%d") for date_str in custom_timeframes]

    # Create a grid of subplots
    num_plots = len(local_minima)
    cols = 2
    rows = (num_plots + 1) // 2  # Add 1 to round up if needed
    fig, axes = plt.subplots(rows, cols, figsize=(18, 12))

    # Flatten the axes array to handle different cases for odd and even numbers of plots
    axes = axes.flatten()
    start_date = df.index[0]

    # Construct the title with cluster information
    cluster_title = f"Cluster {cluster}"

    for i, minima_date in enumerate(local_minima):
        ax = axes[i]

        end_date = minima_date

        # Plot the time series for the specific time frame
        df_subset = df[(df.index >= start_date) & (df.index <= end_date)]

        # Plot the original tweet time series
        ax.plot(df_subset.index, df_subset[covid_id_col], label='COVID-19 cases', color='#840924')  # Changed y-axis label
        ax.set_ylabel('COVID-19 cases', color='#840924')  # Changed y-axis label
        ax.tick_params(axis='y', labelcolor='#840924')

        # Create a secondary y-axis on the right for the second time series
        ax2 = ax.twinx()

        # Plot the original tweet time series with label "Virus Topic"
        ax2.plot(df_subset.index, df_subset[kw_col], label='Virus Topic', color='#1d5fa2')

        # Shift the tweet time series into the future by the specified amount
        if shifts and shifts[i] > 0:
            # Plot the shifted tweet time series with 80% alpha
            ax2.plot(df_subset.index + pd.DateOffset(days=min(shifts[i], 31)), df_subset[kw_col],  # Limit shift to 10 days
                     label=f'Virus Topic (Shifted by {shifts[i]} days)', color='darkorange', alpha=0.8)
        ax2.set_ylabel('Virus Topic', color='#1d5fa2')
        ax2.tick_params(axis='y', labelcolor='#1d5fa2')

        # Set x-axis ticks to show a limited number of dates
        date_ticks = df_subset.index[::max(len(df_subset.index)//num_dates, 1)]
        ax.set_xticks(date_ticks)
        ax.set_xticklabels([date.strftime("%Y-%m-%d") for date in date_ticks])

        # Add titles and legend in the middle upper position with increased font size
        title = f'Timeframe {i + 1}: {start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")}'
        ax.set_title(title, fontsize=title_fontsize)

        # Increase the font size of individual subplot titles
        ax.title.set_fontsize(individual_title_fontsize)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=legend_fontsize)  # Increase legend font size ;loc='upper right',

        # Adjust x-axis limits to prevent overlap
        ax.set_xlim(start_date, end_date)

        start_date = minima_date

    # Add a main title with cluster information
    fig.suptitle(f'Tweet Virus Topic and COVID-19 Cases Time Series for {cluster_title}', fontsize=title_fontsize + 4, y=1.02)

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()

# Example usage
# shifts = [0, 3, 2]  # Example shifts for each timeframe
# plot_time_frames_for_given_minimas_with_shift(df, 'cases', 'tweets', cluster=1, custom_timeframes=custom_timeframes, shifts=shifts, num_dates=5, title_fontsize=16, individual_title_fontsize=14, legend_fontsize=12)

def plot_time_frames_for_given_minimas(df, covid_id, kw, cluster, custom_timeframes=None, num_dates=5, title_fontsize=14, individual_title_fontsize=18):
    covid_id_col = covid_id + '_' + str(cluster)
    kw_col = kw + '_' + str(cluster)

    local_minima = [datetime.strptime(date_str, "%Y-%m-%d") for date_str in custom_timeframes]

    # Create a grid of subplots
    num_plots = len(local_minima)
    cols = 2
    rows = (num_plots + 1) // 2  # Add 1 to round up if needed
    fig, axes = plt.subplots(rows, cols, figsize=(18, 12))

    # Flatten the axes array to handle different cases for odd and even numbers of plots
    axes = axes.flatten()
    start_date = df.index[0]

    for i, minima_date in enumerate(local_minima):
        ax = axes[i]

        end_date = minima_date

        # Plot the time series for the specific time frame
        df_subset = df[(df.index >= start_date) & (df.index <= end_date)]

        # Plot the first time series on the left y-axis
        ax.plot(df_subset.index, df_subset[covid_id_col], label=covid_id, color='#840924')
        ax.set_ylabel(covid_id, color='#840924')
        ax.tick_params(axis='y', labelcolor='#840924')

        # Create a secondary y-axis on the right for the second time series
        ax2 = ax.twinx()
        ax2.plot(df_subset.index, df_subset[kw_col], label=kw, color='#1d5fa2')
        ax2.set_ylabel(kw, color='#1d5fa2')
        ax2.tick_params(axis='y', labelcolor='#1d5fa2')

        # Set x-axis ticks to show a limited number of dates
        date_ticks = df_subset.index[::max(len(df_subset.index)//num_dates, 1)]
        ax.set_xticks(date_ticks)
        ax.set_xticklabels([date.strftime("%Y-%m-%d") for date in date_ticks])

        # Add titles and legend in the middle upper position with increased font size
        title = f'Timeframe {i + 1}: {start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")}'
        ax.set_title(title, fontsize=title_fontsize)

        # Increase the font size of individual subplot titles
        ax.title.set_fontsize(individual_title_fontsize)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        start_date = minima_date

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()

# Example usage
# plot_time_frames_for_given_timeframes(df, 'cases', 'tweets', cluster=1, custom_timeframes=custom_timeframes, num_dates=5, title_fontsize=16, individual_title_fontsize=14)

def plot_rolling_corr(df, covid_id, kw, cluster):
    # build the columns
    tweets_id = kw + '_' + str(cluster)
    covid_id_col = covid_id + '_' + str(cluster)

    # Create a figure with subplots
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot the "cases" and "tweets" columns in the top graph
    ax1.plot(df.index, df[covid_id_col], label=covid_id, color='red')
    ax1.set_ylabel('Covid-19 Cases normalized over HSA population', color='red')
    ax1.tick_params(axis='y', labelcolor='red')

    ax1twin = ax1.twinx()
    ax1twin.plot(df.index, df[tweets_id], label=kw, color='blue')
    ax1twin.set_ylabel('Covid-19 related Tweets normalized over all Tweets per HSA', color='blue')
    ax1twin.tick_params(axis='y', labelcolor='blue')

    # Find local minima and maxima for the first time series
    days_to_shift = 14
    local_minima, local_maxima = find_local_extrema(df[covid_id_col].rolling(window=days_to_shift).mean())
    local_minima.index = [date - timedelta(days=days_to_shift) for date in local_minima.index]

    # Plot local minima and maxima with vertical lines
    ax1.scatter(local_minima.index, local_minima.values, color='orange', label='Local Minima')
    # ax1.scatter(local_maxima.index, local_maxima.values, color='green', label='Local Maxima')

    for idx in local_minima.index:
        ax1.axvline(x=idx, color='orange', linestyle='--', alpha=0.7)

    # for idx in local_maxima.index:
    #     ax1.axvline(x=idx, color='green', linestyle='--', alpha=0.7)

    # Add titles
    ax1.set_title('Cases and Tweets Cluster ' + str(cluster))

    # Add grid lines
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1twin.get_legend_handles_labels()
    ax1twin.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    # Set the common x-axis label
    ax1twin.set_xlabel('Date')

    # Show the plot
    plt.tight_layout()
    plt.show()

    # fig.savefig(path + '_' + str(cluster) + '_keywords_' + kw + '.png', dpi=800)


def plot_rolling_corr_best(df, covid_id, kw, cluster):
    # build the columns
    tweets_id = kw + '_' + str(cluster)
    covid_id_col = covid_id + '_' + str(cluster)

    # Create a figure with subplots
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot the "cases" and "tweets" columns in the top graph
    ax1.plot(df.index, df[covid_id_col], label=covid_id, color='#840924')
    ax1.set_ylabel('Covid-19 Cases normalized over HSA population', color='#840924')
    ax1.tick_params(axis='y', labelcolor='#840924')

    ax1twin = ax1.twinx()
    ax1twin.plot(df.index, df[tweets_id], label=kw, color='#1d5fa2')
    ax1twin.set_ylabel('Covid-19 related Tweets normalized over all Tweets per HSA', color='#1d5fa2')
    ax1twin.tick_params(axis='y', labelcolor='#1d5fa2')

    # Find local minima and maxima for the first time series
    days_to_shift = 14
    local_minima, local_maxima = find_local_extrema(df[covid_id_col].rolling(window=days_to_shift).mean())
    local_minima.index = [date - timedelta(days=days_to_shift) for date in local_minima.index]

    # Plot local minima and maxima with vertical lines
    ax1.scatter(local_minima.index, local_minima.values, color='black', label='Local Minima')
    # ax1.scatter(local_maxima.index, local_maxima.values, color='green', label='Local Maxima')

    for idx in local_minima.index:
        ax1.axvline(x=idx, color='black', linestyle='--', alpha=0.7)

    # for idx in local_maxima.index:
    #     ax1.axvline(x=idx, color='green', linestyle='--', alpha=0.7)

    # Add titles
    ax1.set_title('Cases and Tweets Cluster ' + str(cluster))

    # Add grid lines
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1twin.get_legend_handles_labels()
    ax1twin.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    # Set the common x-axis label
    ax1twin.set_xlabel('Date')

    # Show the plot
    plt.tight_layout()
    plt.show()

    # fig.savefig(path + '_' + str(cluster) + '_keywords_' + kw + '.png', dpi=800)


def plot_time_frames(df, covid_id, kw, cluster, num_dates=5):
    covid_id_col = covid_id + '_' + str(cluster)
    kw_col = kw + '_' + str(cluster)

    # Find local minima for the first time series
    days_to_shift = 14
    local_minima, _ = find_local_extrema(df[covid_id_col].rolling(window=days_to_shift).mean())
    local_minima.index = [date - timedelta(days=days_to_shift) for date in local_minima.index]
    print(list(local_minima.index))
    # Create a grid of subplots
    num_plots = len(local_minima)
    cols = 2
    rows = (num_plots + 1) // 2  # Add 1 to round up if needed
    fig, axes = plt.subplots(rows, cols, figsize=(18, 12))

    # Flatten the axes array to handle different cases for odd and even numbers of plots
    axes = axes.flatten()
    start_date = df.index[0]

    for i, minima_date in enumerate(local_minima.index):
        ax = axes[i]

        end_date = minima_date

        # Plot the time series for the specific time frame
        df_subset = df[(df.index >= start_date) & (df.index <= end_date)]

        # Plot the first time series on the left y-axis
        ax.plot(df_subset.index, df_subset[covid_id_col], label=covid_id, color='#840924')
        ax.set_ylabel(covid_id, color='#840924')
        ax.tick_params(axis='y', labelcolor='#840924')

        # Create a secondary y-axis on the right for the second time series
        ax2 = ax.twinx()
        ax2.plot(df_subset.index, df_subset[kw_col], label=kw, color='#1d5fa2')
        ax2.set_ylabel(kw, color='#1d5fa2')
        ax2.tick_params(axis='y', labelcolor='#1d5fa2')

        # Set x-axis ticks to show a limited number of dates
        date_ticks = df_subset.index[::max(len(df_subset.index)//num_dates, 1)]
        ax.set_xticks(date_ticks)
        ax.set_xticklabels([date.strftime("%Y-%m-%d") for date in date_ticks])

        # Add titles and legend in the middle upper position
        title = f'Time Frame {i + 1}: {start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")}'
        ax.set_title(title)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        start_date = minima_date

        
    # Adjust layout and show the 
    plt.tight_layout()
    plt.show()


def plot_crosscorrelation(df, covid_id, kw, cluster, lag_max=50):
    covid_id_col = covid_id + '_' + str(cluster)
    kw_col = kw + '_' + str(cluster)

    # Find local minima for the first time series
    days_to_shift = 14
    local_minima, _ = find_local_extrema(df_corr_dem[covid_id_col].rolling(window=days_to_shift).mean())
    local_minima.index = [date - timedelta(days=days_to_shift) for date in local_minima.index]

    rss = []  # List to store cross-correlation values for each timeframe
    start_date = df_corr_dem.index[0]
    for minima_date in local_minima.index:
        end_date = minima_date

        # Extract data for the specific time frame
        df_subset_covid = df_corr_dem[covid_id_col].loc[(df_corr_dem.index >= start_date) & (df_corr_dem.index <= end_date)]
        df_subset_kw = df_corr_dem[kw_col].loc[(df_corr_dem.index >= start_date) & (df_corr_dem.index <= end_date)]

        # Calculate and store cross-correlation values for each lag
        rs = []
        for lag in range(-lag_max, lag_max+1):
            rs.append(crosscorr(df_subset_covid, df_subset_kw, lag))

        rss.append(rs)
        start_date = minima_date

    # Plot the cross-correlation heatmap
    fig, ax = plt.subplots(figsize=(16, 10))
    rss = pd.DataFrame(rss)
    sns.heatmap(rss, cmap='RdBu_r', ax=ax)
    ax.set(title='Windowed Time Lagged Cross Correlation for Cluster '+str(cluster)+' and Keywords Infections',
        xlim=[0, (rss.shape[1])], xlabel='Offset \n Day for which the time series were shifted (e.g. + means Tweets were earlier)', ylabel='Timeframes defined thorugh local minima in Covid-19 Cases')
    ax.set_xticklabels([int(int(item) - lag_max) for item in ax.get_xticks()])

    # Add zero offset markers
    for i in range(len(local_minima.index)):
        ax.add_patch(plt.Rectangle((lag_max, i), 1, 1, fill=False, edgecolor='black', lw=2))

    # Add best correlation markers
    best_lags = []
    for i in range(len(local_minima.index)):
        max_pos = rss.loc[i].idxmax()
        max_value = rss.loc[i, max_pos]
        best_lags.append(max_pos-lag_max)
        ax.add_patch(plt.Rectangle((max_pos, i), 1, 1, fill=False, edgecolor='red', lw=2))
        ax.text(max_pos + 0.5, i + 0.5, f"{max_value:.2f}", ha='center', va='center', color='red', fontsize=10, rotation=90, weight='bold')
    # Show the plot
    plt.show()
    print(best_lags)

def plot_crosscorrelation_only_positive(df, covid_id, kw, cluster, lag_max=50, minima_smoothing_window=14):
    covid_id_col = covid_id + '_' + str(cluster)
    kw_col = kw + '_' + str(cluster)

    # Find local minima for the first time series
    days_to_shift = minima_smoothing_window
    local_minima, _ = find_local_extrema(df_corr_dem[covid_id_col].rolling(window=days_to_shift).mean())
    local_minima.index = [date - timedelta(days=days_to_shift) for date in local_minima.index]

    rss = []  # List to store cross-correlation values for each timeframe
    start_date = df_corr_dem.index[0]
    for minima_date in local_minima.index:
        end_date = minima_date

        # Extract data for the specific time frame
        df_subset_covid = df_corr_dem[covid_id_col].loc[(df_corr_dem.index >= start_date) & (df_corr_dem.index <= end_date)]
        df_subset_kw = df_corr_dem[kw_col].loc[(df_corr_dem.index >= start_date) & (df_corr_dem.index <= end_date)]

        # Calculate and store cross-correlation values for each lag
        rs = []
        for lag in range(0, lag_max+1):
            rs.append(crosscorr(df_subset_covid, df_subset_kw, lag))

        rss.append(rs)
        start_date = minima_date

    # Plot the cross-correlation heatmap with 'RdBu_r' colormap (reversed)
    fig, ax = plt.subplots(figsize=(16, 10))
    rss = pd.DataFrame(rss)
    
    sns.heatmap(rss, cmap='RdBu_r', ax=ax, center=0)  # Set center to 0 for white to be at zero
    
    # Set y-axis ticks and labels starting from 1
    ax.set_yticks(np.arange(0.5, len(local_minima.index), 1))
    ax.set_yticklabels(range(1, len(local_minima.index) + 1))

    ax.set(title='Windowed Time Lagged Cross Correlation for Cluster '+str(cluster)+' and Keywords Infections',
        xlim=[0, (rss.shape[1])], xlabel='Number of days the Tweets time series was shifted into the future', ylabel='Timeframes defined through local minima in COVID-19 Cases')

    # Increase font size for headline, x-axis label, and y-axis label
    ax.title.set_fontsize(20)
    ax.xaxis.label.set_fontsize(16)
    ax.yaxis.label.set_fontsize(16)

    ax.set_xticklabels([int(int(item)) for item in ax.get_xticks()])

    # Add zero offset markers
    for i in range(len(local_minima.index)):
        ax.add_patch(plt.Rectangle((0, i), 1, 1, fill=False, edgecolor='black', lw=2))

    # Add best correlation markers
    best_lags = []
    for i in range(len(local_minima.index)):
        max_pos = rss.loc[i].idxmax()
        max_value = rss.loc[i, max_pos]
        best_lags.append(max_pos)
        ax.add_patch(plt.Rectangle((max_pos, i), 1, 1, fill=False, edgecolor='red', lw=2))
        ax.text(max_pos + 0.5, i + 0.5, f"{max_value:.2f}", ha='center', va='center', color='red', fontsize=10, rotation=90, weight='bold')

    # Increase font size for x-axis and y-axis labels
    ax.set_xlabel(ax.get_xlabel(), fontsize=20)
    ax.set_ylabel(ax.get_ylabel(), fontsize=16)

    # Show the plot
    plt.show()
    print(best_lags)


def entire_tf_moving_window_temporal_lag(df, label_cases, label_tweets, cluster, lag = 31, correlation_window= 84, plotting =True):
    ''' 
    left: cases
    right: tweets
    '''
    # take some data
    df_intern = df.copy()
    d1 = df_intern[label_cases+'_'+str(cluster)] # cases
    d2 = df_intern[label_tweets+'_'+str(cluster)] # tweets

    lags = np.arange(0, (lag+1), 1)
    rs = [rolling_correlation(d1, d2.shift(lag), window=correlation_window, corr_method=pearsonr) for lag in lags] # d2 the tweets are shifted with the lag!
    maxx = max(rs)
    optimal_lag = rs.index(maxx)
    og_corr = round(rolling_correlation(d1, d2.shift(0), window=correlation_window, corr_method=pearsonr),2)
    incr_corr = round((maxx), 2)
    if plotting:
        f,ax=plt.subplots(figsize=(14,3))
        ax.plot(rs)

        ax.axvline(0 ,color='blue',linestyle='--',label='Offset')
        ax.axvline(optimal_lag,color='r',linestyle='--',label='Peak correlation')
        ax.set(title=f'Correlation between Tweets and Covid-19 cases for a correlation window of '+str(correlation_window)+' days and cluster '+str(cluster)+':\n Optimal Lag for shifting the Tweets '+str(optimal_lag)+' days increases the rolling Pearson  correlation from ' +str(og_corr) +' to '+ str(incr_corr) , xlabel='Lag in days',ylabel='Pearson correlation')

        plt.axhline(y=0, color='black', linestyle='-')
        plt.legend()
        plt.show()
    return optimal_lag, og_corr, incr_corr

def rolling_correlation(df1, df2, window, corr_method):
    if len(df1) != len(df2):
        raise ValueError("DataFrames must have the same length")

    correlations = []
    for i in range(len(df1) - window + 1):
        subset1 = df1[i:i + window]
        subset2 = df2[i:i + window]

        correlation = subset1.corr(subset2)
        correlations.append(correlation)

    return np.mean(correlations)


def find_local_extrema(data):
    # Using scipy's argrelextrema to find local minima
    local_minima_idx = argrelextrema(data.values, comparator=lambda x, y: x < y, order=5)[0]
    local_minima = data.iloc[local_minima_idx]
    return local_minima



def plot_correlation_improvement(results_dict, topics, title_add_on=''):
    fig, ax = plt.subplots(figsize=(14, 10))
    bar_width = 0.2
    bar_positions = np.arange(len(topics))

    # Define colors for each cluster
    cluster_colors = {'Republican': 'red', 'Democrat': 'blue', 'Swing': 'purple'}

    legend_lines = []
    legend_labels = []

    for i, cluster in enumerate(['Republican', 'Democrat', 'Swing']):
        corr_og_values = [results_dict[topic][cluster]['corr_og'] for topic in topics]
        corr_increased_values = [results_dict[topic][cluster]['corr_increased'] for topic in topics]
        temporal_lag_values = [results_dict[topic][cluster]['lag'] for topic in topics]

        # Plot 'Original Correlation' outside the loop
        scatter_og = ax.scatter(bar_positions + i * bar_width, corr_og_values, marker='.', s=100, color=cluster_colors[cluster], label=None)

        for j in range(len(topics)):
            scatter_increased = ax.scatter(bar_positions[j] + i * bar_width, corr_increased_values[j], marker='o', s=150, color=cluster_colors[cluster], label=None)

            # Add temporal lag as a label to the upper dot to the right
            ax.text(bar_positions[j] + i * bar_width + 0.01, corr_increased_values[j] + 0.005, f'{temporal_lag_values[j]}', ha='center', va='bottom', fontsize=12, color='black', weight='bold')

            # Connect the points with lines
            ax.plot([bar_positions[j] + i * bar_width, bar_positions[j] + i * bar_width], [corr_og_values[j], corr_increased_values[j]], color=cluster_colors[cluster], linestyle='-', linewidth=2)

        # Add to legend only once per cluster
        legend_lines.extend([scatter_og, scatter_increased])
        legend_labels.extend([f'{cluster}: Original Correlation', f'{cluster}: Increased Correlation'])

    # Add vertical lines between topics
    for j in range(len(topics)-1):
        middle_position = (bar_positions[j] + bar_positions[j+1] + bar_width*2 ) / 2 
        ax.axvline(x=middle_position, color='grey', linestyle='--', linewidth=1)
        
    # Add a black line with marker 'o' for Temporal Lag to the legend
    legend_lines.append(Line2D([0], [0], marker='$0$', color='black', markersize=10, linestyle='None', label='Temporal Lag'))
    legend_labels.append('Temporal Lag in days')

    ax.set_xlabel('Topics', fontsize=16)
    ax.set_ylabel('Pearson Correlation of social media posts and COVID-19 cases', fontsize=16)
    ax.set_title('Correlation Improvement for Each Topic and Political Cluster ' + title_add_on, fontsize=22)
    
    # Increase the size of x-axis labels and set as two lines
    ax.set_xticks(bar_positions + bar_width)
    ax.set_xticklabels([label.replace(" ", "\n") for label in topics], fontsize=14, rotation=0, ha='center')
    
    print(legend_labels)
    # Combine legend handles and labels
    ax.legend(legend_lines, legend_labels, loc='best')

    plt.show()



def plot_plotical_clusters(dfs, voting_data_state, cluster, custom_timeframes):
    
    cluster_states = voting_data_state[voting_data_state['political_cluster_2020'] == cluster]['state_shor']

    # Then, filter the rows in 'other_table' where state_shor is not in the list of swing states
    df_cases = dfs[0].copy()
    filtered_df_cases = df_cases[df_cases['Clusters'].isin(list(cluster_states))]
    filtered_df_cases
    plot_state_level_cases(pd.DataFrame({'cases':filtered_df_cases.drop('Clusters', axis=1).T.sum(axis=1)}), 'cases', vertical_dates= custom_timeframes, cluster =cluster)


def plot_state_level_cases(df, fip, vertical_dates=None, cluster=None):
    plt.figure(figsize=(16, 8))

    # Plot the main data
    plt.plot(df.index, df[fip], label='COVID-19 cases', color='#840924')

    # Set the number of date ticks to display (e.g., 10)
    num_ticks = 10
    # Calculate the step for date ticks
    step = len(df) // num_ticks
    date_ticks = df.index[::step]

    # Set the date ticks on the x-axis
    plt.xticks(date_ticks, rotation=45)  # Rotate date labels for readability

    # Draw vertical lines for specified dates
    if vertical_dates:
        for date in vertical_dates:
            plt.axvline(x=date, color='grey', linestyle='--', linewidth=2)

    # Label the x and y axes
    plt.xlabel('Date')
    plt.ylabel('COVID-19 cases normalized over state poppulation')

    # Add a legend
    plt.legend()

    # Add title
    #plt.title('COVID-19 cases normalized over state populatoin for ' + str(cluster) + ' states') # Add your title here


    # Add grid lines
    plt.grid(True, linestyle='--', alpha=0.7)

    # Show the plot
    plt.tight_layout()
    plt.show()

# Example usage:
# plot_state_level_data(df, 'fip', vertical_dates=['2024-01-01', '2024-03-01', '2024-05-01'])



def rolling_correlation(df1, df2, window, corr_method):
    if len(df1) != len(df2):
        raise ValueError("DataFrames must have the same length")

    correlations = []
    for i in range(len(df1) - window + 1):
        subset1 = df1[i:i + window]
        subset2 = df2[i:i + window]

        correlation = subset1.corr(subset2)
        correlations.append(correlation)

    return np.mean(correlations)


def load_features_data(start_date, end_date,states_rename_dict, mva=28, topics ='no_input_list', log_norm_tweets = False, log_norm_cases = False):

    # collect all input features in a df list
    dfs = []

    # load the features which you want to predict, could be cases or deaths
    df_predict = pd.read_csv("../../../data/covid19_cases/daily_new_cases_state.csv", index_col = 0)
    df_predict = df_predict[df_predict.index >= str(start_date)[:10]]
    df_predict = df_predict[df_predict.index < str(end_date)[:10]]
    if log_norm_cases:
        dfs.append(np.log(df_predict.rolling(window=mva).mean().dropna()+1))
    else:
        dfs.append(df_predict.rolling(window=mva).mean().dropna())
    
    df_all_tweets = pd.read_csv("../../../data/tweet_mats/tweets_all_tweets.csv", index_col = 0)
    df_all_tweets = df_all_tweets[df_all_tweets.index >= str(start_date)[:10]]
    df_all_tweets = df_all_tweets[df_all_tweets.index < str(end_date)[:10]]
    if log_norm_tweets:
        df_all_tweets = np.log(df_all_tweets+1)

    # Get a list of all file names in the folder
    folder_path = '../../../data/tweet_mats/'
    file_names = os.listdir(folder_path)

    if topics == 'no_input_list':
        topics = [
        'keywords_covid_baseline_topic',
        'keywords_virus_new',
        'keywords_symptoms_new',
        'keywords_testing_new',
        'keywords_vaccination_new',
        'keywords_preventive_measures_new',
        'keywords_quarantine_new',
        'keywords_travel_restrictions_new',
        'keywords_health_experts_new',
        'keywords_health_care_new',
        'anti_narrative_new',
        'alternative_treatments_new',
        'keywords_anti_vaccines_new',
        'keywords_bigpharma_new'
        ]

    for topic in topics:
        df_tweets = pd.read_csv("../../../data/tweet_mats/tweets_"+topic.replace('keywords_','').replace('_new','')+'.csv', index_col = 0)
        df_tweets = df_tweets[df_tweets.index >= str(start_date)[:10]]
        df_tweets = df_tweets[df_tweets.index < str(end_date)[:10]]
        if log_norm_tweets:
            df_tweets = np.log(df_tweets+1)
        dfs.append((df_tweets.fillna(0).rolling(window=mva).mean().dropna()/df_all_tweets.fillna(0).rolling(window=mva).mean().dropna()).fillna(0).rename(columns = states_rename_dict))
    return dfs

def pop_normalize(data_df, state_data, cluster_id = 'Clusters'   ):
    df = data_df.T#dfs[0].copy().T
    df['state_shor'] = df.index#.astype('float').astype('int64')
    df_merged = pd.merge( df, state_data, on ='state_shor', how='left')
    df_pop_norm = df_merged.iloc[:,:-len(state_data.columns)].T/df_merged['population'].astype('float')
    df_result = df_pop_norm.T
    df_result['Clusters'] = df_merged[cluster_id].astype('str')
    df_result.index = df_merged['state_shor']
    return df_result

def aggregate_cases_tweets_per_cluster(dfs, cluster_id, mva =14, keywords_in = None):

    df_id = 'cases'
    df_cases = dfs[0]
    dfs_tweets = dfs[1:]
    
    # calcualte the rolling corr for each keyword and each cluster
    df_aggr = pd.DataFrame()
    # Get a list of all file names in the folder
    folder_path = '../../../data/tweet_mats/'
    keywords = [el.replace('.csv','').replace('_new','').replace('tweets_','') for el in os.listdir(folder_path)]
    if keywords_in==None:
        keywords = [
        'COVID-19 BaseTopic',
        'Virus',
        'Symptoms',
        'Testing',
        'Vaccination',
        'Preventive Measures',
        'Quarantine',
        'Travel Restrictions',
        'Health Experts',
        'Health & Care',
        'Anti Narrative',
        'Alternative Treatments',
        'Anti Vaccines',
        'Big Pharma'
        ]
    else:
        keywords = keywords_in

    for cluster in set(df_cases[cluster_id]):

            # get all counties in one cluster
            df_cases_cluster = df_cases[df_cases[cluster_id]==cluster].drop([cluster_id], axis=1)

            # sum over all counties in one clusters, potenially apply another moving average
            cases_time_series = pd.DataFrame(np.mean(df_cases_cluster.T, axis=1)).rolling(window=mva).mean().fillna(0)

            # add to final correlation dataframe
            df_aggr[df_id+'_' +str(cluster)] = cases_time_series

            # now add tweets time series and calculate the correlations for each keyword time series
            for kw in range(0,len(keywords)):

                # add political cluster info to tweets data
                df_tweets = dfs_tweets[kw]
                df_tweets_T = df_tweets.T
                df_tweets_T['state_shor'] = df_tweets_T.index#.astype('int64')
                df_tweets_clusters = pd.merge(df_tweets_T, df_cases[cluster_id].reset_index(), on='state_shor', how='left')

                # filter for all counties of a specific cluster
                df_tweets_cluster = df_tweets_clusters[df_tweets_clusters[cluster_id]==cluster].drop([cluster_id,'state_shor'], axis=1)
                tweets_time_series = pd.DataFrame(np.mean(df_tweets_cluster.T, axis=1)).rolling(window=mva).mean().fillna(0)
                df_aggr[keywords[kw]+'_' +cluster] = tweets_time_series
            
    return df_aggr


def plot_state_level_data(df, fip):
    plt.figure(figsize=(10, 5))

    # Set the x-axis as dates
    plt.plot(df.index, df[fip], label='Value')

    # Set the number of date ticks to display (e.g., 10)
    num_ticks = 10

    # Calculate the step for date ticks
    step = len(df) // num_ticks
    date_ticks = df.index[::step]

    # Set the date ticks on the x-axis
    plt.xticks(date_ticks, rotation=45)  # Rotate date labels for readability

    # Label the x and y axes
    plt.xlabel('Date')
    plt.ylabel('Value')

    # Add a legend
    plt.legend()

    # Show the plot
    plt.show()

def plot_bar_chart_over_time(correlation_df, threshold=None, color_dict=None, cluster='Swing'):
    """
    Plots a stacked bar chart over time based on correlation data.

    Parameters:
    - correlation_df: DataFrame containing correlation data with columns ['political_cluster_2020', 'Time Frame', 'Highest Correlation Topic1', 'Correlation Value1'].
    - threshold: Threshold value for adjusting Highest Correlation Topic1.
    - color_dict: Dictionary containing custom colors for each unique correlation topic.
    - cluster: Political cluster category to plot (e.g., 'Swing', 'Democrat', 'Republican').

    Returns:
    - None
    """
    # Adjust Highest Correlation Topic1 based on the threshold
    if threshold is not None:
        correlation_df.loc[correlation_df['Correlation Value1'] < threshold, 'Highest Correlation Topic1'] = 'Correlation<0.5'

    # Group by Category and Highest Correlation Topic, and count occurrences
    result = correlation_df.groupby(['political_cluster_2020', 'Time Frame', 'Highest Correlation Topic1']).size().unstack(fill_value=0)

    # Extract data for the specified political cluster
    df = result.xs(cluster, level='political_cluster_2020')

    # Filter out columns with all zero values
    df = df.loc[:, (df != 0).any(axis=0)]

    # Order columns based on color_dict keys while preserving order of topics not in color_dict
    if color_dict is not None:
        topics = list(color_dict.keys())
        remaining_topics = [col for col in df.columns if col not in topics]
        ordered_topics = topics + remaining_topics
        df = df.reindex(columns=ordered_topics)

    # Filter color_dict to include only topics present in the dataframe
    if color_dict is not None:
        present_topics = df.columns
        color_dict = {topic: color for topic, color in color_dict.items() if topic in present_topics}

    # Plotting stacked bar chart over time
    time_frames = df.index
    num_time_frames = len(time_frames)
    num_topics = len(df.columns)
    bar_width = 0.8 / num_topics  # Adjust bar width based on number of topics

    plt.figure(figsize=(14, 6))
    for i, topic in enumerate(df.columns):
        bar_positions = np.arange(num_time_frames) + i * bar_width
        plt.bar(bar_positions, df[topic], width=bar_width, color=color_dict.get(topic, 'grey'), label=topic)

    # Customize the plot
    plt.title(f'Stacked Bar Chart over Time for {cluster} states', fontsize=16)
    plt.xlabel('Time Frame', fontsize=14)
    plt.ylabel('Values', fontsize=14)
    plt.xticks(np.arange(num_time_frames), time_frames)
    if color_dict is not None:
        plt.legend(fontsize=12, loc='center left', bbox_to_anchor=(1.05, 0.5))

    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_stacked_bar_chart(correlation_df, threshold=None, color_dict=None):
    """
    Plots a stacked bar chart based on correlation data.

    Parameters:
    - correlation_df: DataFrame containing correlation data with columns ['political_cluster_2020', 'Time Frame', 'Highest Correlation Topic1', 'Correlation Value1'].
    - threshold: Threshold value for adjusting Highest Correlation Topic1.
    - color_dict: Dictionary containing custom colors for each unique correlation topic.

    Returns:
    - None
    """
    # Adjust Highest Correlation Topic1 based on the threshold
    if threshold is not None:
        correlation_df.loc[correlation_df['Correlation Value1'] < threshold, 'Highest Correlation Topic1'] = 'Correlation<0.5'

    # Group by Category and Highest Correlation Topic, and count occurrences
    result = correlation_df.groupby(['political_cluster_2020', 'Time Frame', 'Highest Correlation Topic1']).size().unstack(fill_value=0)

    # Extract data for the 'Swing' category
    df = result.xs('Swing', level='political_cluster_2020')

    # Filter out columns with all zero values
    df = df.loc[:, (df != 0).any(axis=0)]

    # Order columns based on color_dict keys while preserving order of topics not in color_dict
    if color_dict is not None:
        topics = list(color_dict.keys())
        remaining_topics = [col for col in df.columns if col not in topics]
        ordered_topics = topics + remaining_topics
        df = df.reindex(columns=ordered_topics)

    # Filter color_dict to include only topics present in the dataframe
    if color_dict is not None:
        present_topics = df.columns
        color_dict = {topic: color for topic, color in color_dict.items() if topic in present_topics}

    # Plotting the stacked bar chart
    ax = df.plot(kind='bar', stacked=True, color=[color_dict[col] for col in df.columns], figsize=(10, 6))

    # Customize the plot
    ax.set_xlabel('Time Frame', fontsize=14)
    ax.set_ylabel('States', fontsize=14)
    ax.set_title('Geo-Social Media Topics with the highest Correlation to COVID-19 cases Swing states', fontsize=16)
    ax.legend(fontsize=12, loc='center left', bbox_to_anchor=(1.05, 0.5))

    plt.tight_layout()
    plt.show()