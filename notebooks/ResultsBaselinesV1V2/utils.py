'''Functions to plot data using matplotlib or plotly.
'''
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


DAG_EDGES_MAP = {'cancer': 4,
                 'earthquake': 4,
                 'survey': 6,
                 'asia': 8,
                 'sachs': 17,
                 'child': 25,
                 'insurance': 52}

DAG_NODES_MAP = {'cancer': 5,
                 'earthquake': 5,
                 'survey': 6,
                 'asia': 8,
                 'sachs': 11,
                 'child': 20,
                 'insurance': 27}


def process_data(df, groupby_cols = ['dataset', 'n_nodes', 'n_edges', 'model']):
    df_grouped = df.groupby(groupby_cols, as_index=False).aggregate(
        sid_low_mean=('sid_low', 'mean'),
        sid_high_mean=('sid_high', 'mean'),
        sid_low_std=('sid_low', 'std'),
        sid_high_std=('sid_high', 'std'),
    )
    df_grouped['p_SID_low_mean'] = df_grouped['sid_low_mean'] / df_grouped['n_edges']
    df_grouped['p_SID_high_mean'] = df_grouped['sid_high_mean'] / df_grouped['n_edges']
    df_grouped['p_SID_low_std'] = df_grouped['sid_low_std'] / df_grouped['n_edges']
    df_grouped['p_SID_high_std'] = df_grouped['sid_high_std'] / df_grouped['n_edges']
    return df_grouped


def plot_runtime(df, 
                 names_dict, 
                 colors_dict, 
                 symbols_dict,
                 methods, 
                 plot_width=750, plot_height=300, font_size=20, save_figs=False, output_name="random_graphs_runtime.html"):

    fig = make_subplots(rows=1, cols=1, shared_yaxes=True)
   
    for method in methods:
        method_df = df[df['model'] == method]
        fig.add_trace(
            go.Scatter(
                x=method_df['n_nodes'].astype(str),
                y=method_df['elapsed_mean'],
                error_y=dict(type='data', array=method_df['elapsed_std'], thickness=2),
                mode='lines+markers',
                name=names_dict[method],
                line=dict(color=colors_dict[method], width=2),
                marker=dict(symbol=symbols_dict[method], size=8, color=colors_dict[method]),
                opacity=0.8,
            )
        )

    # Log scale for y-axis
    fig.update_yaxes(type="log", title='log(elapsed time [s])')

    # X axis title
    fig.update_xaxes(title='Number of Nodes (|V|)')

    # Layout and style
    fig.update_layout(
        legend=dict(orientation="h", xanchor="center", x=0.5, yanchor="bottom", y=1.05),
        template='plotly_white',
        width=plot_width,
        height=plot_height,
        margin=dict(l=10, r=10, b=80, t=10),
        font=dict(size=font_size, family="Serif", color="black")
    )

    if save_figs:
        fig.write_html(output_name)
        fig.write_image(output_name.replace('.html', '.jpeg'), scale=2)

    fig.show()



def double_bar_chart_plotly(all_sum, names_dict, colors_dict, 
                            methods=['Random', 'FGS', 'NOTEARS-MLP', 'Shapley-PC', 'ABAPC (Existing)', 'ABAPC (ASPforABA)'],
                            range_y1=None, range_y2=None, font_size=20,
                            save_figs=False, output_name="bar_chart.html", debug=False,
                            dist_between_lines=0.1565,
                            intra_dis=0.112,
                            inter_dis=0.137,
                            lin_space=9,
                            nl_space=9,
                            start_pos = 0.039,
                            width=1600,
                            height=700):
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    vars_to_plot = ['p_SID_low','p_SID_high']
    for n, var_to_plot in enumerate(vars_to_plot):
        for m, method in enumerate(methods):
            trace_name = 'True Graph Size' if var_to_plot=='nnz' and method=='Random' else method#+' '+var_to_plot
            fig.add_trace(go.Bar(x=all_sum[(all_sum.model==method)]['dataset'], 
                                yaxis=f"y{n+1}",
                                offsetgroup=m+len(methods)*n+(1*n),
                                y=all_sum[(all_sum.model==method)][var_to_plot+'_mean'], 
                                error_y=dict(type='data', array=all_sum[(all_sum.model==method)][var_to_plot+'_std'], visible=True),
                                name=names_dict[trace_name],
                                marker_color=colors_dict[method],
                                opacity=0.6,
                                #  width=0.1
                                showlegend=n==0
                                ))
        if n==0:
            fig.add_trace(go.Bar(x=all_sum[(all_sum.model==method)]['dataset'], 
                                    y=np.zeros(len(all_sum[(all_sum.model==method)]['dataset'])), 
                                    name='',
                                    offsetgroup=m+1,
                                    marker_color='white',
                                    opacity=1,
                                    # width=0.1
                                    showlegend=False
                                    )
                                    )
    second_ticks = False if all('SID' in var for var in vars_to_plot) else True
    # Change the bar mode
    fig.update_layout(barmode='group',
                        bargap=0.15, # gap between bars of adjacent location coordinates.
                        bargroupgap=0.15, # gap between bars of the same location coordinate.)

            legend=dict(orientation="h", xanchor="center", x=0.5, yanchor="top", y=1.1),
            template='plotly_white',
            # autosize=True,
            width=width, 
            height=height,
            margin=dict(
                l=40,
                r=00,
                b=70,
                t=20,
                # pad=10
            ),hovermode='x unified',
            font=dict(size=font_size, family="Serif", color="black"),
            yaxis2=dict(scaleanchor=0, showline=False, showgrid=False, showticklabels=second_ticks, zeroline=True),
            )

    fig.add_annotation(
        xref="paper",
        yref="paper",
        xanchor="center",
        x=0,
        yanchor="bottom",
        y=-0.08,
        text=f"Dataset:",
        showarrow=False,    
        font=dict(
                    family="Serif",
                    size=font_size,
                    color="Black"
                    )
        )
    
    for n, var_to_plot in enumerate(vars_to_plot):
        if vars_to_plot == ['precision', 'recall']:
            if range_y1 is None:
                range_y = [0, 1.3]
            else:
                range_y = range_y1
        elif vars_to_plot == ['fdr', 'tpr']:
            range_y = [0, 1]
        elif 'shd' in var_to_plot or 'SID' in var_to_plot:
            if range_y1 is None and range_y2 is None:
                if 'high' in var_to_plot:
                    range_y = [0, max(all_sum['p_SID_high_mean'])+.3]
                elif 'low' in var_to_plot:
                    range_y = [0, max(all_sum['p_SID_low_mean'])+.3]
                else:
                    range_y = [0, 2] if n==0 else [0, max(all_sum['p_SID_mean'])+.3]
            else:
                range_y = range_y1 if n==0 else range_y2
        if 'n_' in var_to_plot or 'p_' in var_to_plot:
            orig_y = var_to_plot.replace('n_','').replace('p_','').replace('_low','').replace('_high','').upper()
            fig.update_yaxes(title={'text':f'Normalised {orig_y}','font':{'size':font_size}}, secondary_y=n==1, range=range_y)
            if second_ticks == False:
                fig.update_yaxes(title={'text':'','font':{'size':font_size}}, secondary_y=True, range=range_y, showticklabels=False)
        elif var_to_plot=='nnz':
            orig_y = 'Number of Edges in DAG'
            fig.update_yaxes(title={'text':f'{orig_y}','font':{'size':font_size}}, secondary_y=n==1, range=range_y)
        else:
            fig.update_yaxes(title={'text':f'{var_to_plot.title()}','font':{'size':font_size}}, secondary_y=n==1, range=range_y)

    

    name1 = 'Best'
    name2 = 'Worst'
    

    n_x_cat = len(all_sum.dataset.unique())
    list_of_pos = []
    left=start_pos
    for i in range(n_x_cat):
        right = left+intra_dis
        list_of_pos.append((left, right))
        left = right+inter_dis

    for s1,s2 in list_of_pos:
        fig.add_annotation(
            xref="x domain",
            yref="y domain",
            xanchor="left",
            x=s1,
            y=1.015,
                    text=f"{' '*lin_space}{name1}{' '*(lin_space)}",
            showarrow=False,    
            font=dict(
                # family="Courier New, monospace",
                size=font_size,
                color="black"
                )
        , bordercolor='#E5ECF6'
        , borderwidth=2
        , bgcolor="#E5ECF6"
        , opacity=0.8
                )
        fig.add_annotation(
            xref="x domain",
            yref="y domain",
            xanchor="left",
            x=s2,
            y=1.015,
                    text=f"{' '*(nl_space)}{name2}{' '*nl_space}",
            showarrow=False,    
            font=dict(
                # family="Courier New, monospace",
                size=font_size,
                color="black"
                )
        , bordercolor='#E5ECF6'
        , borderwidth=2
        , bgcolor="#E5ECF6"
        , opacity=0.8
                )
    
    # Add vertical lines between bar groups
    s1 = 0
    sign = -1
    for i in range(n_x_cat*2):  # skip last one
        sign *= -1
        s1 += dist_between_lines
        fig.add_shape(
            type="line",
            x0=s1, x1=s1,
            y0=0, y1=1,
            xref="paper",
            yref="paper",
            line=dict(
                color="grey" if sign>0 else "black",
                width=1,
                dash="dash" if sign>0 else "solid",
            ),
            layer="below"
        )
        


    if save_figs:
        fig.write_html(output_name)
        fig.write_image(output_name.replace('.html','.jpeg'))

    fig.show()
