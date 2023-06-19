"""
Work in Progress. Ignore file for now.
"""

import numpy as np
from bokeh.plotting import figure, output_file, save
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.models import LinearAxis, Range1d


class BokehPlot(object):
    def __init__(self, sizing_mode="stretch_both", x_axis_type='datetime'):
        self._sizing_mode = sizing_mode
        self._x_axis_type = x_axis_type
        self.fig = figure(sizing_mode=self._sizing_mode, x_axis_type=self._x_axis_type)

    def line_plot(self, data_series):
        source = ColumnDataSource(data={
            'date': np.array(data_series.index, dtype=np.datetime64),
            'index': data_series.values,
        })

        self.fig.xaxis.axis_label = 'Date'
        self.fig.yaxis.axis_label = 'Index'

        # add a renderer
        self.fig.line(x='date', y='index', line_width=1, color='#ebbd5b', source=source)
        self.fig.add_tools(HoverTool(
            tooltips=[
                ('date', '@date{%F}'),
                ('index', '@index{0.2f}'),  # use @{ } for field names with spaces
            ],

            formatters={
                '@date': 'datetime',  # use 'datetime' formatter for '@date' field
                '@{index}': 'printf',  # use 'printf' formatter for '@{adj close}' field
                # use default 'numeral' formatter for other fields
            },

            # display a tooltip whenever the cursor is vertically in line with a glyph
            mode='vline'
        ))


    def dual_y_plot(self, df1, df2):
        # Create a new plot
        p = figure(sizing_mode="stretch_both", x_axis_type='datetime', y_range=(-0.2, 1.2))
        p.legend.click_policy = "hide"

        source = ColumnDataSource(data={
            'date': np.array(results_df.index, dtype=np.datetime64),
            'P_b': results_df['P_b'].values,
            'P_s': results_df['P_s'].values,
            'BenchmarkReturn': results_df['BenchmarkReturn'].values,
            'StrategyCumReturn': results_df['StrategyCumReturn'].values
        })

        # Primary y-axis
        p.line(x='date', y='P_b', color="blue", line_width=1, source=source, legend_label="P_b")
        p.line(x='date', y='P_s', color="magenta", line_width=1, source=source, legend_label="P_s")

        # secondary y-axis
        y_min = min(min(results_df['BenchmarkReturn'].values), min(results_df['StrategyCumReturn'].values)) - 0.2
        y_max = max(max(results_df['BenchmarkReturn'].values), max(results_df['StrategyCumReturn'].values)) + 0.2
        p.extra_y_ranges['secondary'] = Range1d(y_min, y_max)

        p.line(x='date', y='BenchmarkReturn', color="green", line_width=1, source=source,
               legend_label="Benchmark Return", y_range_name="secondary")
        strat = p.line(x='date', y='StrategyCumReturn', color="red", line_width=1, source=source,
                       legend_label="Strategy Cumulative Return", y_range_name="secondary")
        p.add_layout(LinearAxis(y_range_name="secondary"), 'right')

        # Add the hover tool with the specified tooltips and mode='vline'
        p.add_tools(HoverTool(
            tooltips=[
                ('date', '@date{%F}'),
                ('P_b', '@P_b{0.00 a}'),
                ('P_s', '@P_s{0.00 a}'),
                ('BenchmarkReturn', '@BenchmarkReturn{0.00 a}'),
                ('StrategyCumReturn', '@StrategyCumReturn{0.00 a}'),
            ],

            formatters={
                '@date': 'datetime',  # use 'datetime' formatter for '@date' field
                # use default 'numeral' formatter for other fields
            },

            # display a tooltip whenever the cursor is vertically in line with a glyph
            mode='vline',
            renderers=[strat]
        ))

        # Show the plot
        show(p)
