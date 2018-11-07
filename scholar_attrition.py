from sqlalchemy import create_engine
from pandas import read_sql, date_range, to_datetime, DataFrame, Timestamp

from bokeh.models.axes import LinearAxis
from bokeh.models.plots import Plot, GridPlot, ColumnDataSource, Range1d, GlyphRenderer
from bokeh.models.glyphs import Segment, Text, Line, Patch
from bokeh.models.tools import HoverTool, ResetTool, BoxZoomTool, PreviewSaveTool, PanTool, HelpTool, CrosshairTool
from bokeh.models.formatters import NumeralTickFormatter
from bokeh.models.widgets.inputs import Select
from bokeh.models.widgets.markups import PreText
from bokeh.models.widgets.layouts import VBox, HBox
from bokeh.properties import Instance, String
from bokeh.plotting import curdoc

import logging
logging.basicConfig(level=logging.ERROR)

storage = {}


class ScholarAttrition(VBox):
    extra_generated_classes = [["ScholarAttrition", "ScholarAttrition", "VBox"]]
    jsmodel = "VBox"

    # layout boxes
    input_box = Instance(HBox)

    #outputs
    header = Instance(PreText)
    paragraph = Instance(PreText)
    plot = Instance(GridPlot)

    # inputs
    school_type_select = Instance(Select)
    school_age_select = Instance(Select)
    school_select = Instance(Select)

    connection_string = String()
    school = String()
    school_type = String()
    school_age = String()

    def __init__(self, *args, **kwargs):
        super(ScholarAttrition, self).__init__(*args, **kwargs)

    @classmethod
    def create(cls, connection_string):
        obj = cls()
        obj.connection_string = connection_string
        obj.input_box = HBox()
        obj.plot = GridPlot(toolbar_location='right')

        obj.set_inputs()
        obj.update_plot()
        obj.set_children()

        return obj

    def set_inputs(self):
        nav = self.counts_data

        schools = sorted(nav['school_name'].unique())
        self.school = schools[0]
        nav = nav[nav['school_name'].eq(self.school)]

        academic_years = sorted(nav['academic_year'].unique())
        nav = nav[nav['academic_year'].eq(academic_years[-1])]

        self.school_type = 'All school types'
        self.school_age = 'All school ages'

        self.school_select = Select(name='school_name', title='School', value=self.school, options=schools)
        self.school_age_select = Select(
            name='school_age', title='School Age', value=self.school_age,
            options=['All school ages'] + sorted(nav['school_age'].astype('int').astype('str').unique()))
        self.school_type_select = Select(
            name='school_type', title='School Type', value=self.school_type,
            options=['All school types', 'ES', 'MS', 'HS'])

    def set_children(self):
        self.input_box.children = [
            self.school_type_select,
            self.school_age_select,
            self.school_select,
        ]
        self.children = [self.input_box, self.plot]

    def setup_events(self):
        super(ScholarAttrition, self).setup_events()
        if self.school_select:
            self.school_select.on_change('value', self, 'update_navigation')
        if self.school_type_select:
            self.school_type_select.on_change('value', self, 'update_navigation')
        if self.school_age_select:
            self.school_age_select.on_change('value', self, 'update_navigation')

    def update_navigation(self, obj, attr, old, new_value):
        nav = self.counts_data

        if obj == self.school_select:
            self.school = new_value
        elif obj == self.school_type_select:
            self.school_type = new_value
        elif obj == self.school_age_select:
            self.school_age = new_value

        if self.school_age != 'All school ages':
            nav = nav[nav['school_age'].eq(int(self.school_age))]
        if self.school_type != 'All school types':
            nav = nav[nav['school_type'].eq(self.school_type)]

        self.school_select.options = sorted(nav['school_name'].unique())
        if self.school not in self.school_select.options:
            self.school = self.school_select.options[0]

        self.update_plot()
        self.set_children()
        curdoc().add(self)

    def update_plot(self):

        today = Timestamp.today().strftime('%Y-%m-%d')
        today_date = to_datetime(today)

        df = self.counts_data
        weights = self.similarities_data.xs(self.school, level='school_name', axis=1)
        df_all = df[df['school_name'].eq(self.school)]
        present_academic_year = df['academic_year'].unique()[0]

        tooltips = '''
        <div style="font-size: 12px;">
        <span style="font-weight: bold;">@school_name</span> (@school_type, @school_age years old)<br>
        Grade @grade enrollment, as of @reference_date_string<br>
        <br>
        <span style="font-weight: bold;">Attrition</span>: @exit_count_present_cumm (@exit_percent_present_cumm{'0.0%'} change from October 1)<br>
        <span style="font-weight: bold;">Backfill</span>: @enter_count_present_cumm (@enter_percent_present_cumm{'0.0%'} change from October 1)<br>
        <span style="font-weight: bold;">Current total enrollment</span>: @total_enrollment
        </div>
        '''

        ticker_ref = DataFrame({'month': date_range('2014-10-01', '2015-06-01', freq='MS')})
        ticker_ref['marker'] = ticker_ref['month']
        ticker_ref['month'] = ticker_ref['month'].apply(lambda x: x.strftime('%B'))
        ticker_ref = ticker_ref.set_index('marker')['month'].to_dict()

        x_range1d = Range1d(-0.05, 290.0)
        y_index = max(
            -df_all['exit_percent_present_cumm'].min(),
            df_all['enter_percent_present_cumm'].max(),
            -df_all['exit_percent_past_cumm'].min(),
            df_all['enter_percent_past_cumm'].max()
        )
        y_range1d = Range1d(-y_index - 0.005, y_index + 0.005)
        plots = []
        focus_cols = ['enter_percent_present_cumm', 'exit_percent_present_cumm']
        unique_grades = sorted(df_all['grade'].unique().astype('int'))
        for grade in unique_grades:
            grade_weight = weights[grade]
            benchmark_df = DataFrame(columns=focus_cols)
            df_copy = df[df['academic_year'].eq(present_academic_year)].copy().set_index(['school_name', 'grade'])

            for rid, label in sorted(ticker_ref.items(), reverse=True):
                df_copy = df_copy[df_copy['reference_date'].le(rid)]
                df_ave = df_copy.groupby(level=['school_name', 'grade'])[focus_cols].tail(1).\
                    mul(grade_weight, axis=0).sum() / grade_weight.sum()
                df_ave.name = label
                benchmark_df = benchmark_df.append(df_ave)
            benchmark_df = benchmark_df.reset_index()
            benchmark_df['reference_date'] = benchmark_df['index'].map({y: x for x, y in ticker_ref.items()})
            benchmark_df['reference_id'] = benchmark_df['reference_date'].apply(lambda x: x.toordinal()) - \
                df_all['reference_date'].min().toordinal()

            source_df = df_all[df_all['grade'].eq(grade)]
            source_df_rev = source_df.sort_values('reference_id', ascending=False)
            source_df_trunc = source_df.loc[source_df['reference_date'].le(today_date), :]
            source = ColumnDataSource(source_df)
            source_trunc = ColumnDataSource(source_df_trunc)
            patch_source = ColumnDataSource(dict(
                x_past=source_df['reference_id'].tolist() + source_df_rev['reference_id'].tolist(),
                y_past=source_df['enter_percent_past_cumm'].tolist() + source_df_rev['exit_percent_past_cumm'].tolist()
            ))

            plot1 = Plot(
                x_range=x_range1d,
                y_range=y_range1d,
                min_border_bottom=5,
                min_border_top=10,
                min_border_right=10,
                plot_width=700,
                plot_height=150,
                title=None,
                title_text_font_size='0pt',
                title_text_color='grey',
                outline_line_alpha=0.0)

            plot1.add_layout(
                LinearAxis(
                    axis_label='Grade ' + str(grade), axis_label_text_font_size='9pt', minor_tick_line_alpha=0.0,
                    axis_label_text_color='grey',
                    axis_line_alpha=0.1, major_tick_line_alpha=0.1, major_label_text_color='grey',
                    major_label_text_font_size='7pt', formatter=NumeralTickFormatter(format='0%')
                ), 'left')

            patch = Patch(x='x_past', y='y_past', fill_color='#AFAFAD', fill_alpha=0.25, line_alpha=0.0)
            plot1.add_glyph(patch_source, patch)

            line1 = Line(
                x='reference_id', y='enter_percent_present_cumm', line_width=2, line_color='#f7910b', line_alpha=1.0)
            plot1.add_glyph(source_trunc, line1)

            line2 = Line(
                x='reference_id', y='exit_percent_present_cumm', line_width=2, line_color='#f7910b', line_alpha=1.0)
            plot1.add_glyph(source_trunc, line2)

            line_h = Line(x='reference_id', y=0, line_width=1, line_color='black', line_alpha=0.1)
            line_renderer = GlyphRenderer(data_source=source, glyph=line_h, name='line')
            plot1.add_glyph(source, line_h)

            for ind, series in benchmark_df.iterrows():
                x = series['reference_id']
                y_enter = series['enter_percent_present_cumm']
                y_exit = series['exit_percent_present_cumm']
                label = series['index']

                line = Segment(x0=x, x1=x, y0=-y_index, y1=y_index, line_width=1, line_color='#165788', line_alpha=0.1)
                plot1.add_glyph(line)

                linec1 = Segment(
                    x0=x - 3, x1=x + 3, y0=y_enter, y1=y_enter, line_width=1, line_color='#ed2939', line_alpha=1.0)
                plot1.add_glyph(linec1)

                linec2 = Segment(
                    x0=x - 3, x1=x + 3, y0=y_exit, y1=y_exit, line_width=1, line_color='#ed2939', line_alpha=1.0)
                plot1.add_glyph(linec2)

                text = Text(x=x+3, y=-y_index, text=[label], text_font_size='8pt', text_color='grey', text_alpha=0.5)
                plot1.add_glyph(text)

            hover_tool = HoverTool(
                plot=plot1, renderers=[line_renderer], tooltips=tooltips, always_active=False, mode='vline',
                point_policy='follow_mouse', line_policy='prev')
            crosshair_tool = CrosshairTool(plot=plot1, dimensions=['height'])
            zoom_tool = BoxZoomTool(plot=plot1, dimensions=['width'])
            reset_tool = ResetTool(plot=plot1)
            save_tool = PreviewSaveTool(plot=plot1)
            pan_tool = PanTool(plot=plot1, dimensions=['width'])
            help_tool = HelpTool(plot=plot1, help_tooltip='App help page', redirect='http://data.successacademies.org/blog/')
            plot1.tools.extend([hover_tool, zoom_tool, pan_tool, reset_tool, save_tool, help_tool, crosshair_tool])
            plot1.renderers.extend([line_renderer])
            plots.append([plot1])

        self.plot.children = plots

    @property
    def counts_data(self):

        ref_date = Timestamp.today().strftime('%Y-%m-%d')

        if ref_date in storage.keys():
            return storage[ref_date]['counts'].copy()

        conn = create_engine(self.connection_string)
        cdata = read_sql('SELECT * FROM app__scholar_attrition__counts', conn)
        sdata = read_sql('SELECT * FROM app__scholar_attrition__similarities', conn)
        sdata = sdata.set_index(['school_name_row', 'grade_row', 'school_name_col', 'grade_col'])['similarity'].\
            unstack(['school_name_col', 'grade_col'])
        sdata.index.names = [r.replace('_row', '') for r in sdata.index.names]
        sdata.columns.names = [c.replace('_col', '') for c in sdata.columns.names]
        conn.dispose()

        for k in storage.keys():
            _ = storage.pop(k)
        storage[ref_date] = {}
        storage[ref_date]['counts'] = cdata.copy()
        storage[ref_date]['similarities'] = sdata.copy()

        return cdata

    @property
    def similarities_data(self):

        ref_date = Timestamp.today().strftime('%Y-%m-%d')

        if ref_date in storage.keys():
            return storage[ref_date]['similarities'].copy()

        conn = create_engine(self.connection_string)
        cdata = read_sql('SELECT * FROM app__scholar_attrition__counts', conn)
        sdata = read_sql('SELECT * FROM app__scholar_attrition__similarities', conn)
        sdata = sdata.set_index(['school_name_row', 'grade_row', 'school_name_col', 'grade_col'])['similarity'].\
            unstack(['school_name_col', 'grade_col'])
        sdata.index.names = [r.replace('_row', '') for r in sdata.index.names]
        sdata.columns.names = [c.replace('_col', '') for c in sdata.columns.names]
        conn.dispose()

        for k in storage.keys():
            _ = storage.pop(k)
        storage[ref_date] = {}
        storage[ref_date]['counts'] = cdata.copy()
        storage[ref_date]['similarities'] = sdata.copy()

        return sdata



#from bokeh.plotting import figure, show, output_file
#output_file("/Users/swheeler/Desktoptest.html", title="test example")
#show(plot1)