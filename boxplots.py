from sqlalchemy import create_engine
from pandas import read_sql, DataFrame, Series, to_datetime
from re import split
from time import time as get_time
from utilities.convenience import check_isin
from utilities.calculation import calculate_boxplot, compare_distributions
from utilities.visualization import jitter_offset
from utilities.callbacks import save_datasource_code

from bokeh.models.axes import LinearAxis, CategoricalAxis
from bokeh.models.plots import Plot, GridPlot, ColumnDataSource, Range1d, GlyphRenderer, FactorRange
from bokeh.models.glyphs import Circle, Segment, Rect, Quad
from bokeh.models.tools import HoverTool, ResetTool, PreviewSaveTool, PanTool, BoxZoomTool, TapTool
from bokeh.models.widgets.inputs import Select
from bokeh.models.widgets.groups import CheckboxGroup
from bokeh.models.widgets.buttons import Button
from bokeh.models.widgets.markups import PreText, Paragraph
from bokeh.models.widgets.layouts import VBox, HBox
from bokeh.models.widgets.panels import Tabs, Panel
from bokeh.models.widgets.tables import DataTable, TableColumn
from bokeh.properties import Instance, List, Int, String, Either
from bokeh.plotting import curdoc
from bokeh.models.callbacks import CustomJS

import logging
logging.basicConfig(level=logging.ERROR)

storage = dict()


class Boxplots(VBox):
    extra_generated_classes = [["BoxPlots", "Boxplots", "VBox"]]
    jsmodel = "VBox"

    # layout boxes
    nav_box = Instance(HBox)                  # contains school and grade selectors, and selection/filter notification
    panel_box = Instance(Tabs)                # contains navigation, school health, and boxplot panels
    panel1 = Instance(Panel)                  # contains all navigation controls (except for school)
    panel2 = Instance(Panel)                  # contains school overview
    panel3 = Instance(Panel)                  # contains grade overview
    panel4 = Instance(Panel)                  # contains boxplots
    panel5 = Instance(Panel)                  # contains reports
    panel6 = Instance(Panel)                  # contains red flags
    panel1_box = Instance(HBox)            # contains subject, stateexam, intervention, and otherfilter
    panel3_box = Instance(HBox)           # contains overview plot and indicator filters
    panel6_box = Instance(HBox)           # contains red flags and scholar selector
    stateexam_box = Instance(VBox)            # contains navigation checkboxes for state exam fails
    intervention_box = Instance(VBox)         # contains sped and rti
    otherfilter_box = Instance(VBox)          # contains skips, holdovers, ELL, teacher, and scholar

    # outputs
    plot2 = Either(Instance(GridPlot), Instance(Plot))  # shows school overview plots
    plot3a = Either(Instance(GridPlot), Instance(Plot))  # shows grade overview plots
    plot3b = Either(Instance(GridPlot), Instance(Plot))  # shows weights
    plot4 = Either(Instance(GridPlot), Instance(Plot))   # shows boxplots
    datatable6 = Instance(DataTable)                     # shows red flags
    report_field = Instance(PreText)                     # shows narrative/table summaries relevant to boxplots
    notification_field = Instance(PreText)               # shows school, subject(s), grade(s) and whether filters apply

    # inputs
    download_button = Instance(Button)
    indicator_checkboxgroup = Instance(CheckboxGroup)
    preset_select = Instance(Select)
    school_select = Instance(Select)
    grade_select = Instance(Select)
    teacher_select = Instance(Select)
    scholar_select = Instance(Select)
    skip_checkboxgroup = Instance(CheckboxGroup)
    holdover_checkboxgroup = Instance(CheckboxGroup)
    ell_checkboxgroup = Instance(CheckboxGroup)
    state_exam_literacy_checkboxgroup = Instance(CheckboxGroup)
    state_exam_mathematics_checkboxgroup = Instance(CheckboxGroup)
    state_exam_science_checkboxgroup = Instance(CheckboxGroup)
    sped_checkboxgroup = Instance(CheckboxGroup)
    rti_checkboxgroup = Instance(CheckboxGroup)
    datatable_checkboxgroup = Instance(CheckboxGroup)

    connection_string = String()
    reference_date = String()
    verbose = Int()
    current_scholar_ids = List(Int)

    def __init__(self, *args, **kwargs):
        super(Boxplots, self).__init__(*args, **kwargs)

    @classmethod
    def create(cls, connection_string, reference_date=None, verbose=0):
        obj = cls()

        conn = create_engine(connection_string)

        obj.connection_string = connection_string
        obj.verbose = verbose

        if reference_date is None:
            date_query = 'SELECT MAX(reference_date) FROM scholars;'
            obj.reference_date = read_sql(date_query, conn).iloc[0, 0].strftime('%Y-%m-%d')
        else:
            obj.reference_date = to_datetime(reference_date).strftime('%Y-%m-%d')

        obj.nav_box = HBox()
        obj.panel_box = Tabs(active=1)
        obj.panel1_box = HBox()
        obj.panel3_box = HBox()
        obj.panel6_box = HBox()
        obj.stateexam_box = VBox()
        obj.intervention_box = VBox()
        obj.otherfilter_box = VBox()
        obj.plot2 = GridPlot(children=[[None]], toolbar_location=None, border_space=0, min_border_top=0)
        obj.plot3a = GridPlot(children=[[None]], toolbar_location=None, border_space=0, min_border_top=0)
        obj.plot3b = GridPlot(children=[[None]], toolbar_location=None, border_space=0, min_border_top=0)
        obj.plot4 = GridPlot(children=[[None]])
        obj.report_field = PreText()
        obj.notification_field = PreText()

        obj.datatable_checkboxgroup = CheckboxGroup(name='filter0_datatable')
        obj.indicator_checkboxgroup = CheckboxGroup(name='filter0_indicator')
        obj.preset_select = Select(name='filter0_preset', title='Preset')
        obj.school_select = Select(name='filter1_school_name', title='School')
        obj.grade_select = Select(name='filter1_grade', title='Grade')
        obj.teacher_select = Select(name='filter2_teacher', title='Teacher lookup')
        obj.scholar_select = Select(name='filter2_scholar', title='Scholar lookup')
        obj.skip_checkboxgroup = CheckboxGroup(name='filter2_skip_nav')
        obj.holdover_checkboxgroup = CheckboxGroup(name='filter2_holdover_nav')
        obj.ell_checkboxgroup = CheckboxGroup(name='filter2_ell_status')
        obj.sped_checkboxgroup = CheckboxGroup(name='filter2_sped_nav')
        obj.rti_checkboxgroup = CheckboxGroup(name='filter2_rti_nav')
        obj.state_exam_literacy_checkboxgroup = CheckboxGroup(name='filter2_state_exam_literacy_fails')
        obj.state_exam_mathematics_checkboxgroup = CheckboxGroup(name='filter2_state_exam_mathematics_fails')
        obj.state_exam_science_checkboxgroup = CheckboxGroup(name='filter2_state_exam_science_fails')
        obj.datatable6 = DataTable(row_headers=False, selectable=False, sortable=False, width=800, height=500)
        obj.datatable6.columns = [
            TableColumn(field='Priority', title='Priority', width=110),
            TableColumn(field='Scholar Name', title='Scholar Name', width=200),
            TableColumn(field='Academic Performance', title='ACADEMICS', width=90),
            TableColumn(field='Literacy', title='Literacy', width=60),
            TableColumn(field='Mathematics', title='Math', width=60),
            TableColumn(field='Science', title='Science', width=60),
            TableColumn(field='History', title='History', width=60),
            TableColumn(field='Readiness Investment', title='READINESS', width=90),
            TableColumn(field='Learning Investment', title='LEARNING', width=90)
        ]

        obj.download_button = Button(label='Download (selected indicators only)')

        obj.nav_box.children = [
            obj.school_select,
            obj.grade_select,
            obj.preset_select,
            obj.download_button
        ]

        obj.stateexam_box.children = [
            Paragraph(text='State Exam Fails: Literacy'),
            obj.state_exam_literacy_checkboxgroup,
            Paragraph(text='State Exam Fails: Mathematics'),
            obj.state_exam_mathematics_checkboxgroup,
            Paragraph(text='State Exam Fails: Science'),
            obj.state_exam_science_checkboxgroup
        ]
        obj.intervention_box.children = [
            Paragraph(text='SpED Services'),
            obj.sped_checkboxgroup,
            Paragraph(text='Interventions'),
            obj.rti_checkboxgroup
        ]
        obj.otherfilter_box.children = [
            Paragraph(text='Skips'),
            obj.skip_checkboxgroup,
            Paragraph(text='Holdovers'),
            obj.holdover_checkboxgroup,
            Paragraph(text='ELL Status'),
            obj.ell_checkboxgroup,
            obj.teacher_select,
            obj.scholar_select
        ]

        obj.panel1_box.children = [obj.stateexam_box, obj.intervention_box, obj.otherfilter_box]
        obj.panel3_box.children = [obj.plot3a, obj.indicator_checkboxgroup, obj.plot3b]
        obj.panel6_box.children = [obj.datatable_checkboxgroup, obj.datatable6]
        obj.panel1 = Panel(child=obj.panel1_box, title='Filters', closable=False)
        obj.panel2 = Panel(child=obj.plot2, title='School Overview', closable=False)
        obj.panel3 = Panel(child=obj.panel3_box, title='Grade Overview', closable=False)
        obj.panel4 = Panel(child=obj.plot4, title='Scholar Details', closable=False)
        obj.panel5 = Panel(child=obj.report_field, title='Report', closable=False)
        obj.panel6 = Panel(child=obj.panel6_box, title='Red Flags', closable=False)
        obj.panel_box.tabs = [obj.panel1, obj.panel2, obj.panel3, obj.panel4, obj.panel5, obj.panel6]
        obj.children = [obj.nav_box, obj.notification_field, obj.panel_box]
        _ = obj.get_data('scholars')
        _ = obj.get_data('achievement')
        _ = obj.get_data('history')
        _ = obj.get_data('weights')

        obj.cascade_inputs(hard_reset=True)
        obj.update_plot2()
        obj.update_plot3()
        obj.update_plot4()
        obj.update_notifications()
        obj.update_paragraph()
        obj.update_red_flags()
        curdoc().add(obj)

        return obj

    def setup_events(self):
        super(Boxplots, self).setup_events()

        for wname, widget in self.get_widgets('tier1').items() + self.get_widgets('tier2').items():
            if hasattr(widget, 'active'):
                if widget:
                    widget.on_change('active', self, 'update_navigation')
            elif hasattr(widget, 'value'):
                if widget:
                    widget.on_change('value', self, 'update_navigation')

        for wname, widget in self.get_widgets('custom').items():
            if wname == 'indicator':
                if widget:
                    widget.on_change('active', self, 'update_navigation')
            if wname == 'datatable':
                if widget:
                    widget.on_change('active', self, 'update_records')
            if wname == 'preset':
                if widget:
                    widget.on_change('value', self, 'update_presets')

        for p in self.properties_with_values().values():
            if type(p) in (Plot, GridPlot):
                for cds in p.select(ColumnDataSource):
                    if cds.column_names:
                        cds.on_change('selected', self, 'update_records')

    def get_widgets(self, widget_type):
        """
        returns widget and current active value(s)

        """

        type_dict = {'custom': '0', 'tier1': '1', 'tier2': '2'}

        widgets = dict()
        for p in self.properties():
            prop = getattr(self, p)
            if hasattr(prop, '__module__'):
                if prop.name is not None:
                    if prop.name.startswith('filter'):
                        category_key = prop.name.replace('filter', '')[0]
                        if type_dict[widget_type] == category_key:
                            widget_key = prop.name.replace('filter', '')[2:]
                            widgets[widget_key] = prop

        return widgets

    @staticmethod
    def get_widget_values(widget, active=True):
        if hasattr(widget, 'active'):
            if active:
                return [l for l in widget.labels if widget.labels.index(l) in widget.active]
            else:
                return [l for l in widget.labels if widget.labels.index(l) not in widget.active]
        elif hasattr(widget, 'value'):
            if active:
                return [widget.value] if widget.value is not None else []
            else:
                return [l for l in widget.options if widget.value not in widget.options]

    def _get_options_tier1(self):

        output_dict = dict()
        scholars = self.get_data('scholars')
        wnames = ['school_name', 'grade']
        for wname in wnames:
            widget = self.get_widgets('tier1')[wname]
            all_w = sorted(scholars[wname].dropna().unique())
            current_w = widget.value if widget.value in all_w else all_w[0]
            scholars = scholars[scholars[wname].eq(current_w)]
            output_dict[wname] = all_w[:]

        return output_dict

    def _get_options_tier2(self):

        output_dict = dict()
        scholars = self.get_data('scholars')
        for wname, widget in self.get_widgets('tier1').items():
            values = self.get_widget_values(widget)
            scholars = scholars[scholars[wname].isin(values)]

        for wname, widget in self.get_widgets('tier2').items():
            try:
                all_w = sorted(scholars[wname].unique())
            except TypeError:
                all_w = sorted(set([x for y in scholars[wname].tolist() for x in y]))

            if wname == 'teacher':
                if self.scholar_select.value not in (None, 'All scholars'):
                    selected_teachers = scholars.loc[scholars['scholar'].eq(self.teacher_select.value), wname].tolist()
                    all_w = sorted(set([x for y in selected_teachers for x in y]))
                all_w = ['All {}s'.format(wname)] + all_w
            elif wname == 'scholar':
                if self.teacher_select.value not in (None, 'All teachers'):
                    all_w = sorted(scholars.loc[scholars['teacher'].eq(self.scholar_select.value), wname].unique())
                all_w = ['All {}s'.format(wname)] + all_w
            elif wname == 'ell_status':
                all_ell_statuses = ['Not an ELL', 'Graduated', 'Not graduated', 'Not evaluated']
                all_w = [l for l in all_ell_statuses if l in all_w]
            elif wname in ('sped_nav', 'rti_nav'):
                first_items = ['ICT', '12-1-1']
                last_items = ['No services', 'No interventions', 'Other']
                all_w = [i for i in first_items if i in all_w] + \
                    [i for i in all_w if i not in first_items + last_items] + \
                    [i for i in last_items if i in all_w]
            output_dict[wname] = all_w[:]

        return output_dict

    def _get_options_custom(self):

        output_dict = dict()
        scholars = self.get_data('scholars')
        for wname, widget in self.get_widgets('tier1').items():
            values = self.get_widget_values(widget)
            scholars = scholars[scholars[wname].isin(values)]

        for wname, widget in self.get_widgets('custom').items():
            if wname == 'indicator':
                achievement = self.get_data('achievement')
                achievement = achievement[achievement['scholar_id'].isin(scholars.index)]
                index_cols = ['indicator', 'scholar_id']
                w_checks = achievement.set_index(index_cols)['achievement'].unstack('indicator').isnull().mean()
                w_checks = w_checks[w_checks.lt(0.9)]

                all_w = [
                    'ACADEMIC PERFORMANCE', 'Literacy', 'Mathematics', 'Science', 'History', 'READINESS INVESTMENT',
                    'Absence', 'Tardy', 'Call Ahead', 'Early Dismissal', 'Late Pickup', 'Uniform',
                    'LEARNING INVESTMENT', 'Reading Log', 'Homework', 'NHM', 'Spelling'
                ]
                all_w = [s for s in all_w if s.lower() in [i.lower() for i in w_checks.index]]
            elif wname == 'preset':
                all_w = ['All Aggregate Indicators', 'Academic Performance', 'Readiness Investment', 'Learning Investment']
            elif wname in ('datatable', ):
                continue
            else:
                raise NotImplementedError

            output_dict[wname] = all_w[:]

        return output_dict

    def get_options(self):

        output_dict = dict()

        for wname, all_w in self._get_options_tier1().items() + self._get_options_tier2().items() + \
                self._get_options_custom().items():

            output_dict[wname] = all_w[:]

        return output_dict

    def filter_by_widget(self, exclude=None):
        exclude = [] if exclude is None else exclude
        scholars = self.get_data('scholars')
        if self.verbose > 2:
            print 'start', scholars.shape[0]
        for col, widget in self.get_widgets('tier1').items() + self.get_widgets('tier2').items():
            if col in exclude:
                continue
            if col in ['scholar', 'teacher']:
                if widget.value in ['All teachers', 'All scholars']:
                    continue
            actives = self.get_widget_values(widget)
            scholars = scholars[check_isin(scholars[col], actives)]
            if self.verbose > 2:
                print col, scholars.shape[0]

        return scholars

    def cascade_inputs(self, hard_reset):

        options_dict = self.get_options()

        for wname, widget in self.get_widgets('tier1').items():
            replace_actives = widget.value if widget.value in options_dict[wname] else options_dict[wname][0]
            widget.options = options_dict[wname]
            if widget.value != replace_actives:
                widget.value = replace_actives

        if hard_reset:
            options_dict = self.get_options()

        for wname, widget in self.get_widgets('tier2').items():
            if type(widget) == CheckboxGroup:
                if hard_reset:
                    widget.labels = options_dict[wname]
                    replace_actives = range(len(options_dict[wname]))
                else:
                    replace_actives = [i for i in widget.active if widget.labels[i] in options_dict[wname]]
                if widget.active != replace_actives:
                    if self.verbose > 1:
                        print widget.name
                    widget._callbacks = dict()
                    widget.active = replace_actives
                    widget._callbacks = {'active': [{'callbackname': 'update_navigation', 'obj': self}]}
            elif type(widget) == Select:
                if hard_reset:
                    widget.options = options_dict[wname]
                    replace_actives = options_dict[wname][0]
                else:
                    replace_actives = widget.value if widget.value in options_dict[wname] else options_dict[wname][0]
                if widget.value != replace_actives:
                    if self.verbose > 1:
                        print widget.name
                    widget._callbacks = dict()
                    widget.value = replace_actives
                    widget._callbacks = {'value': [{'callbackname': 'update_navigation', 'obj': self}]}

        if hard_reset:
            for wname, widget in self.get_widgets('custom').items():
                if wname == 'indicator':
                    if len(self.indicator_checkboxgroup.active) > 0:
                        actives = self.get_widget_values(widget)
                        selected_indicators = [x for x in actives if x in options_dict[wname]]
                    else:
                        selected_indicators = ['ACADEMIC PERFORMANCE', 'READINESS INVESTMENT', 'LEARNING INVESTMENT']
                    widget.labels = options_dict[wname]
                    replace_actives = [options_dict[wname].index(s) for s in selected_indicators]
                    if widget.active != replace_actives:
                        widget.active = replace_actives
                        if self.verbose > 1:
                            print widget.name
                elif wname == 'preset':
                    widget.options = options_dict[wname]
                    replace_actives = options_dict[wname][0]
                    if widget.value != replace_actives:
                        widget.value = replace_actives
                        if self.verbose > 1:
                            print widget.name

    def update_navigation(self, obj=None, attr=None, old=None, new=None):

        print obj.name
        if obj.name in ['filter1_school_name', 'filter1_grade']:
            if self.verbose > 1:
                print 'hard reset', obj.name
            self.cascade_inputs(hard_reset=True)
        elif obj.name not in ['filter0_indicator', 'filter0_preset']:
            self.cascade_inputs(hard_reset=False)
        else:
            if self.verbose > 1:
                print 'soft reset', obj.name

        if obj.name in ['filter1_school_name']:
            self.update_plot2()

        if obj.name not in ['filter0_indicator']:
            self.update_plot3()

        self.update_plot4()
        self.update_notifications()

        if obj.name in ['filter1_school_name', 'filter1_grade']:
            self.update_paragraph()
            self.update_red_flags()
        curdoc().add(self)

    def update_presets(self, obj=None, attr=None, old=None, new=None):
        labels = self.indicator_checkboxgroup.labels
        if new == 'All Aggregate Indicators':
            keep = ['ACADEMIC PERFORMANCE', 'READINESS INVESTMENT', 'LEARNING INVESTMENT']
        elif new == 'Academic Performance':
            keep = ['Literacy', 'Mathematics', 'Science', 'History']
        elif new == 'Readiness Investment':
            keep = ['Absence', 'Tardy', 'Call Ahead', 'Late Pickup', 'Early Dismissal', 'Uniform']
        elif new == 'Learning Investment':
            keep = ['Reading Log', 'Homework', 'NHM', 'Spelling']
        self.indicator_checkboxgroup.active = [labels.index(l) for l in labels if l in keep]

    def update_plot2(self):

        school_selected = self.get_widget_values(self.school_select)
        scholars = self.get_data('scholars')
        keep_grades = scholars.loc[scholars['school_name'].isin(school_selected), 'grade'].unique()
        scholars = scholars[scholars['grade'].isin(keep_grades)]
        achievement = self.get_data('achievement')
        achievement = achievement[achievement['scholar_id'].isin(scholars.index)]
        achievement['grade'] = achievement['scholar_id'].map(scholars['grade'].to_dict())
        achievement['school_name'] = achievement['scholar_id'].map(scholars['school_name'].to_dict())

        indicators_to_keep = achievement.set_index(['scholar_id', 'grade', 'indicator'])['achievement'].\
            unstack(['indicator', 'grade']).notnull().mean().gt(.10).unstack('grade')

        dfs = achievement.set_index(['indicator', 'grade', 'school_name'])['achievement'].\
            groupby(level=['indicator', 'grade']).\
            apply(lambda s: compare_distributions(s, ['school_name'])).\
            xs(self.get_widget_values(self.school_select).pop(), level='school_name').\
            unstack('grade').\
            where(indicators_to_keep).\
            stack().\
            to_frame('value').\
            reset_index()

        all_w = [
            'ACADEMIC PERFORMANCE', 'Literacy', 'Mathematics', 'Science', 'History', 'READINESS INVESTMENT',
            'Absence', 'Tardy', 'Call Ahead', 'Early Dismissal', 'Late Pickup', 'Uniform',
            'LEARNING INVESTMENT', 'Reading Log', 'Homework', 'NHM', 'Spelling'
        ]
        sorter_dict = dict(zip([x.lower() for x in all_w], reversed(range(1, len(all_w) + 1))))

        dfs['indicator_position'] = dfs['indicator'].str.lower().map(sorter_dict)
        dfs['grade_position'] = dfs['grade'].astype('int').sub(dfs['grade'].astype('int').min()) + 1
        dfs['fill_color'] = dfs['value'].gt(0.45).replace({True: 'grey', False: 'red'})
        dfs['line_color'] = dfs['value'].gt(0.45).replace({True: 'grey', False: 'red'})
        dfs.loc[dfs['value'].gt(0.45) & dfs['value'].lt(0.55), 'line_color'] = 'red'
        dfs['fill_alpha'] = dfs['value'].gt(0.45).replace({True: 0.2, False: 0.6})
        dfs['line_alpha'] = dfs['value'].gt(0.45).replace({True: 0.2, False: 0.6})
        dfs.loc[dfs['value'].gt(0.45) & dfs['value'].lt(0.55), 'line_alpha'] = 1.0
        dfs['radius'] = dfs['value'].mul(0.25)

        width = 170 + (dfs['grade_position'].nunique() * 50)
        height = dfs['indicator_position'].max() * 30

        source = ColumnDataSource(dfs)
        plot = Plot(
            # x_range=Range1d(start=dfs['grade_position'].min() - 0.5, end=dfs['grade_position'].max() + 0.5),
            # y_range=Range1d(start=dfs['indicator_position'].min() - 0.5, end=dfs['indicator_position'].max() + 0.5),
            x_range=FactorRange(factors=dfs.sort_values('grade_position')['grade'].unique().tolist()),
            y_range=FactorRange(factors=list(reversed([x.title() if x != 'NHM' else x for x in all_w]))),
            min_border_bottom=10,
            min_border_top=10,
            min_border_right=10,
            min_border_left=10,
            plot_width=width,
            plot_height=height,
            title=None,
            title_text_font_size='0pt',
            title_text_color='black',
            outline_line_alpha=0.0)

        circle = Circle(
            x='grade_position', y='indicator_position', radius='radius', line_color='line_color',
            fill_color='fill_color', fill_alpha='fill_alpha', line_alpha='line_alpha')
        plot.add_glyph(source, circle)

        plot.add_layout(
            CategoricalAxis(
                axis_label='Indicator', axis_label_text_font_size='10pt', minor_tick_line_alpha=0.0,
                axis_line_alpha=0.0, major_tick_line_alpha=0.0, major_label_text_color='grey',
                major_label_text_font_size='9pt'
            ), 'left')

        plot.add_layout(
            CategoricalAxis(
                axis_label='Grade', axis_label_text_font_size='10pt', minor_tick_line_alpha=0.0,
                axis_line_alpha=0.0, major_tick_line_alpha=0.0, major_label_text_color='grey',
                major_label_text_font_size='9pt'
            ), 'above')

        seg1 = Segment(
            x0=dfs['grade_position'].min() - 0.5, x1=dfs['grade_position'].max() + 0.5,
            y0=sorter_dict['readiness investment'] + 0.5, y1=sorter_dict['readiness investment'] + 0.5,
            line_width=1, line_color='#165788', line_alpha=0.2)
        seg2 = Segment(
            x0=dfs['grade_position'].min() - 0.5, x1=dfs['grade_position'].max() + 0.5,
            y0=sorter_dict['learning investment'] + 0.5, y1=sorter_dict['learning investment'] + 0.5,
            line_width=1, line_color='#165788', line_alpha=0.2)

        plot.add_glyph(source, seg1)
        plot.add_glyph(source, seg2)

        self.plot2.children = [[plot]]

    def update_plot3(self):

        scholar_selected = self.get_widget_values(self.scholar_select)
        teacher_selected = self.get_widget_values(self.teacher_select)
        school_selected = self.get_widget_values(self.school_select)
        grade_selected = self.get_widget_values(self.grade_select)

        scholars_all = self.get_data('scholars')
        scholars_all = scholars_all[scholars_all['grade'].isin(grade_selected)]

        if 'All scholars' not in scholar_selected:
            scholars = self.filter_by_widget(exclude=['school_name', 'teacher', 'scholar'])
            scholar_filter = scholars['scholar'].isin(scholar_selected) | ~scholars['school_name'].\
                isin(school_selected)
            scholars = scholars[scholar_filter]
        elif 'All teachers' not in teacher_selected:
            scholars = self.filter_by_widget(exclude=['school_name', 'teacher', 'scholar'])
            scholar_filter = check_isin(scholars['teacher'], teacher_selected) | ~scholars['school_name'].\
                isin(school_selected)
            scholars = scholars[scholar_filter]
        else:
            scholars = self.filter_by_widget(exclude=['school_name'])

        achievement = self.get_data('achievement')
        achievement = achievement[achievement['scholar_id'].isin(scholars_all.index)]
        achievement_filtered = achievement[achievement['scholar_id'].isin(scholars.index)].copy()

        achievement_filtered['grade'] = achievement_filtered['scholar_id'].map(scholars['grade'].to_dict())
        achievement_filtered['school_name'] = achievement_filtered['scholar_id'].map(scholars['school_name'].to_dict())
        achievement['grade'] = achievement['scholar_id'].map(scholars_all['grade'].to_dict())
        achievement['school_name'] = achievement['scholar_id'].map(scholars_all['school_name'].to_dict())

        weights = self.get_data('weights')
        weights = weights[weights['grade'].astype('str').isin(self.get_widget_values(self.grade_select))]

        dfs = achievement.set_index(['indicator', 'grade', 'school_name'])['achievement'].\
            groupby(level=['indicator', 'grade']).\
            apply(lambda s: compare_distributions(s, ['school_name'])).\
            xs(self.get_widget_values(self.school_select).pop(), level='school_name').\
            reset_index('grade', drop=True)

        dfs_filtered = achievement_filtered.set_index(['indicator', 'grade', 'school_name'])['achievement'].\
            groupby(level=['indicator', 'grade']).\
            apply(lambda s: compare_distributions(s, ['school_name'])).\
            xs(self.get_widget_values(self.school_select).pop(), level='school_name').\
            reset_index('grade', drop=True)

        indicators_to_keep = achievement.set_index(['scholar_id', 'indicator'])['achievement'].\
            unstack('indicator').notnull().mean().gt(.10)

        dfs = dfs[indicators_to_keep].to_frame('value').reset_index()
        dfs['value_filtered'] = dfs['indicator'].map(dfs_filtered.to_dict())
        dfs['weight'] = dfs['indicator'].map(weights.set_index('indicator')['value']).fillna(0.0)
        fill_weights = ['Academic Performance', 'Learning Investment', 'Readiness Investment']
        dfs.loc[dfs['indicator'].isin(fill_weights), 'weight'] = 1.0

        indicators = [
            'ACADEMIC PERFORMANCE', 'Literacy', 'Mathematics', 'Science', 'History', 'READINESS INVESTMENT', 'Absence',
            'Tardy', 'Call Ahead', 'Early Dismissal', 'Late Pickup', 'Uniform', 'LEARNING INVESTMENT', 'Reading Log',
            'Homework', 'NHM', 'Spelling'
        ]
        indicators = [i.lower() for i in indicators if i.lower() in dfs['indicator'].drop_duplicates().str.lower().tolist()]
        ind_range = list(reversed(range(len(indicators))))

        a_offset = .0

        dfs['low_border'] = 0.0
        dfs['middle_border'] = 0.45
        dfs['high_border'] = 1.0

        dfs['top_border'] = dfs['indicator'].str.lower().map(dict(zip(indicators, [x + 0.65 for x in ind_range])))
        dfs['bottom_border'] = dfs['indicator'].str.lower().map(dict(zip(indicators, [x + 1.35 for x in ind_range])))
        dfs['weight_top_border'] = dfs['indicator'].str.lower().map(dict(zip(indicators, [x + 0.85 for x in ind_range])))
        dfs['weight_bottom_border'] = dfs['indicator'].str.lower().map(dict(zip(indicators, [x + 1.15 for x in ind_range])))
        dfs['segment_top'] = dfs['indicator'].str.lower().map(dict(zip(indicators, [x + 0.5 for x in ind_range])))
        dfs['segment_bottom'] = dfs['indicator'].str.lower().map(dict(zip(indicators, [x + 1.5 for x in ind_range])))

        dfs['evaluation'] = 'Below'
        dfs.loc[dfs['value'].gt(0.45), 'evaluation'] = 'Above'

        source_a = ColumnDataSource(dfs)
        plot1 = Plot(
            x_range=Range1d(start=a_offset - 0.01, end=1.0 - a_offset + 0.06),
            y_range=Range1d(start=-0.01, end=len(indicators) + 1.06),
            min_border_bottom=10,
            min_border_top=10,
            min_border_right=10,
            min_border_left=10,
            plot_width=300,
            plot_height=(len(indicators) * 47),
            title='Health Score',
            title_text_font_size='12pt',
            title_text_color='black',
            outline_line_alpha=0.0)

        patch1 = Quad(
            left='low_border', right='middle_border', top='top_border', bottom='bottom_border',
            fill_color='red', fill_alpha=0.6, line_color=None)

        patch2 = Quad(
            left='middle_border', right='high_border', top='top_border', bottom='bottom_border',
            fill_color='grey', fill_alpha=0.2, line_color=None)

        plot1.add_glyph(source_a, patch1)
        plot1.add_glyph(source_a, patch2)

        seg1 = Segment(
            x0='value', x1='value', y0='segment_bottom', y1='segment_top',
            line_width=2, line_color='black', line_alpha=1.0)
        seg2 = Segment(
            x0='value_filtered', x1='value_filtered', y0='segment_bottom', y1='segment_top',
            line_width=1, line_color='black', line_alpha=1.0, line_dash=[5, 2])
        plot1.add_glyph(source_a, seg1)
        plot1.add_glyph(source_a, seg2)

        plot2 = Plot(
            x_range=Range1d(start=a_offset - 0.01, end=1.0 - a_offset + 0.06),
            y_range=Range1d(start=-0.01, end=len(indicators) + 1.06),
            min_border_bottom=10,
            min_border_top=10,
            min_border_right=10,
            min_border_left=10,
            plot_width=300,
            plot_height=(len(indicators) * 47),
            title='Indicator Weight',
            title_text_font_size='12pt',
            title_text_color='black',
            outline_line_alpha=0.0)

        patch_weight = Quad(
            left='low_border', right='weight', top='weight_top_border', bottom='weight_bottom_border',
            fill_color='#165788', fill_alpha=0.7, line_color=None)
        plot2.add_glyph(source_a, patch_weight)

        self.plot3a.children = [[plot1]]
        self.plot3b.children = [[plot2]]

    def update_plot4(self):

        # get list of indicators to display
        use_indicators = [x.lower() for x in self.get_widget_values(self.indicator_checkboxgroup)]

        # get list of all scholars within selected grade
        scholars_all = self.get_data('scholars')
        scholars_all = scholars_all[scholars_all['grade'].isin(self.get_widget_values(self.grade_select))]

        # get list of all scholars in selected grade that meet filter criteria (other than teacher or scholar focus)
        scholars_grade = self.filter_by_widget(exclude=['school_name', 'scholar', 'teacher'])

        # get list of all schoalrs in school that meet filter criteria
        scholars = self.filter_by_widget()

        achievement = self.get_data('achievement')
        achievement = achievement[achievement['indicator'].str.lower().isin(use_indicators)]
        achievement_grade = achievement[achievement['scholar_id'].isin(scholars_grade.index)]
        achievement_all = achievement[achievement['scholar_id'].isin(scholars_all.index)]

        indicator_list = achievement.indicator.unique().tolist()
        indicators = [indicator_list[[x.lower() for x in indicator_list].index(ui)] for ui in use_indicators]

        keep_metrics = ['lower_bar', 'q1', 'median', 'q3', 'upper_bar']
        boxplot_metrics = achievement.groupby('indicator')['achievement'].\
            apply(calculate_boxplot).loc(axis=0)[:, keep_metrics]
        boxplot_metrics_grade = achievement_grade.groupby('indicator')['achievement'].\
            apply(calculate_boxplot).loc(axis=0)[:, keep_metrics]
        # boxplot_metrics_all = achievement_all.groupby('indicator')['achievement'].\
        #    apply(calculate_boxplot).loc(axis=0)[:, keep_metrics]

        achievement = achievement[achievement['scholar_id'].isin(scholars.index)]

        boxplot_categories_grade = achievement_grade.set_index(['indicator', 'scholar_id']).\
            groupby(level='indicator')['achievement'].\
            apply(lambda grp: calculate_boxplot(grp, categorize=True)).\
            unstack('indicator').\
            loc[scholars.index, :].\
            rename(columns=lambda c: c + ' Category (Filtered)')
        boxplot_categories_all = achievement_all.set_index(['indicator', 'scholar_id']).\
            groupby(level='indicator')['achievement'].\
            apply(lambda grp: calculate_boxplot(grp, categorize=True)).\
            unstack('indicator').\
            loc[scholars.index, :].\
            rename(columns=lambda c: c + ' Category')

        plot_df = achievement.set_index(['indicator', 'scholar_id', 'reference_date'])['achievement'].\
            unstack('indicator').reset_index('reference_date', drop=True)

        save_df = plot_df.join(scholars.drop(['id'], axis=1)).\
            join(boxplot_categories_grade).\
            join(boxplot_categories_all)

        for indicator in indicators:
            plot_df[indicator + ' Offset'] = 0.5 + jitter_offset(plot_df[indicator].dropna())

        self.download_button.callback = CustomJS(
            args=dict(objArray=ColumnDataSource(save_df)), code=save_datasource_code)

        plot_df = plot_df.reset_index()
        plot_df['scholar'] = plot_df['scholar_id'].map(scholars['scholar'].to_dict())
        source = ColumnDataSource(plot_df)

        xrang = Range1d(start=-0.05, end=1.05)
        tooltips = '<div><span style="font-size: 12px;">@scholar (@scholar_id)</span>'
        global_width = 650
        global_height = 160

        plots = []
        for indicator in indicators:

            plot = Plot(
                x_range=xrang,
                y_range=Range1d(start=-0.05, end=1.05),
                min_border_top=0,
                min_border_bottom=0,
                min_border_right=0,
                min_border_left=50,
                plot_width=global_width,
                plot_height=global_height,
                title=None,
                title_text_font_size='0pt',
                outline_line_alpha=0.0)

            plot.add_layout(
                LinearAxis(
                    axis_label=indicator, axis_label_text_font_size='10pt', minor_tick_line_alpha=0.0,
                    axis_line_alpha=0.0, major_tick_line_alpha=0.0, major_label_text_color='grey',
                    major_label_text_font_size='0pt'
                ), 'left')

            plot.add_layout(
                LinearAxis(
                    axis_label=None, axis_label_text_font_size='0pt', minor_tick_line_alpha=0.0, axis_line_alpha=0.25,
                    major_tick_line_alpha=0.0, major_label_text_color='grey', major_label_text_font_size='7pt',
                ), 'below')

            lower, q1, q2, q3, upper = boxplot_metrics_grade.xs(indicator)

            rect_kwargs = dict(fill_color='#165788', line_color='white', line_alpha=0.0, fill_alpha=0.1)
            rect1_2 = Rect(x=(q1+lower)/2., y=0.5, width=(q1-lower), height=1.1, **rect_kwargs)
            seg3 = Segment(x0=q2, x1=q2, y0=-0.05, y1=1.05, line_color='#165788', line_width=2, line_alpha=0.25)
            rect4_5 = Rect(x=(upper+q3)/2., y=0.5, width=(upper-q3), height=1.1, **rect_kwargs)

            plot.add_glyph(rect1_2)
            plot.add_glyph(seg3)
            plot.add_glyph(rect4_5)

            scholar_markers = Circle(
                x=indicator, y=indicator + ' Offset', size=8, line_color='#f7910b', fill_color='#f7910b',
                fill_alpha=0.25, line_alpha=1.0)

            scholar_markers_s = Circle(
                x=indicator, y=indicator + ' Offset', size=10, line_color='#f7910b', fill_color='#f7910b',
                fill_alpha=0.25, line_alpha=1.0)

            scholar_markers_ns = Circle(
                x=indicator, y=indicator + ' Offset', size=8, line_color='#165788', fill_color='#165788',
                fill_alpha=0.25, line_alpha=0.25)

            plot.add_glyph(source, scholar_markers)
            scholar_renderer = GlyphRenderer(
                data_source=source, glyph=scholar_markers, selection_glyph=scholar_markers_s,
                nonselection_glyph=scholar_markers_ns, name='scholar_markers')

            hover_tool = HoverTool(plot=plot, renderers=[scholar_renderer], tooltips=tooltips, always_active=False)
            tap_tool = TapTool(plot=plot, renderers=[scholar_renderer])
            zoom_tool = BoxZoomTool(plot=plot, dimensions=['width'])
            reset_tool = ResetTool(plot=plot)
            save_tool = PreviewSaveTool(plot=plot)
            pan_tool = PanTool(plot=plot, dimensions=['width'])
            plot.tools.extend([hover_tool, zoom_tool, tap_tool, pan_tool, reset_tool, save_tool])
            plot.renderers.extend([scholar_renderer])

            lower, q1, q2, q3, upper = boxplot_metrics.xs(indicator)

            seg_middle2 = Segment(x0=q2, x1=q2, y0=0.25, y1=0.75, line_color='black', line_width=1)
            seg_lower2 = Segment(x0=lower, x1=q1, y0=0.5, y1=0.5, line_color='black', line_width=1)
            seg_upper2 = Segment(x0=q3, x1=upper, y0=0.5, y1=0.5, line_color='black', line_width=1)

            plot.add_glyph(seg_middle2)
            plot.add_glyph(seg_lower2)
            plot.add_glyph(seg_upper2)

            plots.append([plot])

        self.plot4.children = plots
        self.plot4.toolbar_location = 'right'
        self.plot4.border_space = 0

    def update_notifications(self):

        use_indicators = [x.lower() for x in self.get_widget_values(self.indicator_checkboxgroup)]
        achievement = self.get_data('achievement')
        achievement = achievement[achievement['indicator'].str.lower().isin(use_indicators)]
        indicator_list = achievement.indicator.unique().tolist()
        indicators = [indicator_list[[x.lower() for x in indicator_list].index(ui)] for ui in use_indicators]

        notification_text = '  |  '.join(indicators) + '\n' + \
            self.get_widget_values(self.school_select)[0] + ' - ' + \
            'Grade {}'.format(self.get_widget_values(self.grade_select)[0]) + '\n' + \
            'Data current as of {}'.format(self.reference_date)
        filtered_shape = self.filter_by_widget().shape
        original_shape = self.filter_by_widget(exclude=self._get_options_tier2().keys()).shape
        if original_shape > filtered_shape:
            notification_text += '\nFilters applied'
        self.notification_field.text = notification_text

    def update_records(self, obj=None, attr=None, old=None, new=None):
        if obj.name in ('filter0_datatable', ):
            ids = [int(obj.labels[i]) for i in obj.active] if len(obj.active) > 0 else None
            self.update_paragraph(scholar_ids=ids)
        else:
            self.update_paragraph()
        curdoc().add(self)

    def get_selected_scholars(self):

        scholar_ids = []
        for p in self.properties_with_values().values():
            if type(p) in (Plot, GridPlot):
                for cds in p.select(ColumnDataSource):
                    if 'scholar_id' in cds.column_names:
                        for v in cds.selected.values():
                            scholar_ids += [int(cds.data['scholar_id'][ind]) for ind in v['indices']]

        return scholar_ids

    def update_red_flags(self):
        # get list of all scholars within selected grade
        scholars_all = self.get_data('scholars')
        scholars_all = scholars_all[scholars_all['grade'].isin(self.get_widget_values(self.grade_select))]

        # get list of all schoalrs in school that meet filter criteria
        scholars = self.filter_by_widget()

        achievement = self.get_data('achievement')
        achievement_all = achievement[achievement['scholar_id'].isin(scholars_all.index)]

        c_dict = {'Lower Bar': '-', 'Lower Outliers': '--', 'Middle Box': '', 'Upper Bar': '+', 'Upper Outliers': '++'}
        keep_cols = [
            'Academic Performance', 'Literacy', 'Mathematics', 'Science', 'History', 'Readiness Investment',
            'Learning Investment']
        check_cols = ['Academic Performance', 'Literacy', 'Mathematics', 'Science', 'History']
        boxplot_categories = achievement_all.set_index(['indicator', 'scholar_id']).\
            groupby(level='indicator')['achievement'].\
            apply(lambda grp: calculate_boxplot(grp, categorize=True)).\
            replace(c_dict).\
            unstack('indicator').\
            loc[scholars.index, keep_cols].\
            fillna('')

        has_holdover = scholars['holdover_nav'].astype('int').gt(0)
        no_sped = scholars['sped_nav'].apply(lambda l: 'No services' in l)
        no_rti = scholars['rti_nav'].apply(lambda l: 'No interventions' in l)
        n_lower = boxplot_categories[check_cols].apply(lambda col: col.str.contains(r'[-]')).sum(axis=1)

        boxplot_categories['priority'] = 4
        boxplot_categories.loc[n_lower.gt(0) & has_holdover & no_sped & no_rti, 'priority'] = 1
        boxplot_categories.loc[n_lower.gt(0) & ~has_holdover & no_sped & no_rti, 'priority'] = 2
        boxplot_categories.loc[n_lower.gt(0) & ~has_holdover & ~(no_sped & no_rti), 'priority'] = 3
        boxplot_categories['Priority'] = boxplot_categories['priority'].\
            map({1: 'HO: yes, IEP: no', 2: 'HO: no, IEP: no', 3: 'HO: no, IEP: yes'})
        boxplot_categories = boxplot_categories.dropna(subset=['Priority'])
        boxplot_categories = boxplot_categories[boxplot_categories['Priority'].ne(4)].\
            sort_values(
                ['priority', 'Academic Performance', 'Literacy', 'Mathematics', 'Science', 'History'],
                ascending=[True, False, False, False, False, False]
            ).drop(['priority'], axis=1)
        boxplot_categories = boxplot_categories.reset_index()
        boxplot_categories['Scholar Name'] = boxplot_categories['scholar_id'].map(scholars['scholar'])

        self.datatable_checkboxgroup.active = []
        self.datatable_checkboxgroup.labels = boxplot_categories['scholar_id'].astype('int').astype('str').tolist()
        self.datatable6.source = ColumnDataSource(boxplot_categories)
        self.datatable6.height = 26 * (boxplot_categories.shape[0] + 1)

    def update_paragraph(self, scholar_ids=None):

        indicator_dict = {
            'Academic Performance': ['Literacy', 'Mathematics', 'Science', 'History'],
            'Readiness Investment': ['Absence', 'Tardy', 'Call Ahead', 'Late Pickup', 'Early Dismissal', 'Uniform'],
            'Learning Investment': ['Reading Log', 'Homework', 'NHM', 'Spelling']
        }

        if scholar_ids is None:
            scholar_ids = self.get_selected_scholars()

        # get list of all scholars within selected grade
        scholars_all = self.get_data('scholars')
        scholars_all = scholars_all[scholars_all['grade'].isin(self.get_widget_values(self.grade_select))]

        # get list of all scholars in selected grade that meet filter criteria (other than teacher or scholar focus)
        scholars_grade = self.filter_by_widget(exclude=['school_name', 'scholar', 'teacher'])

        # get list of all schoalrs in school that meet filter criteria
        scholars = self.filter_by_widget()

        achievement = self.get_data('achievement')
        # achievement_grade = achievement[achievement['scholar_id'].isin(scholars_grade.index)]
        achievement_all = achievement[achievement['scholar_id'].isin(scholars_all.index)]

        # keep_metrics = ['q1', 'median', 'q3']
        # boxplot_metrics = achievement.groupby('indicator')['achievement'].\
        #     apply(calculate_boxplot).loc(axis=0)[:, keep_metrics]
        # boxplot_metrics_grade = achievement_grade.groupby('indicator')['achievement'].\
        #     apply(calculate_boxplot).loc(axis=0)[:, keep_metrics]
        # boxplot_metrics_all = achievement_all.groupby('indicator')['achievement'].\
        #     apply(calculate_boxplot).loc(axis=0)[:, keep_metrics]

        achievement = achievement[achievement['scholar_id'].isin(scholars.index)]
        indicators = achievement.indicator.unique().tolist()

        boxplot_categories_grade = achievement_all.set_index(['indicator', 'scholar_id']).\
            groupby(level='indicator')['achievement'].\
            apply(lambda grp: calculate_boxplot(grp, categorize=True)).\
            unstack('indicator').\
            loc[scholars.index, :].\
            rename(columns=lambda c: c + ' Category')

        plot_df = achievement.set_index(['indicator', 'scholar_id', 'reference_date'])['achievement'].\
            unstack('indicator').reset_index('reference_date', drop=True)

        save_df = plot_df.join(scholars.drop(['id'], axis=1)).join(boxplot_categories_grade)

        if len(scholar_ids) == 0:

            text = ''
            for k in ['Academic Performance', 'Readiness Investment', 'Learning Investment']:
                history = self.get_data('history')
                v = indicator_dict[k]
                if k in indicators:
                    use_indicators = v
                else:
                    use_indicators = [i for i in v if i in indicators]
                if len(use_indicators) == 0:
                    continue
                sorter_dict = dict(zip(use_indicators, range(len(use_indicators))))
                text += '==' + k + '==' + '\n\n'
                history = history[history['indicator'].isin(use_indicators)]
                grade_filter = history['grade'].astype('str').isin(self.get_widget_values(self.grade_select))
                scholar_filter = history['scholar_id'].isin(scholars.index)
                excused_filter = history['score'].lt(1.0)
                if k == 'Academic Performance':
                    school_history = history[scholar_filter & grade_filter].\
                        groupby(['indicator', 'name', 'date'])['score'].\
                        mean().reset_index().sort_values('date')
                    mean_dict = history[history['name'].isin(school_history['name'].unique()) & grade_filter]. \
                        groupby('name')['score'].mean().to_dict()
                    school_history['Grade Average'] = school_history['name'].map(mean_dict).\
                        apply(lambda x: '{:0.2f}'.format(x).replace('nan', 'n/a'))
                    school_history['score'] = school_history['score'].\
                        apply(lambda x: '{:0.2f}'.format(x).replace('nan', 'n/a'))
                    school_history['sorter'] = school_history['indicator'].map(sorter_dict)
                    school_history = school_history.sort_values(['sorter', 'date'], ascending=[True, False]).\
                        rename(columns={'score': 'Score', 'name': 'Assessment', 'indicator': 'Indicator'}).\
                        set_index(['Indicator', 'Assessment'])[['Score', 'Grade Average']]
                else:
                    school_history = history[scholar_filter & grade_filter & excused_filter].\
                        groupby(['indicator', 'score', 'date'])['name'].\
                        count().reset_index().sort_values('date').rename(columns={'score': 'excused', 'name': 'Count'})
                    indicator_filter = history['indicator'].isin(school_history['indicator'].unique())
                    filtered_history = history[indicator_filter & grade_filter & excused_filter].copy()
                    filtered_history['school_name'] = filtered_history['scholar_id'].\
                        map(scholars_grade['school_name'].to_dict())
                    mean_dict = filtered_history.dropna(subset=['school_name']).\
                        groupby(['date', 'school_name'])['score'].count().\
                        groupby(level=['date']).mean().to_dict()
                    school_history['Grade Average'] = school_history['date'].map(mean_dict).\
                        apply(lambda x: '{:0.1f}'.format(x).replace('nan', 'n/a'))
                    school_history['sorter'] = school_history['indicator'].map(sorter_dict)
                    school_history = school_history.sort_values(['sorter', 'date'], ascending=[True, False]).\
                        rename(columns={'date': 'Date', 'indicator': 'Indicator'}).\
                        set_index(['Indicator', 'Date'])[['Count', 'Grade Average']]

                text += school_history.to_string(sparsify=True) + '\n\n'
        else:
            reference_df = save_df.loc[scholar_ids, :].copy()
            text = ''
            divider = '-' * 60
            for scholar_id in scholar_ids:
                scholar_id = int(scholar_id)
                s = reference_df.loc[scholar_id].copy()
                s.loc[indicators] = s.loc[indicators].apply(lambda x: '{:0.2f}'.format(x).replace('nan', 'n/a'))
                achievement = s[indicators].to_frame('Index')
                achievement['Category'] = s[[i + ' Category' for i in indicators]]. \
                    rename(index=lambda x: x.replace(' Category', ''))
                achievement.index.names = [None]

                individual_cols = ['scholar', 'school_name', 'start_date', 'ell_status', 'grade']
                scholar_name, school_name, start_date, ell_status, grade = s[individual_cols]

                header1 = '{scholar_name} ({scholar_id})'.format(scholar_name=scholar_name, scholar_id=scholar_id)
                header2 = '{school_name} - Grade {grade}'.format(school_name=school_name, grade=grade)
                line1_left = 'SA scholar since: {start_date}'.format(start_date=start_date)
                line1_right = 'ELL status: {ell_status}'.format(ell_status=ell_status)
                spacer = (len(divider) - len(line1_left) - len(line1_right)) * ' '
                line1 = line1_left + spacer + line1_right

                teacher = Series(s['teacher'], index=['Homeroom', 'Literacy', 'Mathematics', 'Science', 'History']). \
                    to_frame('Teachers')

                # text = ''
                text += divider + '\n'
                text += header1 + '\n'
                text += header2 + '\n'
                text += divider + '\n'
                text += line1 + '\n'
                text += '\n'
                text += teacher.to_string(index=True) + '\n'

                if (s['skip_nav'] > 0) or (s['holdover_nav'] > 0):
                    skiphold_df = DataFrame(columns=['Academic Year', '     Skips', '     Holdovers'])
                    skips = s['skip']
                    holds = s['holdover']
                    for i in range(len(skips)):
                        skiphold_df.loc[i, ['Academic Year', '     Skips']] = skips[i]
                    for i in range(len(holds)):
                        skiphold_df.loc[i, ['Academic Year', '     Holdovers']] = holds[i]

                    if skiphold_df.shape[0] == 0:
                        skiphold_df.loc[0, :] = '--', '--', '--'
                    skiphold_df = skiphold_df.fillna('--').sort_values('Academic Year')
                    text += '\n'
                    text += skiphold_df.to_string(index=False) + '\n'

                if 'No servies' not in s['sped_nav']:
                    sped_df = DataFrame(
                        columns=['SpED Service', 'Subject', 'Frequency', 'Duration', 'Size', 'Time'], index=range(len(s['sped'])))
                    for i in sped_df.index:
                        sped_df.loc[i, :] = s['sped'][i]
                    text += '\n'
                    text += sped_df.to_string(index=False) + '\n'

                if 'No interventions' not in s['rti_nav']:
                    rti_df = DataFrame(
                        columns=['Intervention', 'Days in Intervention'], index=range(len(s['rti'])))
                    for i in rti_df.index:
                        rti_df.loc[i, :] = s['rti'][i]
                    text += '\n'
                    text += rti_df.to_string(index=False) + '\n'

                if len(s['state_exam']) > 0:
                    state_exam_df = DataFrame(
                        columns=['Subject', '  Academic Year', 'State Exam Score'], index=range(len(s['state_exam'])))
                    for i in state_exam_df.index:
                        state_exam_df.loc[i, :] = s['state_exam'][i]
                    state_exam_df = state_exam_df.sort_values(['  Academic Year', 'Subject'])
                    text += '\n'
                    text += state_exam_df.to_string(index=False) + '\n'

                text += '\n\n'

                for k in ['Academic Performance', 'Readiness Investment', 'Learning Investment']:
                    history = self.get_data('history')
                    v = indicator_dict[k]
                    if k in indicators:
                        use_indicators = v
                    else:
                        use_indicators = [i for i in v if i in indicators]
                    if len(use_indicators) == 0:
                        continue
                    text += '==' + k + '==' + '\n\n'
                    history = history[history['indicator'].isin(use_indicators)]
                    scholar_history = history[history['scholar_id'].eq(scholar_id)].sort_values('date')
                    if k == 'Academic Performance':
                        mean_dict = history[history['name'].isin(scholar_history['name'].unique())]. \
                            groupby('name')['score'].mean().to_dict()
                        scholar_history['Grade-Wide Average'] = scholar_history['name'].map(mean_dict)
                        keep_cols = ['date', 'score', 'Grade-Wide Average']
                    else:
                        keep_cols = ['date', 'score']
                    for indicator in use_indicators:
                        if indicator not in scholar_history['indicator'].unique():
                            continue
                        text += '--' + indicator + '--' + '\n\n'
                        scholar_history_i = scholar_history[scholar_history['indicator'].eq(indicator)]
                        for name in scholar_history_i['name'].unique():
                            use_df = scholar_history_i[scholar_history_i['name'].eq(name)][keep_cols]. \
                                rename(columns=lambda x: x.title())
                            text += name + '\n' + use_df.to_string(index=False) + '\n\n'
                        text += '\n\n'
                    text += '\n'

        self.report_field.text = text

    def get_data(self, set_name):

        if set_name not in ('scholars', 'weights', 'achievement', 'history'):
            raise NotImplementedError

        conn = create_engine(self.connection_string)

        if self.reference_date in storage.keys():
            if set_name in storage[self.reference_date].keys():
                if self.verbose > 2:
                    print 'Pulled {} from cache.'.format(set_name)
                return storage[self.reference_date][set_name].copy()
        else:
            ref_dates = read_sql('SELECT DISTINCT reference_date::text FROM scholars', conn)['reference_date'].tolist()
            for k in storage.keys():
                if k not in ref_dates:
                    _ = storage.pop(k)
            storage[self.reference_date] = dict()

        if set_name == 'scholars':
            query = '''
            SELECT
                id, scholar_id, scholar, start_date, school_name, school, school_age, school_type, grade,
                ell_status, skip_nav, holdover_nav, reference_date,
                COALESCE(state_exam_literacy_fails::text, 'N/A') AS state_exam_literacy_fails,
                COALESCE(state_exam_mathematics_fails::text, 'N/A') AS state_exam_mathematics_fails,
                COALESCE(state_exam_science_fails::text, 'N/A') AS state_exam_science_fails,
                string_to_array(regexp_replace(teacher, '^\{|\}$|\{\{|\}\}|"', '', 'g'), '},{') AS teacher,
                string_to_array(regexp_replace(skip, '^\{|\}$|\{\{|\}\}|"', '', 'g'), '},{') AS skip,
                string_to_array(regexp_replace(holdover, '^\{|\}$|\{\{|\}\}|"', '', 'g'), '},{') AS holdover,
                string_to_array(regexp_replace(state_exam, '^\{|\}$|\{\{|\}\}|"', '', 'g'), '},{') AS state_exam,
                string_to_array(regexp_replace(sped, '^\{|\}$|\{\{|\}\}|"', '', 'g'), '},{') AS sped,
                string_to_array(regexp_replace(rti, '^\{|\}$|\{\{|\}\}|"', '', 'g'), '},{') AS rti,
                regexp_replace(sped_nav, '^\{|\}$|\{\{|\}\}|"', '', 'g') AS sped_nav,
                regexp_replace(rti_nav, '^\{|\}$|\{\{|\}\}|"', '', 'g') AS rti_nav
            FROM scholars
            '''
            query += "WHERE reference_date = '{ref_date}'".format(ref_date=self.reference_date)
        elif set_name == 'history':
            query = '''
            SELECT
                indicator,
                name,
                scholar_id,
                date,
                grade,
                score,
                reference_date
            FROM history
            WHERE reference_date = '{ref_date}'
            '''.format(ref_date=self.reference_date)
        elif set_name == 'weights':
            query = '''
            SELECT
                indicator,
                grade,
                value,
                reference_date
            FROM weights
            WHERE reference_date = '{ref_date}'
            '''.format(ref_date=self.reference_date)
        elif set_name == 'achievement':
            query = '''
            SELECT
                indicator,
                scholar_id,
                achievement,
                reference_date
            FROM achievement
            WHERE reference_date = '{ref_date}'
            '''.format(ref_date=self.reference_date)

        data = read_sql(query, conn)

        if set_name == 'scholars':
            data['skip'] = data['skip'].apply(lambda y: [z.split(',') for z in y])
            data['holdover'] = data['holdover'].apply(lambda y: [z.split(',') for z in y])
            data['state_exam'] = data['state_exam'].apply(lambda y: [z.split(',') for z in y])
            data['sped'] = data['sped'].apply(lambda y: [z.split(',') for z in y])
            data['sped_nav'] = data['sped_nav'].apply(lambda y: y.split(','))
            data['rti'] = data['rti'].apply(lambda y: [z.split(',') for z in y])
            data['rti_nav'] = data['rti_nav'].apply(lambda y: y.split(','))
            data['teacher'] = data['teacher'].apply(lambda y: split(r'(?<!\s),(?!\s)', y[0]))
            data = data.set_index(['scholar_id'])

        storage[self.reference_date][set_name] = data.copy()
        if self.verbose > 0:
            print 'Pulled {} from database.'.format(set_name)

        return data
