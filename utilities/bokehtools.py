from bokeh.models.widgets.groups import CheckboxGroup


def bk_set_actives(widget, options, hard_reset=False):
    if type(widget) not in (CheckboxGroup,):
        raise TypeError('_set_active only applicable to CheckboxGroup widgets')
    if hard_reset:
        widget.active = [i for i in range(len(widget.labels)) if (widget.labels[i] in options)]
    else:
        widget.active = [i for i in range(len(widget.labels)) if (widget.labels[i] in options) and (i in widget.active)]


def bk_setup_widget_event(widget, callback):
    if hasattr(widget, 'active'):
        if widget:
            widget.on_change('active', callback)
    elif hasattr(widget, 'value'):
        if widget:
            widget.on_change('value', callback)
    elif hasattr(widget, 'clicks'):
        if widget:
            widget.on_change('clicks', callback)


def bk_get_widget_values(widget, active=True):
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
