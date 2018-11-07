from os.path import dirname, abspath, join
from os import getcwd


def standardize_values(v):
    if type(v) == list:
        if len(v) > 0:
            if type(v[0]) == list:
                output = v
            else:
                joiner = '; ' if ',' in v[0] else ', '
                output = joiner.join(v)
        else:
            output = v
    else:
        output = v

    return output


def unique_values(k):
    newk = []
    for i in k:
        if i not in newk:
            newk.append(i)
    newk = list(set([x for y in newk for x in y]))
    return newk


def isin(x, compare):
    return any(i for i in compare if i in x)


def check_isin(s, pattern_list, to_lower=True):
    if list not in s.apply(type).unique():
        if to_lower:
            return s.str.lower().isin([x.lower() for x in pattern_list])
        else:
            return s.isin(pattern_list)
    else:
        if to_lower:
            return s.apply(lambda x: any(i for i in pattern_list if i.lower() in [y.lower() for y in x]))
        else:
            return s.apply(lambda x: any(i for i in pattern_list if i in x))


def load_sql_template(sql_files, sql_path=None):

    if sql_path is None:
        try:
            sql_path = join(dirname(dirname(dirname(abspath(__file__)))), 'sql')
        except NameError:
            sql_path = join(getcwd(), 'magnus', 'sql')
            print 'Working in dev mode.'

    text = ''

    if type(sql_files) != list:
        sql_files = [sql_files]

    for script_file in sql_files:
        with open(join(sql_path, script_file), 'r') as sqlfile:
            text += '\n\n' + sqlfile.read()
    return text


def pct_to_string(string):
    try:
        output = '{}%'.format(int(string * 100))
    except ValueError:
        output = None
    return None
