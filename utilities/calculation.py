from pandas import Series, DataFrame, concat
from numpy import NaN, Inf, floor, argmin, sort, random, isnan
from sklearn.neighbors import KDTree
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from numpy.random import permutation


def combine_subjects_by_rules(df, constituent, output_weights=True):

    weight_dict = {
        (-1, 0, 1, 2, 3, 4): {
            'Literacy': 0.5,
            'Mathematics': 0.4,
            'Science': 0.1,
            'History': 0.0,

            'Absence': 0.3,
            'Call Ahead': 0.1,
            'Early Dismissal': 0.1,
            'Late Pickup': 0.1,
            'Tardy': 0.3,
            'Uniform': 0.1,

            'Reading Log': 0.3,
            'Homework': 0.2,
            'NHM': 0.1,
            'Spelling': 0.1
        },
        (5, 6, 7, 8, 9, 10, 11, 12): {
            'Literacy': 0.4,
            'Mathematics': 0.4,
            'Science': 0.1,
            'History': 0.1,

            'Absence': 0.3,
            'Call Ahead': 0.1,
            'Early Dismissal': 0.1,
            'Late Pickup': 0.1,
            'Tardy': 0.3,
            'Uniform': 0.1,

            'Reading Log': 0.3,
            'Homework': 0.0,
            'NHM': 0.1,
            'Spelling': 0.1
        }
    }

    df_copy = df.copy()
    keep = df_copy.notnull().mean().gt(0.0)
    weights = Series(index=df_copy.columns).fillna(0.0)

    for k, v in weight_dict.items():
        if constituent in k:
            for indicator in weights.index:
                weights[indicator] = v[indicator]

    output = df_copy.multiply(weights).loc[:, keep].sum(axis=1).div(weights[keep].sum())
    weights = weights[keep] / weights[keep].sum()

    if output_weights:
        return weights, output
    else:
        return output


def combine_subjects(df, output_weights=True):
    # print df.index.get_level_values('grade')[0]
    df_copy = df.copy()
    weights = Series(index=df_copy.columns).fillna(0.0)
    df = df.loc[:, ~df.isnull().all()]
    df = df.fillna(df.mean())

    lr = RandomForestRegressor(n_estimators=50)
    for col in df.columns:
        use_df = df.dropna(subset=[col]).drop([col], axis=1)
        use_df = use_df.fillna(use_df.mean())
        _ = lr.fit(use_df, df.dropna(subset=[col]).fillna(df.mean())[col])
        corr = Series(lr.predict(use_df), index=df.index).corr(df_copy[col])
        corr = 0.0 if isnan(corr) else corr
        importances = Series(lr.feature_importances_, index=use_df.columns).reindex(weights.index).fillna(0.0)
        weights += (importances * corr)

    weights = weights.where(weights.ge(0.0), 0.0).div(float(df.shape[1] - 1)) * df_copy.notnull().mean()
    # weights *= (df.var() / df.var().max()).fillna(0.0)
    # weights = weights.where(weights.gt(0.1) | weights.eq(0.0), 0.1)
    # print '---------------'
    # print df.index.get_level_values('grade').values[0]
    # print weights / weights.sum()
    # other_weights = df.corr().where(df.corr().ne(1.0)).mean()
    # print other_weights / other_weights.sum()
    output = df_copy[df.columns].fillna(1.0).reindex(columns=weights.index).multiply(weights).sum(axis=1).div(weights.sum())
    if output_weights:
        return weights, output
    else:
        return output


def combine_subjects_lr(df):

    weights = Series(index=df.columns).fillna(0.0)
    df = df.loc[:, ~df.isnull().all()]
    df = df.fillna(df.mean())

    lr = LinearRegression(fit_intercept=False)
    for col in df.columns:
        use_df = df.drop([col], axis=1)
        _ = lr.fit(use_df, df[[col]])
        corr = Series(lr.predict(use_df).ravel(), index=df.index).corr(df[col])
        corr = 0.0 if isnan(corr) else corr
        coefs = Series(lr.coef_.ravel(), index=use_df.columns).reindex(weights.index).fillna(0.0)
        weights += (corr * coefs)

    weights = weights.where(weights.ge(0.0), 0.0).div(float(df.shape[1] - 1))
    # weights = weights.where(weights.gt(0.1) | weights.eq(0.0), 0.1)
    print '---------------'
    print df.index.get_level_values('grade').values[0]
    print weights / weights.sum()
    # other_weights = df.corr().where(df.corr().ne(1.0)).mean()
    # print other_weights / other_weights.sum()
    output = df.reindex(columns=weights.index).multiply(weights).sum(axis=1).div(weights.sum())
    return output


def combine_subjects_original(df):
    weights = df.corr().mean()
    weights = weights.fillna(weights.mean())
    weights = weights.where(weights.ge(0.0), 0.0)
    output = df.multiply(weights).sum(axis=1).div(weights.sum())
    return output


def combine_subjects_pca(dataframe, diagnostics=False):

    df = dataframe.loc[:, dataframe.notnull().any().values].copy()
    df = df.fillna(df.mean())
    pca = PCA(n_components=df.shape[1])
    random_outcome = 0
    canonical_outcome = pca.fit(df).explained_variance_ratio_
    for _ in range(50):
        new_df = df.apply(lambda x: permutation(x), axis=1).apply(lambda x: permutation(x), axis=0)
        random_outcome += pca.fit(new_df).explained_variance_ratio_
    random_outcome /= 50
    keep_ind = ((canonical_outcome - random_outcome) > 0).sum()
    pca = PCA(n_components=keep_ind)
    proj = pca.fit_transform(df)
    proj = pca.inverse_transform(proj)
    proj = DataFrame(proj, columns=df.columns, index=df.index)
    weights = proj.var() / proj.var().sum()
    proj = proj.multiply(weights).sum(axis=1).div(weights.sum()).to_frame('composite')
    corr = concat([df, proj], axis=1).corr().loc[df.columns, :].drop(df.columns, axis=1)
    output = proj['composite']

    if diagnostics:
        return corr, pca.explained_variance_ratio_, output
    else:
        return output


def combined_subject_diagnostics(achievement_df):
    corrs = []
    expl_var = []
    weights = []
    for grade in sorted(achievement_df.index.get_level_values('grade').drop_duplicates()):
        adf = achievement_df.loc(axis=0)[:, grade]
        c, e, w = combine_subjects(adf, diagnostics=True)
        corrs.append(c)
        expl_var.append(e)
        weights.append(w)

    for ind, s in enumerate(expl_var):
        print ''
        print ind
        print s


def estimate_growth(s1, s2, match_levels, join_cols=None):
    if s1.empty:
        return Series([NaN] * s2.shape[0], index=s2.index)
    if s2.empty:
        return Series([NaN] * s1.shape[0], index=s1.index)

    join_cols = join_cols if join_cols is not None else s1.index.names
    drop_levels = [indname for indname in s1.index.names if indname not in match_levels]
    s1 = s1.dropna().copy()
    s2 = s2.dropna().copy()
    s1 = s1.reset_index(drop_levels, drop=True)
    s2_scholars = s2.reset_index(drop_levels, drop=True)
    s1 = s1.reindex_like(s2_scholars)
    s1.index = s2.index

    def calculate_max_change(df):
        if df['change'].notnull().sum() < 10:
            return Series([df['change'].max()] * df.shape[0], index=df.index)
        vals = df[['start_score', 'change']].dropna()
        kdt = KDTree(vals[['start_score']]).query_radius(vals[['start_score']], r=0.025)
        output_values = DataFrame.from_records(kdt, index=vals.index).stack()
        output_values.loc[:] = vals['change'].iloc[output_values.values].values
        output_values = output_values.groupby(level=vals.index.names).agg([max, min, lambda x: x.count()]).\
            reindex(df.index).rename(columns={'<lambda>': 'count'})
        return output_values

    s_comp = s1.to_frame('start_score')
    s_comp['end_score'] = s2
    s_comp['change'] = (s_comp['end_score'] / s_comp['start_score']).replace({Inf: NaN, -Inf: NaN}) - 1
    s_comp.loc[s_comp['change'].gt(1.0), 'change'] = 1.0
    changes = s_comp.groupby(level=join_cols, group_keys=False).apply(calculate_max_change)
    #changes = changes.drop([0], axis=1)
    s_comp[['max_change', 'min_change', 'n_neighbors']] = changes[['max', 'min', 'count']]
    s_comp['score'] = (s_comp['change'] / s_comp['max_change'])

    # handle negative maximum change
    neg_change = s_comp['max_change'].le(0.0)
    alt_change = (s_comp['change'].loc[neg_change].abs() / s_comp['min_change'].loc[neg_change].abs()) * -1.0
    s_comp.loc[neg_change, 'score'] = alt_change
    s_comp.loc[s_comp['score'].lt(-1.0), 'score'] = s_comp.loc[s_comp['score'].lt(-1.0), 'change']
    s_comp.loc[s_comp['start_score'].eq(0.0), 'score'] = s_comp.loc[s_comp['start_score'].eq(0.0), 'end_score']

    # safeguards
    s_comp.loc[s_comp['score'].gt(1.0), 'score'] = 1.0
    s_comp.loc[s_comp['score'].lt(-1.0), 'score'] = -1.0
    s_comp['score'] = s_comp['score'].replace({Inf: NaN, -Inf: NaN})

    # downweight small neighborhoods
    s_comp['score'] *= (s_comp['n_neighbors'].fillna(1.0) / s_comp['n_neighbors'].add(2).fillna(1.0))

    output = s_comp['score']

    return output


def calculate_boxplot(x, categorize=False):

    q1, q2, q3 = x.quantile(q=[0.25, 0.5, 0.75])
    iqr = q3 - q1
    upper_bar = q3 + iqr
    lower_bar = q1 - iqr
    upper_bar_members = ((x < upper_bar) & (x > q3)).mean()
    lower_bar_members = ((x > lower_bar) & (x < q1)).mean()
    upper_outliers = (x > upper_bar).mean()
    lower_outliers = (x < lower_bar).mean()

    if categorize:
        output = Series(['Middle Box'] * x.shape[0], index=x.index)
        output.loc[(x < upper_bar) & (x > q3)] = 'Upper Bar'
        output.loc[(x > lower_bar) & (x < q1)] = 'Lower Bar'
        output.loc[x > upper_bar] = 'Upper Outliers'
        output.loc[x < lower_bar] = 'Lower Outliers'
        output = output.fillna('Uncategorized')
    else:
        upper_bar = x[(x <= upper_bar) & (x >= q3)].max()
        lower_bar = x[(x >= lower_bar) & (x <= q1)].min()

        output = Series(
            [
                lower_outliers, lower_bar_members, lower_bar,
                q1, q2, q3, iqr,
                upper_bar, upper_bar_members, upper_outliers
            ],
            index=[
                'lower_outliers', 'lower_bar_members', 'lower_bar',
                'q1', 'median', 'q3', 'iqr',
                'upper_bar', 'upper_bar_members', 'upper_outliers',
            ]
        )

    return output


def hpd(x, alpha):
    """Calculate highest posterior density (HPD) of array for given alpha (adapted from PyMC
    source code at https://github.com/pymc-devs/pymc/blob/2.3/pymc/utils.py).

    Arguments:
        x (numpy.array): An array containing MCMC samples
        alpha (float): Desired probability of type I error

    Returns:
        list: lower and upper bounds of HPD


    """

    x = x.copy()
    x = sort(x)
    n = len(x)
    cred_mass = 1.0 - alpha

    interval_idx_inc = int(floor(cred_mass * n))
    n_intervals = n - interval_idx_inc
    interval_width = x[interval_idx_inc:] - x[:n_intervals]

    if len(interval_width) == 0:
        raise Exception('Too few elements for interval calculation')

    min_idx = argmin(interval_width)
    hdi_min = x[min_idx]
    hdi_max = x[min_idx + interval_idx_inc]
    return [hdi_min, hdi_max]


def cut_scores(s, min_cut=0.5, labelled_output=False, verbose=True, adjust_min=False):

    if s.gt(min_cut).sum() == 0:
        if labelled_output:
            perf = s.to_frame('final_score')
            perf['category'] = 'below'
            if verbose:
                print 'Warning: too few values above min_cut to determine meaningful cut scores.'
            return perf['category']
        else:
            return min_cut, (1.0 + min_cut) / 2.0, 1.0

    meeting_border = s[s.gt(min_cut)].mean()
    approaching_border = s.mean() if (s.gt(min_cut).mean() < 0.5) and adjust_min else min_cut
    exceeding_border = s[s.gt(meeting_border)].mean()

    # approaching_border = s[~subsetter].mean() if subsetter.mean() <= min_cut else min_cut
    # exceeding_border = meeting_border + interval if meeting_border + interval < 1.0 else 1.0

    if verbose:
        print 'Approaching:', '{:0.3f}'.format(approaching_border)
        print 'Meeting:', '{:0.3f}'.format(meeting_border)
        print 'Exceeding:', '{:0.3f}'.format(exceeding_border)

    if labelled_output:
        perf = s.to_frame('final_score')
        perf['category'] = 'below'
        perf.loc[perf['final_score'].gt(approaching_border), 'category'] = 'approaching'
        perf.loc[perf['final_score'].gt(meeting_border), 'category'] = 'meeting'
        perf.loc[perf['final_score'].ge(exceeding_border), 'category'] = 'exceeding'
        return perf['category']
    else:
        return approaching_border, meeting_border, exceeding_border


def compare_distributions(parent, group_levels, cutoff=None):

    p = parent.dropna().copy()
    if cutoff is not None:
        p = p[p >= cutoff]
    p = p.tolist()[:]

    def grouped_comparisons(child):
        child = child[:] if type(child) == list else child.tolist()[:]
        p_copy = p[:]
        for i in child:
            _ = p_copy.pop(p_copy.index(i))
        random.seed(42)
        v_dist = random.choice(p_copy, size=10000, replace=True)
        random.seed(42)
        v_comp = random.choice(child, size=10000, replace=True)
        return (v_dist <= v_comp).mean()

    output = parent.dropna().groupby(level=group_levels).apply(grouped_comparisons)
    return output


def compare_distribution(dist1, dist2, cutoff=None):

    random.seed(42)
    rand1 = random.choice(dist1.dropna().tolist()[:], size=10000, replace=True)
    random.seed(42)
    dist2 = dist2.dropna()
    if cutoff is not None:
        dist2 = dist2[dist2 >= cutoff]
    rand2 = random.choice(dist2.tolist()[:], size=10000, replace=True)
    return (rand1 >= rand2).mean()


def aggregate_indicators(df, index_cols, unstack_col, group_col, metric_col, label):
    df = df.copy()
    agg = DataFrame()
    aggw = DataFrame(index=df['indicator'].unique(), columns=df['grade'].unique())
    grp = df.set_index(index_cols + [unstack_col])[metric_col]\
        .unstack(unstack_col)\
        .groupby(level=group_col, group_keys=False)
    for grade, grp_df in grp:
        weights, grp_df = combine_subjects_by_rules(grp_df, constituent=int(grade))
        grp_df = grp_df.to_frame(metric_col).reset_index()
        grp_df[unstack_col] = label
        agg = agg.append(grp_df)
        aggw[grade] = weights
    agg = agg.reindex_axis(df.columns, axis=1)

    return agg, aggw.fillna(0.0) / aggw.sum()


def scaler(x, robust=True):
    if robust:
        y = x - x.median()
        return y / y.abs().median()
    else:
        y = x - x.mean()
        return y / y.std()
