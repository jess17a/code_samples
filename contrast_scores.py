import os
import io
from pandas import DataFrame, read_sql

from dma.database.cursors import yield_cursor

#: SQL query files directory
ABS_PATH = os.path.abspath(os.path.dirname(__file__))


def contrast_scores(conn, host_name, env_prefix):
    """
    Contrast scores are question level discrimination indices - they run from 0.0 to 1.0, and indicate how well a
    question differentiates students who have historically tended to perform well in a subject from students who
    have historically performed poorly. They are calculated by ranking scholars in order from best historical
    performance to worst (using a weighted average over the previous 6 months of assessments), and then splitting those
    rankings into a top partition (top 25%) and bottom partition (bottom 25%). The percentage of scholars in the bottom
    partition who got the question right is subtracted from the percentage of scholars in the bottom partition who
    got the question right. So if 100% in the top partition get it right and 0% in the bottom partition get it right,
    the contrast score is 1.0. If the contrast score is negative, it is truncated to 0.0.

    Fields in this table include

        subject_id bigint, the SMS SubjectID
        assessment_id bigint NOT NULL, the SMS AssessmentID
        assessment_question_id bigint NOT NULL, the SMS AssessmentQuestionID
        contrast double precision, the contrast score for the question
        contrast_weighted double precision, the constrast score, where each value in the high and lower partitions are
            weighted by how extreme they are (more extreme get more weight)
        efficiency double precision, see http://journals.sagepub.com/doi/abs/10.1177/001316447503500311
        coverage double precision, the percentage of scholars who took the assessment who answered the question
        mean_percent_correct double precision, the average percent correct for the question
        success boolean, a flag indicating whether the calculations ran with out errors

    :param conn: a sqlalchemy connection engine
    :return: None
    """

    with yield_cursor(db_type="PG", host_name=host_name, env_prefix=env_prefix) as cur:
        cur.execute('TRUNCATE TABLE magnus.contrast_scores;')

    path = os.path.join(ABS_PATH, "sql/assessment_history.sql")
    with open(path, "r") as query:
        sa = read_sql(sql=query.read(), con=conn).set_index('assessment_id')

    all_ids = sa.index.drop_duplicates().tolist()
    all_outputs = DataFrame()
    for a_id in all_ids:
        n_scholars = sa['scholar_count'].xs(a_id)
        past_assessment_ids = ', '.join([str(aid) for aid in sa['past_assessment_ids'].xs(a_id)])
        scholar_ids = ', '.join([str(aid) for aid in sa['scholar_ids'].xs(a_id)])

        path = os.path.join(ABS_PATH, "sql/assessment_and_historicals.sql")
        with open(path, "r") as query:
            query = query.read().format(past_assessment_ids=past_assessment_ids,
                                        scholar_ids=scholar_ids)

        df = read_sql(sql=query, con=conn).\
            set_index(['subject_id', 'assessment_id', 'assessment_question_id', 'scholar_id'])['percent_correct'].\
            unstack(['subject_id', 'assessment_id', 'assessment_question_id'])

        # saftey check to make sure only includes scholars who took target assessment and > 50% of historical assessment
        # and only includes questions that had over 50% coverage among target scholars
        try:
            filt = df.xs(a_id, axis=1, level='assessment_id').notnull().any(axis=1)
        except KeyError:
            filt = df.iloc[:, 0].apply(lambda t: True)
        df = df.loc[filt, :]
        df = df.loc[:, df.isnull().mean().lt(0.5)]
        df = df.loc[df.isnull().mean(axis=1).lt(0.5), :]

        use_cols = df.xs(a_id, level='assessment_id', axis=1, drop_level=False).columns
        if (use_cols.shape[0] == df.shape[1]):
            df_history = df.copy()
        else:
            df_history = df.drop(use_cols, axis=1)

        output_cols = ['contrast', 'contrast_weighted', 'efficiency',
                       'coverage', 'mean_percent_correct', 'success']
        output = DataFrame(index=use_cols, columns=output_cols)

        for col in use_cols:
            success = True
            total_scholars = df[col].notnull().sum()
            count_weight = total_scholars / \
                float(n_scholars) if n_scholars >= total_scholars else 1.0

            if df[col].std() == 0.0:
                output.loc[col, 'contrast'] = 0.0
                output.loc[col, 'contrast_weighted'] = 0.0
                output.loc[col, 'efficiency'] = 0.0
                output.loc[col, 'coverage'] = count_weight
                output.loc[col, 'mean_percent_correct'] = df[col].mean()
                output.loc[col, 'success'] = success
                continue

            corrs = df_history.apply(lambda x: df[col].corr(x, method='pearson'))
            corrs = corrs.where(corrs.gt(0.0)).fillna(0.0)
            col_mean = 0.0 if df[col].mean() < 0.0 else df[col].mean()
            diffs = (1.0 - df_history.apply(lambda x: x.mean() - col_mean).abs()) ** 6.6439
            weights = corrs * diffs
            avg = df_history.loc[:, weights.index].mul(weights).sum(axis=1).div(weights.sum())

            if (avg.notnull().sum() == 0.0) or avg.quantile([0.25, 0.75]).eq(0).all():
                output.loc[col, 'contrast'] = 0.0
                output.loc[col, 'contrast_weighted'] = 0.0
                output.loc[col, 'efficiency'] = 0.0
                output.loc[col, 'coverage'] = count_weight
                output.loc[col, 'mean_percent_correct'] = df[col].mean()
                output.loc[col, 'success'] = success
                continue

            try:
                q1 = avg.quantile(0.25)
                q3 = avg.quantile(0.75)
                q_cuts = avg.apply(lambda x: 1).copy()
                q_cuts[avg.le(q1)] = 0
                q_cuts[avg.ge(q3)] = 2
                high = df[col][q_cuts.eq(2)]
                low = df[col][q_cuts.eq(0)]
                high_w = avg[q_cuts.eq(2)]
                low_w = avg[q_cuts.eq(0)]
                high_w /= high_w.max()
                low_w = (1 - low_w) / (1 - low_w).max()

                output_mean = high.mean() - low.mean() if high.mean() >= low.mean() else 0.0
                high_w_mean = (high * high_w).sum() / high_w.sum()
                low_w_mean = (low * low_w).sum() / low_w.sum()
                output_w_mean = high_w_mean - low_w_mean if high_w_mean >= low_w_mean else 0.0
                max_correct = high.mean() + low.mean()
                denom = 2.0 - max_correct if max_correct > 1.0 else max_correct
                output_effic = output_mean / denom if denom > 0.0 else 0.0
            except Exception:
                output_mean = 0.0
                output_w_mean = 0.0
                output_effic = 0.0
                success = False

            output.loc[col, 'contrast'] = output_mean
            output.loc[col, 'contrast_weighted'] = output_w_mean
            output.loc[col, 'efficiency'] = output_effic
            output.loc[col, 'coverage'] = count_weight
            output.loc[col, 'mean_percent_correct'] = df[col].mean()
            output.loc[col, 'success'] = success

        output = output.astype('float')
        output['success'] = output['success'].astype('bool')
        all_outputs = all_outputs.append(output.copy(), ignore_index=False)

    all_outputs = all_outputs.reset_index()

    delete_statement = '''
        DELETE FROM magnus.contrast_scores
        WHERE assessment_id = {aid}
            AND assessment_question_id = {aqid}
    '''

    with yield_cursor(db_type="PG", host_name=host_name, env_prefix=env_prefix) as cur:
        for ind, row in all_outputs.iterrows():
            cur.execute(delete_statement.format(aid=row.assessment_id,
                                                aqid=row.assessment_question_id))

        # Send data to database
        buffer = io.StringIO()
        all_outputs.to_csv(buffer, sep="\t", header=False, index=False, encoding="utf-8")
        buffer.seek(0)
        cur.copy_from(file=buffer, table="magnus.contrast_scores",
                      sep="\t", null="", size=32768)
        buffer.close()
