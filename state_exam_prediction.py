import numpy as np
import pandas as pd
from utilities.state_prediction.load_dat import get_achievement_data, __main__ as load_main
from utilities.state_prediction.model import model_predict, model_data, report_error, calculate_error, \
    calculate_percent_incorrect, pred_ints
from sklearn.decomposition import PCA
from pandas import get_dummies
from numpy import intersect1d
import time
from utilities.state_prediction.cutoff_scores import cutoffs_dist_math_15 as cutoffs_dist_math
from utilities.state_prediction.cutoff_scores import min_scores_math_15 as min_scores_math
from utilities.state_prediction.cutoff_scores import max_scores_math_15 as max_scores_math
from utilities.state_prediction.cutoff_scores import cutoffs_dist_ela_15 as cutoffs_dist_ela
from utilities.state_prediction.cutoff_scores import min_scores_ela_15 as min_scores_ela
from utilities.state_prediction.cutoff_scores import max_scores_ela_15 as max_scores_ela
from dateutil.relativedelta import relativedelta

from utilities.state_prediction.sql_queries import achievement_weekly_query


# Pulled from leverageHM_functions #
def remove_nans(x):
    x[x.isnull()] = np.nanmean(x)
    check = x.isnull().values.sum()
    if check != 0:
        print "Warning: Still {num} nans present in inputs".format(num=check)
    return x


def parse_and_combine(state_subset, assess, verbose):
    # Break into subject categories #
    state_info = pd.DataFrame(columns=['scholar_id', 'subject', 'grade', 'numeric_score', 'level_achieved',
                                       'school_short', 'school_long', 'score', 'reference_date'])
    valid_subjects = np.intersect1d(np.unique(state_subset.subject), np.unique(assess.subject))
    for subject in valid_subjects:
        s_subject = state_subset.loc[state_subset.subject.eq(subject)]
        a_subject = assess.loc[assess.subject.eq(subject)].reset_index()
        # Per subject, break into grades available #
        for grade in np.unique(s_subject.grade):
            if verbose:
                print 'Looking at grade {g} {sub}'.format(g=grade, sub=subject)
            s_scholars = np.unique(s_subject.loc[s_subject.grade.eq(grade)].scholar_id)
            a_scholars = np.unique(a_subject.loc[a_subject.grade.eq(grade)].scholar_id)
            # Find overlap in students who have state test exams and assessment coverage #
            valid_scholars = np.intersect1d(s_scholars, a_scholars)
            if len(valid_scholars) == 0:
                print 'No common scholars found at grade {g} {sub}'.format(g=grade, sub=subject)
                continue
            s_tmp = s_subject.loc[s_subject.scholar_id.isin(valid_scholars) & s_subject.grade.eq(grade)]
            # Look into why there are duplicates! #
            s_tmp = s_tmp.drop_duplicates(subset='scholar_id')
            a_tmp = a_subject.loc[a_subject.scholar_id.isin(valid_scholars) & a_subject.grade.eq(grade)]
            data = pd.merge(a_tmp, s_tmp, on=['scholar_id', 'subject', 'grade', 'academic_year'], how='inner')
            # Push into dataframe for later use #
            state_info = state_info.append(data)
    return state_info


def reduce_students(x1, stu_missing, hold_stu, rate=0.9):
    # print 'Average missing student values: {val}'.format(val=np.mean(stu_missing))
    hold_stu = hold_stu.sort_values(by='percent', ascending=True)
    ind = np.round(len(stu_missing)*rate)
    # Keep top 90% of students, and all questions ##
    students = hold_stu['scholar_id'].iloc[:int(ind)]
    # Filter out students to keep ##
    x1 = x1.loc[x1.index.isin(students), :]
    return x1


def process_data_by_achievement(train, test, curr_year, prev_years, flag='random_forest', folds=10, threshold=0.25,
                                verbose=False, use_pca=False):
    inputs = list()
    outputs = list()
    results = pd.DataFrame(columns=['Grade', 'Subject', 'Errors_Regression', 'Errors_Classification', 'New_Dates'])
    grades_train = np.unique(train.grade)
    grades_test = np.unique(test.grade)
    grades_available = np.intersect1d(grades_train, grades_test)
    # Need this indexer for pulling max and min state test scores #
    g = 0
    exam_prob_df = pd.DataFrame(columns=['scholar_id', 'grade', 'subject', 'num_dates', 'confidence', 'level_achieved',
                                         'numeric_actual', 'numeric_min', 'numeric_max', 'p1', 'p2', 'p3', 'p4',
                                         'p1_adj', 'p2_adj', 'p3_adj', 'p4_adj', 'p1_class', 'p2_class', 'p3_class',
                                         'p4_class', 'conf_1', 'conf_2', 'conf_3', 'conf_4'])
    for grade in grades_available:
        # prev_flag decides whether to include more than 1 year previous data #
        prev_flag = True
        train_by_grade = train.loc[train.grade.eq(grade) & train.academic_year.eq(prev_years[0])]
        test_by_grade = test.loc[test.grade.eq(grade) & test.academic_year.eq(curr_year)]

        subjects_available = np.unique(train.subject)
        for subject in subjects_available:
            print 'Looking at grade {g} {sub}'.format(g=grade, sub=subject)
            min_math_score = min_scores_math[g]
            max_math_score = max_scores_math[g]
            min_lit_score = min_scores_ela[g]
            max_lit_score = max_scores_ela[g]
            if subject == 'Mathematics':
                min_score = min_scores_math[g]
                max_score = max_scores_math[g]
                cutoffs = (np.array(cutoffs_dist_math[g])-min_score)/(max_score-min_score)
            elif subject == 'Literacy':
                min_score = min_scores_math[g]
                max_score = max_scores_math[g]
                cutoffs = (np.array(cutoffs_dist_ela[g])-min_score)/(max_score-min_score)
            else:
                if verbose:
                    print 'No cutoff values for grade {g} {sub}. Skipping for now...'.format(g=grade, sub=subject)
                continue
            errors_looped = list()
            class_looped = list()
            # train_temp = train_by_grade.loc[train_by_grade.subject.eq(subject)]
            # Pull out subject-specific training data #
            train_lit = train_by_grade.loc[train_by_grade.subject.eq('Literacy')]
            train_math = train_by_grade.loc[train_by_grade.subject.eq('Mathematics')]
            # Drop any duplicates in each training set
            train_lit = train_lit.drop_duplicates(subset=['scholar_id', 'reference_date'])
            train_math = train_math.drop_duplicates(subset=['scholar_id', 'reference_date'])
            # Pivot into rectangular matrix to train the model #
            x_lit = train_lit[['scholar_id', 'reference_date', 'score']].pivot(
                index='scholar_id', columns='reference_date', values='score')
            x_math = train_math[['scholar_id', 'reference_date', 'score']].pivot(
                index='scholar_id', columns='reference_date', values='score')
            # subset to keep lengths consistent #
            x_math = x_math.loc[np.intersect1d(x_math.index, x_lit.index)]
            x_lit = x_lit.loc[np.intersect1d(x_math.index, x_lit.index)]
            x_scholars = x_lit.index

            # Confirm shape #
            if (x_lit.shape[0] == 0 | x_lit.shape[1] == 0) | x_lit.empty:
                if verbose:
                    print 'Grade {grade} {subject}: at least one dimension has no values. Skipping...'.format(
                        grade=grade, subject=subject)
                continue

            # Scramble relevant scholars #
            x_scholars = pd.Series(np.random.permutation(x_scholars))
            # Include only scholars shared across all sources #
            x_lit = x_lit.loc[x_scholars]
            x_math = x_math.loc[x_scholars]
            x_scholars = x_lit.index

            # Should be trivial - Check that lit and math scholars align #
            if not all(x_lit.index == x_math.index):
                if verbose:
                    print 'Not all lit indices match math indices. Skipping'
                continue

            # GENERATE TESTING SET HERE #
            test_math = test_by_grade.loc[test_by_grade.academic_year.eq(curr_year) & test_by_grade.subject.eq(
                'Mathematics')].drop_duplicates(subset=['scholar_id', 'reference_date']).sort_values(by='scholar_id')
            test_math = test_math[['scholar_id', 'reference_date', 'score']].pivot(
                index='scholar_id', columns='reference_date', values='score')
            test_lit = test_by_grade.loc[test_by_grade.academic_year.eq(curr_year) & test_by_grade.subject.eq(
                'Literacy')].drop_duplicates(subset=['scholar_id', 'reference_date']).sort_values(by='scholar_id')
            test_lit = test_lit[['scholar_id', 'reference_date', 'score']].pivot(
                index='scholar_id', columns='reference_date', values='score')
            test_math = test_math.loc[np.intersect1d(test_math.index, test_lit.index)]
            test_lit = test_lit.loc[np.intersect1d(test_math.index, test_lit.index)]

            # Subset and remove NaN's, if necessary #
            x_lit, continue_flag1 = remove_missing(x_lit, threshold, verbose, subject)
            x_math, continue_flag2 = remove_missing(x_math, threshold, verbose, subject)
            test_math, continue_flag3 = remove_missing(test_math, threshold, verbose, subject)
            test_lit, continue_flag4 = remove_missing(test_lit, threshold, verbose, subject)

            # No guarantee that the same number of indices will be removed from math or lit matrices.
            # This will confirm that the same scholars are used for math and lit, regardless of shape
            # Expects lit, math in proper order
            x_lit, x_math, break_flag1 = conform_inputs(x_lit, x_math, verbose)
            test_lit, test_math, break_flag2 = conform_inputs(test_lit, test_math, verbose)

            # Want the final list of scholars that will be used going forward#
            test_scholars = test_lit.index

            if any([break_flag1, break_flag2]):
                print 'WARNING!: Unable to conform lit and math matrices. Skipping grade {g} {sub}'.format(
                    g=grade, sub=subject)
                continue

            # In case remove_missing actually removed any students #
            if len(x_scholars) != len(x_lit.index):
                if verbose:
                    print 'Scholars were removed. Need to sub-sample training data...'
                x_scholars = np.intersect1d(x_lit.index, x_math.index)
                x_lit = x_lit.loc[x_scholars]
                x_math = x_math.loc[x_scholars]
                # Shouldn't happen ! #
                if not all(x_lit.index == x_math.index):
                    if verbose:
                        print 'Lit and Math training matrices not aligned. Skipping...'
                    continue

            if any([continue_flag1, continue_flag2, continue_flag3, continue_flag4]):
                print 'WARNING!: Grade {g} {sub} Missing more than 25% of values after processing.'.format(g=grade, sub=subject)

            # Check now for grade 3 and change prev_flag status (or not) #
            if prev_flag & (grade == 3):
                if verbose:
                    print 'Cannot include previous year for grade 3 data.'
                prev_flag = False

            if not prev_flag:
                if verbose:
                    print 'Will not include previous year data'
            else:
                # Find previous year's data for students in current testing set #
                prev_data = train.loc[train.academic_year.eq(prev_years[1]) & train.scholar_id.isin(test_scholars)]
                prev_levels = prev_data.loc[prev_data.subject.eq(subject)].\
                    drop_duplicates(subset=['scholar_id']).reset_index()
                prev_scores_lit = prev_data.loc[prev_data.subject.eq('Literacy')].\
                    drop_duplicates(subset=['scholar_id']).reset_index()
                prev_scholars_lit = prev_scores_lit.scholar_id
                prev_scores_math = prev_data.loc[prev_data.subject.eq('Mathematics')].\
                    drop_duplicates(subset=['scholar_id']).reset_index()
                prev_scholars_math = prev_scores_math.scholar_id

                # Filter out overlapping previous year's students #
                prev_scholars = np.intersect1d(prev_scholars_lit, prev_scholars_math)

                # Re-filter for scholars in case some were dropped #
                prev_scores_lit = prev_scores_lit.loc[prev_scores_lit.scholar_id.isin(test_scholars)].\
                    sort_values(by=['scholar_id'])
                prev_scores_math = prev_scores_math.loc[prev_scores_math.scholar_id.isin(test_scholars)].\
                    sort_values(by=['scholar_id'])

                # Filter out numeric scores #
                lit_scores = prev_scores_lit[['scholar_id', 'numeric_score']].reset_index()
                math_scores = prev_scores_math[['scholar_id', 'numeric_score']].reset_index()

                y_prev = pd.DataFrame()
                y_prev['scholar_id'] = lit_scores.scholar_id
                y_prev['numeric_score_lit'] = lit_scores.numeric_score
                y_prev['numeric_score_math'] = math_scores.numeric_score
                y_prev['level_achieved'] = prev_levels.level_achieved.loc[prev_levels.scholar_id.isin(prev_scholars)]

            # x_network = state[['scholar_id', 'reference_date', 'network_adjusted']].pivot(
            #    index='scholar_id', columns='reference_date', values='network_adjusted')

            # Outputs do not change - Don't need to calculate them each loop #
            n = np.round(0.75*x_lit.shape[0])
            train_subject = train_by_grade.loc[train_by_grade.subject.eq(subject)]
            y = train_subject[['scholar_id', 'level_achieved', 'numeric_score']].drop_duplicates(subset='scholar_id')
            y = y.loc[y.scholar_id.isin(x_scholars)].sort_values(by='scholar_id').reset_index()
            # Arrange y by sorted scholar id #
            y['scholar_id_cat'] = y.scholar_id.astype('category')
            y.scholar_id_cat.cat.set_categories(x_scholars, inplace=True)
            y = y.sort_values(by='scholar_id_cat')
            if not all(y.scholar_id == x_scholars):
                print 'WARNING!: Current input and output (numeric scores) scholar orders not aligned. ' \
                      'Skipping grade {g} {sub}'.format(g=grade, sub=subject)
                continue
            y['scaled'] = (y.numeric_score.copy()-min_score)/(max_score-min_score)
            y_train = y.loc[y.scholar_id.isin(x_scholars[:n])]
            y_test = y.loc[y.scholar_id.isin(x_scholars[n:])]

            # Generate initial probabilities of level achieved for training set:
            classes = np.unique(y_train.level_achieved)
            init_level_prob = np.zeros(len(classes))
            cnt = 0
            for lvl in classes:
                # Find fraction of each level that equals iter+1
                init_level_prob[cnt] = float(y_train.level_achieved.loc[y_train.level_achieved.eq(lvl)].shape[0])/\
                                       float(y_train.shape[0])
                cnt += 1
            # Convert to dict
            init_level_dict = dict(zip(*[classes, init_level_prob]))
            if verbose:
                print 'Available classes: {c}'.format(c=init_level_dict)

            if prev_flag:
                # y_prev_test = y_prev.loc[y_prev.scholar_id.isin(relevant_scholars[n:])]
                y_prev['scaled_lit'] = (y_prev.numeric_score_lit.copy()-min_lit_score)/(max_lit_score-min_math_score)
                y_prev['scaled_math'] = (y_prev.numeric_score_math.copy()-min_math_score)/\
                                        (max_math_score-min_math_score)

            [x, weeks_train] = generate_features(x_lit, x_math, np.unique(x_lit.columns), x_scholars)
            [test_set, weeks_test] = generate_features(test_lit, test_math, np.unique(test_lit.columns), test_lit.index)

            # Add in school variables #
            # train_sub = train_by_grade.loc[train_by_grade.scholar_id.isin(x_scholars)].\
            #     drop_duplicates(subset='scholar_id').sort(columns='scholar_id')
            # cnt = 0
            # for school in np.unique(train_by_grade.school_short):
            #     if cnt == 0:
            #         train_schools = pd.DataFrame(columns=[school], data=train_sub[school].reset_index())
            #     else:
            #         school_dat = pd.DataFrame(columns=[school], data=train_sub[school].reset_index())
            #         train_schools = train_schools.join(school_dat, how='left')
            #     cnt += 1
            # train_schools.index = x_scholars

            test_sub = test_by_grade.loc[test_by_grade.scholar_id.isin(test_scholars) & test_by_grade.subject.eq(subject)].drop_duplicates(subset='scholar_id').sort(columns='scholar_id').reset_index()
            # cnt = 0
            # for school in np.unique(test_sub.school_short):
            #     if cnt == 0:
            #         test_schools = pd.DataFrame(columns=[school], data=test_sub[school].reset_index())
            #     else:
            #         school_dat = pd.DataFrame(columns=[school], data=test_sub[school].reset_index())
            #         test_schools = test_schools.join(school_dat, how='left')
            #     cnt += 1
            # test_schools.index = test_scholars
            tmp_dates = np.sort(np.intersect1d(weeks_train, weeks_test))

            # To allow for calculating multiple dates, uncomment for loop below #
            # for i in range(0, len(tmp_dates)):
            #     if i == 0:
            #         print 'Using all dates'
            #     else:
            #         print 'Knocking off recent date {date}'.format(date=tmp_dates[-1])
            #         tmp_dates = tmp_dates[:-1]
            #     if (flag == 'LDA') & (len(tmp_dates) == 1):
            #         print 'LDA pseudoinverse won\'t work with too few features. Breaking'
            #         break
            x_tmp = x
            # keep_cols = np.hstack([tmp_dates, np.array(all_schools.columns)])
            x_tmp = x_tmp[tmp_dates]
            # x_tmp = x_tmp.join(train_schools, how='inner')
            # Add in previous year's performance #
            # if prev_flag:
            #     x_tmp = np.hstack([x_tmp, np.transpose(np.atleast_2d(y_prev.scaled_lit))])
            #     x_tmp = np.hstack([x_tmp, np.transpose(np.atleast_2d(y_prev.scaled_math))])
            x1 = x_tmp

            n_comp = 5
            if use_pca:
                if x1.shape[1] < n_comp:
                    if verbose:
                        print 'Not enough features to transform into PCA space. Breaking'
                    break
                pca = PCA(n_components=n_comp)
                x_pca = pca.fit_transform(x1)
                x1 = x_pca
                # Reserve 25% of data for EXTRA testing purposes #
                x_train = np.matrix(x_pca[:n, :])
                x_test = np.matrix(x_pca[n:, :])
                # x_test_scholars = relevant_scholars[n:]
            else:
                x_train = np.matrix(x1)[:n, :]
                x_test = np.matrix(x1)[n:, :]
                # x_test_scholars = relevant_scholars[n:]
            # run the model to predict scaled numeric values #
            calc = 'regression'
            [_, error, model_reg] = model_data(x_train, y_train.scaled, flag, folds, init_level_dict, calc, verbose)
            if verbose:
                print 'Unique dates included = {dates}, giving {n_feat} features before PCA'.format(
                    dates=len(tmp_dates), n_feat=x1.shape)
                report_error(subject, grade, flag, error)
            # Verify error #
            tmp_reg = model_predict(model_reg, x_test)
            y_hat_train = pd.DataFrame(data=tmp_reg, columns=['actual'])
            verify = calculate_error(np.ravel(y_hat_train), y_test.scaled)
            if verbose:
                print 'Verified regression error: {err}'.format(err=verify)
                print 'Predicting level probabilities...'
                print 'Classes to be passed in for classification: {l}'.format(l=np.unique(y_train.level_achieved))

            # Run classification to find probabilities of 1/2/3/4 for each student #
            calc = 'classification'
            # Want to provide initial probabilities for each class
            [_, _, model_class] = model_data(x_train, y_train.level_achieved, flag, folds, init_level_dict, calc)
            tmp_class = model_predict(model_class, x_test)
            y_hat_train_level = pd.DataFrame(data=tmp_class, columns=['actual'])
            if verbose:
                print 'Percent Incorrect using gradient-boosted classifier: {val}'.\
                    format(val=calculate_percent_incorrect(y_hat_train_level.actual, y_test.level_achieved))

            # PREDICTIONS FOR APP HERE #

            # Keep relevant dates for test set #
            test_use = test_set[tmp_dates]

            # In case something happened to the scholar order #
            if not all(test_use.index == test_scholars):
                if verbose:
                    print 'Test scholars array and testing set index not aligned. Will overwrite with testing set index'
                test_scholars = test_use.index

            # Use trained classifier to return probabilities of level achieved given testing set data #
            if verbose:
                print 'Calculating probabilities of level achieved for test set...'
            all_levels = [1, 2, 3, 4]
            predicted_levels = np.unique(model_class.predict(test_use))
            missing_level = [lvl for lvl in all_levels if lvl not in predicted_levels]
            level_probs = model_class.predict_proba(test_use)
            # If predictions do not return a particular level, predict_proba will not return values for that level.
            # This adds zeros for any missing level columns
            level_probs_df = pd.DataFrame()
            adj = 0
            for lvl in all_levels:
                if (lvl in missing_level) & (level_probs.shape[1] < 4):
                    add_probs = [0] * level_probs.shape[0]
                    level_probs_df[str(lvl)] = add_probs
                    adj += 1
                else:
                    level_probs_df[str(lvl)] = level_probs[:, (lvl-1)-adj]

            # Run predictions of numeric score on testing set #
            tmp_reg = model_predict(model_reg, test_use)
            err_down, err_up = pred_ints(model_reg, test_use, p=95)
            # tmp_class = model_predict(model_class, x1)
            y_hat = pd.DataFrame(data=tmp_reg, columns=['actual'])
            y_hat['max_vals'] = err_up
            y_hat['min_vals'] = err_down
            # Classify based on cutoff score #
            estimated_level = pd.DataFrame()
            est = y_hat.actual.copy()
            est[y_hat.actual < cutoffs[0]] = 1
            est[(y_hat.actual > cutoffs[0]) & (y_hat.actual < cutoffs[1])] = 2
            est[(y_hat.actual > cutoffs[1]) & (y_hat.actual < cutoffs[2])] = 3
            est[y_hat.actual > cutoffs[2]] = 4
            estimated_level['actual'] = est
            # Run classification for max values #
            est = y_hat.max_vals.copy()
            est[y_hat.max_vals < cutoffs[0]] = 1
            est[(y_hat.max_vals > cutoffs[0]) & (y_hat.max_vals < cutoffs[1])] = 2
            est[(y_hat.max_vals > cutoffs[1]) & (y_hat.max_vals < cutoffs[2])] = 3
            est[y_hat.max_vals > cutoffs[2]] = 4
            estimated_level['max_vals'] = est
            # Run classification for min values #
            est = y_hat.min_vals.copy()
            est[y_hat.min_vals < cutoffs[0]] = 1
            est[(y_hat.min_vals > cutoffs[0]) & (y_hat.min_vals < cutoffs[1])] = 2
            est[(y_hat.min_vals > cutoffs[1]) & (y_hat.min_vals < cutoffs[2])] = 3
            est[y_hat.min_vals > cutoffs[2]] = 4
            estimated_level['min_vals'] = est

            # Generate empirical probability profiles #
            actual_4_percent = np.zeros(4)
            est_4_percent = np.zeros((estimated_level.shape[1], 4))
            for a in range(0, estimated_level.shape[1]):
                for b in range(0, 4):
                    est_4_percent[a, b] = np.sum(estimated_level.iloc[:, a] == (b+1))/\
                                          float(estimated_level.shape[0])
                    if a == 0:
                        actual_4_percent[b] = np.sum(y.level_achieved.iloc[:] == (b+1))/\
                                                 float(y.level_achieved.shape[0])
            est_4_percent = pd.DataFrame(data=est_4_percent, columns=['1', '2', '3', '4'],
                                         index=estimated_level.columns)
            est_4_per_mean = est_4_percent.mean(axis=0)
            conf = 1.0-np.abs(est_4_per_mean - actual_4_percent)
            confidence = 1.0 - np.sum(np.abs(est_4_per_mean - actual_4_percent))
            # Calculate incorrect classification #
            incorrect_class = calculate_percent_incorrect(y_hat_train_level.actual, y_test.level_achieved)
            if verbose:
                print 'Correct classification using calculated numeric scores: {err}'.format(err=1.0-incorrect_class)

            # Calculate probability distribution for each scholar #
            prob = np.zeros((estimated_level.shape[0], 4))
            prob_adjusted = np.zeros((estimated_level.shape[0], 4))
            for a in range(0, estimated_level.shape[0]):
                for b in range(0, 4):
                    prob[a, b] = np.sum(estimated_level.iloc[a, :] == (b+1))/\
                                 float(estimated_level.shape[1])*confidence
                prob_adjusted[a, :] = prob[a, :]*conf
            prob = pd.DataFrame(data=prob, columns=['p1', 'p2', 'p3', 'p4'])
            prob['p1_adj'] = prob_adjusted[:, 0]
            prob['p2_adj'] = prob_adjusted[:, 1]
            prob['p3_adj'] = prob_adjusted[:, 2]
            prob['p4_adj'] = prob_adjusted[:, 3]
            prob['p1_class'] = level_probs_df['1']
            prob['p2_class'] = level_probs_df['2']
            prob['p3_class'] = level_probs_df['3']
            prob['p4_class'] = level_probs_df['4']
            if prev_flag:
                # There will likely be missing values from last year - Any missing values will be zero #
                vals = list()
                for stu in test_scholars:
                    val = y_prev.level_achieved.loc[y_prev.scholar_id == stu].values
                    if not any(val):
                        val = 0
                    else:
                        if np.isnan(val[0]):
                            val = 0
                        else:
                            val = val[0]
                    vals.append(val)
                prob['level_achieved'] = vals
            else:
                if verbose:
                    print 'No data for previous year. Adding zeros to previous level achieved'
                prob['level_achieved'] = [0] * len(y_hat.actual.values)
            prob['numeric_actual'] = y_hat.actual.values
            prob['numeric_min'] = y_hat.min_vals.values
            prob['numeric_max'] = y_hat.max_vals.values
            prob['confidence'] = confidence
            prob['conf_1'] = conf[0]
            prob['conf_2'] = conf[1]
            prob['conf_3'] = conf[2]
            prob['conf_4'] = conf[3]
            prob['scholar_id'] = test_scholars
            prob['grade'] = test_sub.grade
            prob['subject'] = subject
            prob['num_dates'] = str(len(tmp_dates))

            exam_prob_df = exam_prob_df.append(prob, ignore_index=True)
            # Taking average of model error and verified external error #
            mean_error = (verify+error)*0.5
            errors_looped.append(mean_error)
            class_looped.append(incorrect_class)

            add_me = {'Grade': grade, 'Subject': subject, 'Errors_Regression': errors_looped,
                      'Errors_Classification': class_looped, 'New_Dates': tmp_dates}
            results = results.append(add_me, ignore_index=True)
        g += 1
    if verbose:
        print 'Done with processing'
    return inputs, outputs, results, exam_prob_df


def add_features(state, verbose):
    if verbose:
        print 'Calculating median-adjusted achievement indices...'
    add_me = pd.DataFrame(columns=['scholar_id', 'grade', 'reference_date', 'subject', 'network_adjusted',
                                   'school_adjusted'])
    for grade in np.unique(state.grade):
        by_grade = state.loc[state.grade == grade]
        for subject in np.unique(by_grade.subject):
            by_subject = by_grade.loc[by_grade.subject == subject]
            for date in np.unique(state.reference_date):
                subset = by_subject.loc[by_subject.reference_date.eq(date)].drop_duplicates(subset='scholar_id').\
                    sort_values(by='scholar_id').reset_index()
                # network_median = np.median(subset.score)
                # tmp_network = subset.score - network_median
                tmp_network = [0]*len(subset.score)
                scholars = subset.scholar_id
                network_df = pd.DataFrame(data={'scholar_id': scholars, 'network_adjusted': tmp_network}).reset_index()
                school_df = pd.DataFrame(columns=['scholar_id', 'school_adjusted', 'school'])
                school_dummies = get_dummies(subset.school_short)
                school_dummies['scholar_id'] = scholars
                for school in np.unique(subset.school_short):
                    school_vals = subset.score.loc[subset.school_short.eq(school)]
                    # school_median = np.median(school_vals)
                    # tmp_school = school_vals - school_median
                    tmp_school = [0]*len(school_vals)
                    tmp_scholars = subset.scholar_id.loc[subset.school_short.eq(school)]
                    tmp = pd.DataFrame(data={'scholar_id': tmp_scholars, 'school_adjusted': tmp_school,
                                             'school': [school]*len(tmp_scholars)})
                    school_df = school_df.append(tmp, ignore_index=True)
                if verbose:
                    print '{sch} grade {g} {sub} {date}'.format(sch=school, g=grade, sub=subject, date=date)
                # Sort by scholar ID so that there are no problematic overlaps #
                school_df = school_df.sort_values(by='scholar_id').reset_index()
                temp = pd.DataFrame(data={'scholar_id': scholars, 'grade': subset.grade,
                                          'reference_date': subset.reference_date, 'subject': subset.subject,
                                          'network_adjusted': network_df.network_adjusted, 'school': school_df.school,
                                          'school_adjusted': school_df.school_adjusted})
                temp = pd.merge(temp, school_dummies, on='scholar_id', how='right')
                add_me = add_me.append(temp, ignore_index=True)
                # print 'Total null values so far: {n}'.format(n=add_me.isnull().values.sum())
    if verbose:
        print 'Merging into state_info...'
        print 'Replacing {n} null values with zero'.format(n=add_me.isnull().values.sum())
    add_me = add_me.fillna(0)
    if verbose:
        print 'Number null values after replacement: {n}'.format(n=add_me.isnull().values.sum())
    state = pd.merge(state, add_me, on=['scholar_id', 'grade', 'subject', 'reference_date'])
    return state


def remove_missing(x, threshold, verbose, subject):
    continue_flag = False
    missing = float(x.isnull().values.sum())/float(x.shape[0]*x.shape[1])
    if verbose:
        print 'Before removal, missing input values ({sub}): {miss}, out of {stu} students and {d} dates'.format(
            sub=subject, miss=missing, stu=x.shape[0], d=x.shape[1])
    if missing >= threshold:
        if verbose:
            print 'Attempting to Subsample training data...'
    while missing >= threshold:
        if verbose:
            print 'Current shape: {shape}'.format(shape=x.shape)
        hold_stu = pd.DataFrame()
        stu_missing = np.atleast_1d(x.isnull().values.sum(axis=1)/float(x.shape[1]))
        hold_stu['percent'] = stu_missing
        hold_stu['scholar_id'] = x.index
        if len(stu_missing) < 100:
            if verbose:
                print 'Do not want to reduce further. Breaking loop.'
            break
        else:
            x = reduce_students(x, stu_missing, hold_stu)
        missing = float(x.isnull().values.sum())/float(x.shape[0]*x.shape[1])
        if verbose:
            print 'New missing ({sub}): {miss}'.format(sub=subject, miss=missing)
    if verbose:
        print 'NEW percent missing input values ({sub}): {miss}, out of {stu} students and {d} dates'.format(
            sub=subject, miss=missing, stu=x.shape[0], d=x.shape[1])
    if missing > 0.25:
        continue_flag = True
    x = remove_nans(x)
    return x, continue_flag


def generate_features(x_lit, x_math, dates, x_scholars):
    vel_lit = np.diff(x_lit[dates])
    vel_lit = np.hstack([np.zeros((vel_lit.shape[0], 1)), vel_lit])
    vel_math = np.diff(x_math[dates])
    vel_math = np.hstack([np.zeros((vel_math.shape[0], 1)), vel_math])
    top_level = ['Lit Position', 'Lit Velocity', 'Math Position', 'Math Velocity']

    # Replace dates with 'Week x' so that the features can be applied across years #
    weeks = []
    for d in range(0, len(dates)):
        week = 'Week ' + str(d)
        weeks.append(week)
    # date_labels = np.tile(dates, len(top_level))
    date_labels = np.tile(weeks, len(top_level))
    name_labels = np.tile(top_level, len(dates))
    labels = [date_labels, name_labels]
    tuples = list(zip(*labels))
    cols = pd.MultiIndex.from_tuples(tuples)
    x = pd.DataFrame(data=np.hstack([x_lit, vel_lit, x_math, vel_math]), columns=cols)
    x.index = x_scholars
    return x, weeks


def conform_inputs(lit, math, verbose):
    if lit.shape[1] != math.shape[1]:
        print 'Shouldnt get here!!'
        print 'Number of features in lit ({n1}) different from math ({n2})'.format(n1=lit.shape[1], n2=math.shape[1])
        return lit, math, True

    if lit.shape[0] != math.shape[0]:
        schol = intersect1d(lit.index.values, math.index.values)
        lit_out = lit.loc[schol]
        math_out = math.loc[schol]
        if verbose:
            print 'Number of scholars in lit ({n1}) different from math ({n2})'.format(n1=lit.shape[0], n2=math.shape[0])
            print 'Conforming to minimum number of scholars {n}'.format(n=len(schol))
        return lit_out, math_out, False
    else:
        return lit, math, False


def __main__(conn, scholars_df, today, verbose=False):
    t = time.time()
    # NOTE!!! As of 9/5/15, assessment data is actually WEEKLY ACADEMIC ACHIEVEMENT SCORES!!! Not assessment scores
    # Coerce all scholar IDs to be ints so there will be no matching problems down the line #
    one_year = relativedelta(years=1)
    prev_year1 = today-one_year
    prev_year2 = prev_year1-one_year
    # NEED TO AUTOMATE THIS!!!
    current_year = '2015-2016'
    previous_year1 = '2014-2015'
    previous_year2 = '2013-2014'

    load_t = time.time()
    query = achievement_weekly_query.format(end_date=today)
    achieve1 = get_achievement_data(conn, query, verbose)
    query = achievement_weekly_query.format(end_date=prev_year1)
    achieve2 = get_achievement_data(conn, query, verbose)
    query = achievement_weekly_query.format(end_date=prev_year2)
    achieve3 = get_achievement_data(conn, query, verbose)
    if verbose:
        print 'took {t} seconds to load'.format(t=time.time() - load_t)

    assess = achieve1.append(achieve2.append(achieve3))
    [state, school_key] = load_main(conn)

    # Temporary cleaning step for any scholar(s) missing scholar_ids
    if state.scholar_id.isnull().sum() > 0:
        print 'WARNING!! Found missing scholar_ids in state_exam_results table.'
        state = state.loc[~state.scholar_id.isnull()]

    state.scholar_id = [int(x) for x in state.scholar_id]
    assess.scholar_id = [int(x) for x in assess.scholar_id]
    school_key.scholar_id = [int(x) for x in school_key.scholar_id]
    # subset state info #
    prev_years = [previous_year1, previous_year2]
    relevant_years = [current_year, previous_year1, previous_year2]

    # Generate training set #
    state_subset = state.loc[state.description.isin(prev_years)]
    state_subset = state_subset[['scholar_id', 'current_grade', 'numeric_score', 'level_achieved', 'assessment_type',
                                 'description']]
    state_subset = state_subset.rename(columns={'current_grade': 'grade', 'assessment_type': 'subject',
                                                'description': 'academic_year'})
    state_subset.subject = state_subset.subject.replace(to_replace=['ELA', 'Math'], value=['Literacy', 'Mathematics'])
    prev_state = state_subset.loc[state_subset.academic_year.isin(prev_years)]

    assess = assess[np.isfinite(assess.scholar_id)]
    assess_subset = assess.loc[(assess.academic_year.isin(relevant_years))]
    assess_subset = assess_subset[['subject', 'grade', 'scholar_id', 'academic_year', 'reference_date', 'score']]
    # prev_assess = assess_subset.loc[assess_subset.academic_year.eq(previous_year1)]
    if verbose:
        print 'Finished loading data'

    school_key = school_key.rename(columns={'description': 'academic_year'})
    school_key = school_key.loc[(school_key.academic_year.isin(relevant_years))]
    assess_subset = pd.merge(assess_subset, school_key, on=['scholar_id', 'academic_year'])
    prev_state_info = parse_and_combine(prev_state, assess_subset, verbose)
    # prev_state_info = add_features(prev_state_info, verbose)
    train = prev_state_info
    relevant_subjects = np.unique(train.subject)

    curr_assess = assess_subset.loc[assess_subset.academic_year.eq(current_year) & (assess_subset.reference_date <= today)]
    curr_assess = curr_assess.loc[curr_assess.subject.isin(relevant_subjects)]

    # Try this. May not be the best way to ensure grade integrity #
    curr_assess = curr_assess.drop(['grade', 'school_long'], 1)
    curr_assess = pd.merge(scholars_df[['scholar_id', 'grade']].drop_duplicates(subset='scholar_id'), curr_assess, on='scholar_id')
    curr_assess = pd.merge(curr_assess, school_key[['scholar_id', 'school_long', 'academic_year']], on=['scholar_id', 'academic_year'])
    curr_assess = curr_assess.rename(columns={'school_name': 'school_long'})
    # curr_schools = scholars_df[['scholar_id', 'school']].loc[scholars_df.scholar_id.isin(np.unique(curr_assess.scholar_id))].drop_duplicates(subset=['scholar_id'])
    # curr_schools = curr_schools.rename(columns={'school': 'school_short'})
    # curr_data = pd.merge(curr_assess, curr_schools, on='scholar_id')
    # curr_test = add_features(curr_assess, verbose)
    test = curr_assess

    # curr_state = state_info.loc[state_info.academic_year.eq(current_year)]
    flag = 'random_forest'

    # Run regression #
    threshold = 0.05
    folds = 5
    [_, _, _, prob_df] = process_data_by_achievement(train, test, current_year, prev_years, flag, folds,
                                                       threshold, verbose, use_pca=False)
    if verbose:
        print 'Processing took {t} seconds'.format(t=time.time()-t)

    # tmp_err = list()
    # legend_entries = list()
    # for i in range(0, err.shape[0]):
    #     errors = err.iloc[i].Errors_Regression
    #     tmp_err.append(np.mean(np.ravel(errors)))
    #     l_string = 'Grade {g} {sub}'.format(g=err.iloc[i].Grade, sub=err.iloc[i].Subject)
    #     legend_entries.append(l_string)
    #     print 'Average error: {avg}'.format(avg=np.mean(np.ravel(errors)))
    #     plt.figure(1)
    #     plt.plot(errors)
    #     plt.show()
    # plt.title('Mean Absolute Error Across Number Dates Removed - Dummy Variables Added')
    # plt.legend(legend_entries, loc='upper left')
    # print 'Avg average error {err}'.format(err=np.mean(tmp_err))
    #
    # tmp_err = list()
    # legend_entries = list()
    # for i in range(0, err.shape[0]):
    #     errors = err.iloc[i].Errors_Classification
    #     tmp_err.append(np.mean(np.ravel(errors)))
    #     l_string = 'Grade {g} {sub}'.format(g=err.iloc[i].Grade, sub=err.iloc[i].Subject)
    #     legend_entries.append(l_string)
    #     print 'Average error: {avg}'.format(avg=np.mean(np.ravel(errors)))
    #     plt.figure(2)
    #     plt.plot(errors)
    #     plt.show()
    # plt.title('Mean Classification Percent Incorrect Across Number Dates Removed - Dummy Variables Added')
    # plt.legend(legend_entries, loc='upper left')
    # print 'Avg classification error {err}'.format(err=np.mean(tmp_err))
    if verbose:
        print 'Finished'

    return prob_df
