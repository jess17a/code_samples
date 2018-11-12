from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
import random
import pandas as pd
from numpy import ravel, matrix, unique, round, sum, abs, percentile, corrcoef


def model_data(x, y, flag, folds, probs=[], calc='regression', verbose=False):
    if flag == 'random_forest':
        # Set number of estimators #
        n_est = 200
        # return forest_cross_validated(X,y,folds,n_est)
        return run_forest(x, y, probs, n_est, calc, verbose)
    else:
        print "No model chosen; returning empty array"
        return []


def forest_model_regressor(x1, x2, y1, est=100):
    rf = RandomForestRegressor(n_estimators=est)
    rf.fit(x1, y1)
    y_hat = rf.predict(x2)
    return rf.feature_importances_, y_hat, rf


def forest_model_classifier(x1, x2, y1, probs, est=100, verbose=False):
    rf = RandomForestClassifier(n_estimators=est, class_weight=probs)
    if verbose:
        print 'Forest classifier sees the following classes: {c}'.format(c=unique(y1))
    rf.fit(x1, y1)
    y_hat = rf.predict(x2)
    return rf.feature_importances_, y_hat, rf


def run_forest(x, y, probs, n_est=100, calc='regression', verbose=False):
    n = int(round(x.shape[0]*0.60))
    x1 = x[n:, :]
    x2 = x[:n, :]
    y1 = y[n:]
    y2 = y[:n]
    if calc == 'regression':
        [weights, y_hat, model] = forest_model_regressor(x1, x2, ravel(y1).reshape(-1, 1), n_est)
        error = calculate_error(y2, y_hat)
    elif calc == 'classification':
        num_class = len(unique(y))
        x = pd.DataFrame(data=x)
        y = pd.DataFrame(data=y)
        if verbose:
            print 'classes present in y1: {c}'.format(c=unique(y1))
        cnt = 0
        # NOTE: This is a temporary cheat. Want to replace this soon. #
        if not len(unique(y1)) == num_class:
            if verbose:
                print 'Not all classes present in input.'
            flag = False
            while not flag:
                if verbose:
                    print 'Resampling (loop {cnt})...'.format(cnt=cnt)
                train_ind = random.sample(xrange(x.shape[0]), int(round(0.6*x.shape[0])))
                test_ind = [a for a in range(0, x.shape[0]-1) if a not in train_ind]
                x1 = matrix(x.iloc[train_ind, :])
                x2 = matrix(x.iloc[test_ind, :])
                # Passing in 1D array is deprecated. reshape to fix that
                y1 = ravel(y.iloc[train_ind])
                y2 = ravel(y.iloc[test_ind])
                cnt += 1
                if len(unique(y1)) == num_class:
                    flag = True
        [weights, y_hat, model] = forest_model_classifier(x1, x2, ravel(y1).reshape(-1, 1), probs, n_est, verbose)
        error = calculate_percent_incorrect(y2, y_hat)
    if verbose:
        print 'Finished training model'
    return weights, error, model


def model_predict(model, x):
    y_hat = model.predict(x)
    return y_hat


def model_predict_modulated(x, y, weights, n_est=100):
    rf = RandomForestRegressor(n_estimators=n_est)
    rf = rf.fit(x, y, sample_weight=weights)
    y_hat = rf.predict(x)
    return rf, y_hat


def calculate_error(y, y_hat):
    err = sum(abs(y-y_hat))/float(len(y))
    return err


def calculate_percent_incorrect(y, y_hat):
    err = sum(ravel(y) != y_hat)/float(len(y))
    # print 'Percent incorrect classification: {err}'.format(err=err)
    return err


def report_error(subj, grade, flag, err):
        print "Average absolute error for subject {subj} in grade {grade} using {flag} model: {err}".format(
            subj=subj, grade=grade, flag=flag, err=err)


def calculate_confidence(y_actual, y_estimate):
    eps = calculate_error(y_actual, y_estimate)
    corr = corrcoef(y_actual, y_estimate)[0][1]  # Take cross-correlation term
    confidence = (1-eps)*corr*100
    return confidence


# Taken from http://blog.datadive.net/prediction-intervals-for-random-forests/
def pred_ints(model, X, p=95):
    # print 'generating percentiles...'
    err_down = []
    err_up = []
    for x in range(len(X)):
        preds = []
        for pred in model.estimators_:
            preds.append(pred.predict(X.iloc[x, :].reshape(1, -1))[0])
        err_down.append(percentile(preds, (100 - p) / 2.0))
        err_up.append(percentile(preds, 100 - (100 - p) / 2.0))
    return err_down, err_up
