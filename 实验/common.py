"""Train frozen pipelines for a within-dataset classification study."""

import hashlib
import json
import platform
from pathlib import Path
import time
from importlib.metadata import version

import joblib
import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.base import clone
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data' / 'seismic-bumps.arff'
RESULTS = ROOT / 'results'
MODELS = ROOT / 'models'
CHECKS = ROOT / 'checks'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_plan():
    return json.loads((ROOT / 'protocol.json').read_text(encoding='utf-8'))


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def save_csv(name, data):
    data.to_csv(RESULTS / name, index=False, encoding='utf-8-sig', float_format='%.17g')


def read_data(plan):
    if digest(DATA) != plan['arff_sha256']:
        raise ValueError('ARFF differs from the fixed source')
    rows, attributes = arff.loadarff(DATA)
    table = pd.DataFrame(rows)
    categories = [name for name in attributes.names() if attributes[name][0] == 'nominal']
    for name in categories:
        table[name] = table[name].str.decode('ascii')
    y = table.pop('class').astype(int).to_numpy()
    categories.remove('class')
    numbers = [name for name in table.columns if name not in categories]
    if table.shape != (2584, 18) or set(y) != {0, 1} or y.sum() != 170:
        raise ValueError('Unexpected official source shape or labels')
    if table.isna().any().any() or not np.isfinite(table[numbers]).all().all():
        raise ValueError('Check missing or non-finite data before fitting')
    for name in categories:
        if not set(table[name]).issubset(attributes[name][1]):
            raise ValueError(f'Unexpected category in {name}')
    groups = table.groupby(list(table.columns), sort=False, dropna=False).ngroup().to_numpy()
    return table, y, groups, categories, numbers


def split_folds(x, y, groups, settings):
    splitter = StratifiedGroupKFold(n_splits=settings['folds'], shuffle=True, random_state=settings['seed'])
    folds = list(splitter.split(x, y, groups))
    for training, validation in folds:
        if len(np.intersect1d(groups[training], groups[validation])):
            raise ValueError('An identical-input group crossed a split')
        if len(np.unique(y[training])) != 2 or len(np.unique(y[validation])) != 2:
            raise ValueError('A split lacks one class')
    return folds


def pipeline(name, categories, numbers, plan):
    seed = plan['model_seed']
    if name == 'Dummy':
        return DummyClassifier(strategy='prior')
    numeric = StandardScaler() if name in ('LR', 'RBF-SVM') else 'passthrough'
    preprocessing = ColumnTransformer([
        ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categories),
        ('numeric', numeric, numbers),
    ])
    if name == 'LR':
        estimator = LogisticRegression(max_iter=4000, random_state=seed)
    elif name == 'RF':
        estimator = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=1)
    elif name == 'RBF-SVM':
        estimator = SVC(kernel='rbf', probability=False, cache_size=256)
    elif name == 'XGBoost':
        estimator = XGBClassifier(random_state=seed, **plan['xgboost_fixed'])
    else:
        raise ValueError(name)
    return Pipeline([('preprocessing', preprocessing), ('model', estimator)])


def score_method(name):
    return 'decision_function' if name == 'RBF-SVM' else 'predict_proba'


def positive_scores(raw_scores):
    return raw_scores[:, 1] if raw_scores.ndim == 2 else raw_scores


def select_cutoff(y, score):
    precision, recall, thresholds = precision_recall_curve(y, score)
    p, r = precision[:-1], recall[:-1]
    denominator = 4 * p + r
    f2 = np.divide(5 * p * r, denominator, out=np.zeros_like(p), where=denominator > 0)
    index = np.flatnonzero(f2 == f2.max())[-1]
    return float(thresholds[index]), float(f2[index])


def metrics(y, score, decision):
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y, decision, labels=[0, 1]).ravel())
    return {
        'n': len(y), 'positives': tp + fn,
        'ap': float(average_precision_score(y, score)), 'roc_auc': float(roc_auc_score(y, score)),
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'precision': tp / (tp + fp) if tp + fp else 0.0,
        'recall': tp / (tp + fn), 'f1': 2 * tp / (2 * tp + fp + fn),
        'f2': 5 * tp / (5 * tp + 4 * fn + fp),
        'accuracy': (tp + tn) / len(y),
    }


