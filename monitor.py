import sys
import json
import logging
import argparse

import pandas as pd
import numpy as np
import mlflow

from scipy.stats import ks_2samp
from scipy.spatial.distance import jensenshannon

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MONITOR_VERSION = '1.3 (2026-10-31) - KS/PSI/JS, pairs with fraud_detection.ipynb v1.3'

parser = argparse.ArgumentParser(description='Monthly monitoring job: KS, PSI, JS divergence, precision and random audit, logged to MLFlow')
parser.add_argument('--month', required=True, help='e.g. 2020-07')
parser.add_argument('--tracking', default='sqlite:///mlflow.db')
args = parser.parse_args()

with open('artifacts/config.json') as f:
    config = json.load(f)

reference_sample = pd.read_csv('artifacts/reference_sample.csv')
scored_df = pd.read_csv(f'scored/{args.month}.csv')
verdicts_df = pd.read_csv(f'verdicts/{args.month}.csv')
logger.info(f'monitor.py v{MONITOR_VERSION}')
logger.info(f'{args.month}: {len(scored_df):,} scored | {len(verdicts_df)} investigated')


def run_ks_test(data_ref, data_prod, feature_name):
    """
    KS test for continuous features. With tens of thousands of rows the p-value is always ~0,
    so the verdict is based on the KS statistic D (max gap between the two CDFs).
    """
    statistic, p_value = ks_2samp(data_ref, data_prod)

    if statistic < 0.05:
        interpretation = 'No significant drift'
    elif statistic < 0.15:
        interpretation = 'MARGINAL'
    else:
        interpretation = 'DRIFT DETECTED'

    logger.info(f'KS {feature_name:<22} D: {statistic:.4f}  p-value: {p_value:.2e}  {interpretation}')
    return statistic, p_value


def calculate_psi(reference_pct, production, bin_edges, feature_name):
    """
    Population Stability Index. Reference proportions and bin edges come from training time
    (deciles of the training data), production values are binned on those same edges.
    """
    edges = np.array(bin_edges, dtype=float)
    edges[0], edges[-1] = -np.inf, np.inf
    prod_pct = np.histogram(production, edges)[0] / len(production)

    epsilon = 1e-10
    psi = 0
    breakdown = []

    for i, (p, q) in enumerate(zip(reference_pct, prod_pct)):
        p = max(p, epsilon)
        q = max(q, epsilon)
        psi_i = (p - q) * np.log(p / q)
        psi += psi_i
        breakdown.append({'bin': f'{edges[i]:.3g} to {edges[i + 1]:.3g}', 'ref': p, 'prod': q, 'psi_contribution': psi_i})

    if psi < 0.1:
        interpretation = 'No significant shift'
    elif psi < 0.25:
        interpretation = 'Moderate shift - investigate'
    else:
        interpretation = 'Significant shift - action needed'

    if psi >= 0.1:
        logger.info(f'PSI {feature_name:<22} {psi:.4f}  {interpretation}')
    return psi, pd.DataFrame(breakdown)


def calculate_js_divergence(reference_pct, production, feature_name):
    """
    Jensen-Shannon divergence for categorical features. scipy returns the square root, so square it.
    """
    prod_counts = production.value_counts(normalize=True)
    all_categories = sorted(set(reference_pct.keys()) | set(prod_counts.index))

    epsilon = 1e-10
    p = np.array([reference_pct.get(cat, 0) for cat in all_categories]) + epsilon
    q = np.array([prod_counts.get(cat, 0) for cat in all_categories]) + epsilon
    p, q = p / p.sum(), q / q.sum()

    js_div = jensenshannon(p, q) ** 2

    if js_div < 0.05:
        interpretation = 'Very similar distributions'
    elif js_div < 0.1:
        interpretation = 'Some difference - monitor'
    else:
        interpretation = 'Significant difference'

    logger.info(f'JS  {feature_name:<22} {js_div:.4f}  {interpretation}')
    return js_div


# data drift: KS on continuous features, PSI on binned features and on the score, JS on category
logger.info('=== Data Drift ===')
ks_results = {f: run_ks_test(reference_sample[f], scored_df[f], f) for f in config['ks_features']}

psi_results, psi_breakdowns = {}, {}
for f, ref in config['reference_bins'].items():
    if f in scored_df.columns:
        psi_results[f], psi_breakdowns[f] = calculate_psi(ref['pct'], scored_df[f].values, ref['edges'], f)
psi_score, _ = calculate_psi(config['score_bins']['pct'], scored_df['score'].values, config['score_bins']['edges'], 'score')

js_category = calculate_js_divergence(config['reference_category'], scored_df['category'], 'category')

worst_feature = max(psi_results, key=psi_results.get)
logger.info(f'PSI computed for {len(psi_results)} features | max {psi_results[worst_feature]:.4f} ({worst_feature}) | score {psi_score:.4f}')
if psi_results[worst_feature] >= 0.1:
    logger.info(f'Bins contributing most to {worst_feature} PSI:\n{psi_breakdowns[worst_feature].sort_values('psi_contribution', ascending=False).head(3).to_string(index=False)}')

# model performance on what the investigators actually looked at
logger.info('=== Model Performance ===')
top_alerts = verdicts_df[verdicts_df['selection'] == 'top']
random_audit = verdicts_df[verdicts_df['selection'] == 'random']
precision_top = top_alerts['is_fraud'].mean()
random_fraud_rate = random_audit['is_fraud'].mean() if len(random_audit) else np.nan
logger.info(f'Precision of top alerts: {precision_top:.3f} (floor {config['precision_floor']})')
logger.info(f'Fraud rate in random audit: {random_fraud_rate:.4f}')

# summary
alarms = []
if psi_results[worst_feature] >= 0.25:
    alarms.append(f'data drift: {worst_feature} PSI {psi_results[worst_feature]:.2f}')
if any(stat >= 0.15 for stat, _ in ks_results.values()):
    worst_ks = max(ks_results, key=lambda f: ks_results[f][0])
    alarms.append(f'data drift: {worst_ks} KS D {ks_results[worst_ks][0]:.2f}')
if js_category >= 0.1:
    alarms.append(f'data drift: category JS {js_category:.3f}')
if psi_score >= 0.25:
    alarms.append(f'score drift: PSI {psi_score:.2f}')
if precision_top < config['precision_floor']:
    alarms.append(f'precision {precision_top:.3f} below floor {config['precision_floor']}')

mlflow.set_tracking_uri(args.tracking)
mlflow.set_experiment('fraud-monitoring')
with mlflow.start_run(run_name=args.month):
    mlflow.log_param('month', args.month)
    mlflow.log_param('champion_version', config['champion_version'])
    mlflow.log_metric('n_scored', len(scored_df))
    mlflow.log_metric('n_investigated', len(verdicts_df))
    mlflow.log_metric('precision_top', precision_top)
    mlflow.log_metric('random_audit_fraud_rate', random_fraud_rate)
    mlflow.log_metric('psi_score', psi_score)
    mlflow.log_metric('psi_max_feature', psi_results[worst_feature])
    mlflow.log_metric('js_category', js_category)
    mlflow.log_metric('n_alarms', len(alarms))
    for f, value in psi_results.items():
        mlflow.log_metric(f'psi_{f}', value)
    for f, (stat, _) in ks_results.items():
        mlflow.log_metric(f'ks_{f}', stat)
    mlflow.set_tag('alarms', '; '.join(alarms) if alarms else 'none')

if alarms:
    for alarm in alarms:
        logger.warning(f'ALARM: {alarm}')
    sys.exit(1)

logger.info('No alarms')
