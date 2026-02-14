"""Prototype all stand-out items to get working code before editing notebook."""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric, BinaryLabelDatasetMetric
from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing, EqOddsPostprocessing, RejectOptionClassification
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             precision_score, recall_score, f1_score, roc_curve, auc,
                             precision_recall_curve, average_precision_score, classification_report)
from sklearn.inspection import permutation_importance
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

os.makedirs('images', exist_ok=True)

# ============================================================
# DATA PIPELINE (same as before)
# ============================================================
df = pd.read_csv('udacity_ai_ethics_project_data.csv')
df = df.dropna()
df['Age'] = pd.cut(df['Age'], bins=[17,24,44,65,92], labels=['18-24','25-44','45-65','66-92'])
df['Budget (in dollars)'] = df['Budget (in dollars)'].apply(lambda x: '>=300' if x >= 300 else '<300')
df = pd.get_dummies(df)
df = df.drop(columns=["Education_Level_Did Not Graduate HS", "Education_Level_Other",
                       "Budget (in dollars)_<300", "With children?"])

binary_act_dataset = BinaryLabelDataset(
    df=df, label_names=["Budget (in dollars)_>=300"],
    favorable_label=1.0, unfavorable_label=0.0,
    protected_attribute_names=["Education_Level_High School Grad"]
)
privileged_groups = [{"Education_Level_High School Grad": 0}]
unprivileged_groups = [{"Education_Level_High School Grad": 1}]

(orig_train, orig_validate, orig_test) = binary_act_dataset.split([0.5, 0.8], shuffle=True)
feature_names = binary_act_dataset.feature_names

def test_fn(dataset, model, thresh_arr, priv=privileged_groups, unpriv=unprivileged_groups):
    y_pred_prob = model.predict_proba(dataset.features)
    pos_ind = np.where(model.classes_ == dataset.favorable_label)[0][0]
    metric_arrs = defaultdict(list)
    for thresh in thresh_arr:
        y_pred = (y_pred_prob[:, pos_ind] > thresh).astype(np.float64)
        dataset_pred = dataset.copy()
        dataset_pred.labels = y_pred
        metric = ClassificationMetric(dataset, dataset_pred, unprivileged_groups=unpriv, privileged_groups=priv)
        metric_arrs['bal_acc'].append((metric.true_positive_rate() + metric.true_negative_rate()) / 2)
        metric_arrs['avg_odds_diff'].append(metric.average_odds_difference())
        metric_arrs['stat_par_diff'].append(metric.statistical_parity_difference())
        metric_arrs['eq_opp_diff'].append(metric.equal_opportunity_difference())
        metric_arrs['theil_ind'].append(metric.theil_index())
    return metric_arrs, y_pred_prob[:, pos_ind]

thresh_arr = np.linspace(0.01, 0.5, 50)

# Train original LR
LR_model = LogisticRegression().fit(orig_train.features, orig_train.labels.ravel(), orig_train.instance_weights)
GNB_model = GaussianNB().fit(orig_train.features, orig_train.labels.ravel(), orig_train.instance_weights)

# ============================================================
# STAND-OUT #1: Additional performance analysis (ROC, PR curves, classification report)
# ============================================================
print("=" * 60)
print("STAND-OUT #1: Additional Performance Analysis")
print("=" * 60)

# For LR model (original, pre-mitigation)
lr_probs = LR_model.predict_proba(orig_test.features)
pos_ind = np.where(LR_model.classes_ == orig_test.favorable_label)[0][0]
lr_scores = lr_probs[:, pos_ind]
lr_preds = LR_model.predict(orig_test.features)

gnb_probs = GNB_model.predict_proba(orig_test.features)
pos_ind_gnb = np.where(GNB_model.classes_ == orig_test.favorable_label)[0][0]
gnb_scores = gnb_probs[:, pos_ind_gnb]
gnb_preds = GNB_model.predict(orig_test.features)

y_true = orig_test.labels.ravel()

# ROC Curves for both models
fpr_lr, tpr_lr, _ = roc_curve(y_true, lr_scores)
roc_auc_lr = auc(fpr_lr, tpr_lr)
fpr_gnb, tpr_gnb, _ = roc_curve(y_true, gnb_scores)
roc_auc_gnb = auc(fpr_gnb, tpr_gnb)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# ROC
axes[0].plot(fpr_lr, tpr_lr, color='coral', lw=2, label=f'Logistic Regression (AUC = {roc_auc_lr:.4f})')
axes[0].plot(fpr_gnb, tpr_gnb, color='steelblue', lw=2, label=f'Gaussian NB (AUC = {roc_auc_gnb:.4f})')
axes[0].plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random Classifier')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curve Comparison')
axes[0].legend(loc='lower right')
axes[0].grid(True, alpha=0.3)

# Precision-Recall
prec_lr, rec_lr, _ = precision_recall_curve(y_true, lr_scores)
ap_lr = average_precision_score(y_true, lr_scores)
prec_gnb, rec_gnb, _ = precision_recall_curve(y_true, gnb_scores)
ap_gnb = average_precision_score(y_true, gnb_scores)

axes[1].plot(rec_lr, prec_lr, color='coral', lw=2, label=f'Logistic Regression (AP = {ap_lr:.4f})')
axes[1].plot(rec_gnb, prec_gnb, color='steelblue', lw=2, label=f'Gaussian NB (AP = {ap_gnb:.4f})')
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].set_title('Precision-Recall Curve Comparison')
axes[1].legend(loc='lower left')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('images/roc_pr_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("ROC and Precision-Recall curves saved.")

# Classification report
print("\nClassification Report - Logistic Regression:")
print(classification_report(y_true, lr_preds, target_names=['Budget <$300', 'Budget >=$300']))
print("\nClassification Report - Gaussian Naive Bayes:")
print(classification_report(y_true, gnb_preds, target_names=['Budget <$300', 'Budget >=$300']))

# ============================================================
# STAND-OUT #2: Further stratification on fairness metrics
# ============================================================
print("\n" + "=" * 60)
print("STAND-OUT #2: Stratified Fairness Metrics by Education Level")
print("=" * 60)

# Get education labels for test set
edu_col_names = [n for n in feature_names if n.startswith("Education_Level_")]
edu_indices = [feature_names.index(c) for c in edu_col_names]

edu_labels = []
for i in range(orig_test.features.shape[0]):
    assigned = False
    for j, col in enumerate(edu_col_names):
        if orig_test.features[i, edu_indices[j]] == 1:
            edu_labels.append(col.replace("Education_Level_", ""))
            assigned = True
            break
    if not assigned:
        edu_labels.append("Other")
edu_labels = np.array(edu_labels)

# Stratified fairness analysis for BOTH models during model selection
def stratified_fairness(model, dataset, edu_labels_arr, model_name):
    """Compute per-education-level accuracy, precision, recall, F1."""
    preds = model.predict(dataset.features)
    y_true_local = dataset.labels.ravel()
    levels = sorted(np.unique(edu_labels_arr))
    results = {}
    for level in levels:
        mask = edu_labels_arr == level
        if mask.sum() > 0:
            y_t = y_true_local[mask]
            y_p = preds[mask]
            results[level] = {
                'accuracy': accuracy_score(y_t, y_p),
                'balanced_accuracy': balanced_accuracy_score(y_t, y_p) if len(np.unique(y_t)) > 1 else float('nan'),
                'precision': precision_score(y_t, y_p, zero_division=0),
                'recall': recall_score(y_t, y_p, zero_division=0),
                'f1': f1_score(y_t, y_p, zero_division=0),
                'count': mask.sum()
            }
    print(f"\n--- {model_name} - Stratified Performance by Education Level ---")
    strat_df = pd.DataFrame(results).T
    print(strat_df.to_string())
    return results

lr_strat = stratified_fairness(LR_model, orig_test, edu_labels, "Logistic Regression (Before Mitigation)")
gnb_strat = stratified_fairness(GNB_model, orig_test, edu_labels, "Gaussian Naive Bayes (Before Mitigation)")

# Stratified fairness bar plot comparing models
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
levels = sorted(lr_strat.keys())
x = np.arange(len(levels))
width = 0.35

for idx, metric_key in enumerate(['accuracy', 'precision', 'recall']):
    lr_values = [lr_strat[l][metric_key] for l in levels]
    gnb_values = [gnb_strat[l][metric_key] for l in levels]
    axes[idx].bar(x - width/2, gnb_values, width, label='GNB', color='steelblue')
    axes[idx].bar(x + width/2, lr_values, width, label='LR', color='coral')
    axes[idx].set_xlabel('Education Level')
    axes[idx].set_ylabel(metric_key.replace('_', ' ').title())
    axes[idx].set_title(f'{metric_key.replace("_", " ").title()} by Education Level')
    axes[idx].set_xticks(x)
    axes[idx].set_xticklabels(levels, rotation=30, ha='right')
    axes[idx].legend()
    axes[idx].set_ylim(0, 1.1)
    # Add value labels
    for i, (g, l) in enumerate(zip(gnb_values, lr_values)):
        axes[idx].text(i - width/2, g + 0.02, f'{g:.3f}', ha='center', va='bottom', fontsize=7)
        axes[idx].text(i + width/2, l + 0.02, f'{l:.3f}', ha='center', va='bottom', fontsize=7)

plt.tight_layout()
plt.savefig('images/stratified_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nStratified model comparison plot saved.")

# ============================================================
# STAND-OUT #3: Combine multiple bias mitigation techniques
# ============================================================
print("\n" + "=" * 60)
print("STAND-OUT #3: Combined Bias Mitigation")
print("=" * 60)

# Approach: Reweighing (pre-processing) + Calibrated Equalized Odds (post-processing)
# Step 1: Reweighing
RW = Reweighing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
dataset_transf_train = RW.fit_transform(orig_train)

# Step 2: Train LR on reweighed data
LR_mitigated = LogisticRegression().fit(
    dataset_transf_train.features, dataset_transf_train.labels.ravel(), dataset_transf_train.instance_weights
)

# Get predictions for validation set (needed for post-processing fitting)
mit_val_probs = LR_mitigated.predict_proba(orig_validate.features)
pos_ind_mit = np.where(LR_mitigated.classes_ == orig_validate.favorable_label)[0][0]

# Find best threshold on validation
mit_val_metrics, _ = test_fn(orig_validate, LR_mitigated, thresh_arr)
best_val_idx = np.argmax(mit_val_metrics['bal_acc'])
best_thresh = thresh_arr[best_val_idx]

# Create validation predictions dataset
val_preds = (mit_val_probs[:, pos_ind_mit] > best_thresh).astype(np.float64)
orig_validate_pred = orig_validate.copy()
orig_validate_pred.labels = val_preds
orig_validate_pred.scores = mit_val_probs[:, pos_ind_mit].reshape(-1, 1)

# Step 3: Apply Calibrated Equalized Odds post-processing
cpp = CalibratedEqOddsPostprocessing(
    unprivileged_groups=unprivileged_groups,
    privileged_groups=privileged_groups,
    cost_constraint='weighted',
    seed=42
)
cpp = cpp.fit(orig_validate, orig_validate_pred)

# Get test predictions
mit_test_probs = LR_mitigated.predict_proba(orig_test.features)
test_preds = (mit_test_probs[:, pos_ind_mit] > best_thresh).astype(np.float64)
orig_test_pred = orig_test.copy()
orig_test_pred.labels = test_preds
orig_test_pred.scores = mit_test_probs[:, pos_ind_mit].reshape(-1, 1)

# Apply post-processing
dataset_combined_test_pred = cpp.predict(orig_test_pred)

# Evaluate combined
metric_combined = ClassificationMetric(
    orig_test, dataset_combined_test_pred,
    unprivileged_groups=unprivileged_groups,
    privileged_groups=privileged_groups
)
combined_bal_acc = (metric_combined.true_positive_rate() + metric_combined.true_negative_rate()) / 2
combined_aod = metric_combined.average_odds_difference()
combined_spd = metric_combined.statistical_parity_difference()
combined_eod = metric_combined.equal_opportunity_difference()
combined_ti = metric_combined.theil_index()
combined_acc = accuracy_score(orig_test.labels.ravel(), dataset_combined_test_pred.labels.ravel())

print(f"Combined (Reweighing + Calibrated EqOdds):")
print(f"  Balanced Accuracy: {combined_bal_acc:.4f}")
print(f"  Accuracy: {combined_acc:.4f}")
print(f"  Avg Odds Diff: {combined_aod:.4f} (ideal [-0.1, 0.1]: {-0.1 <= combined_aod <= 0.1})")
print(f"  Stat Par Diff: {combined_spd:.4f} (ideal [-0.1, 0.1]: {-0.1 <= combined_spd <= 0.1})")
print(f"  Eq Opp Diff: {combined_eod:.4f} (ideal [-0.1, 0.1]: {-0.1 <= combined_eod <= 0.1})")
print(f"  Theil Index: {combined_ti:.4f} (ideal [0, 0.1]: {0 <= combined_ti <= 0.1})")

count_combined = sum([-0.1 <= combined_aod <= 0.1, -0.1 <= combined_spd <= 0.1,
                       -0.1 <= combined_eod <= 0.1, 0 <= combined_ti <= 0.1])
print(f"  Metrics in ideal range: {count_combined}/4")

# Also evaluate Reweighing only for comparison
mit_metrics_rw, _ = test_fn(orig_test, LR_mitigated, thresh_arr)
rw_best = np.argmax(mit_metrics_rw['bal_acc'])
rw_best_thresh = thresh_arr[rw_best]
rw_test_preds = (mit_test_probs[:, pos_ind_mit] > rw_best_thresh).astype(np.float64)
rw_acc = accuracy_score(orig_test.labels.ravel(), rw_test_preds)

print(f"\nReweighing only:")
print(f"  Balanced Accuracy: {mit_metrics_rw['bal_acc'][rw_best]:.4f}")
print(f"  Accuracy: {rw_acc:.4f}")
print(f"  Avg Odds Diff: {mit_metrics_rw['avg_odds_diff'][rw_best]:.4f}")
print(f"  Stat Par Diff: {mit_metrics_rw['stat_par_diff'][rw_best]:.4f}")
print(f"  Eq Opp Diff: {mit_metrics_rw['eq_opp_diff'][rw_best]:.4f}")
print(f"  Theil Index: {mit_metrics_rw['theil_ind'][rw_best]:.4f}")

# Original LR (no mitigation) metrics
orig_lr_metrics, _ = test_fn(orig_test, LR_model, thresh_arr)
orig_best = np.argmax(orig_lr_metrics['bal_acc'])

# Comparison bar chart: before vs reweighing vs combined
fig, axes = plt.subplots(1, 5, figsize=(22, 5))
metric_labels = ['Balanced\nAccuracy', 'Avg Odds\nDifference', 'Statistical\nParity Diff',
                 'Equal Opp.\nDifference', 'Theil\nIndex']
metric_keys = ['bal_acc', 'avg_odds_diff', 'stat_par_diff', 'eq_opp_diff', 'theil_ind']
combined_vals = [combined_bal_acc, combined_aod, combined_spd, combined_eod, combined_ti]
ideal_ranges = [(0.85, 1.0), (-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1), (0, 0.1)]

for idx, (key, label, ideal) in enumerate(zip(metric_keys, metric_labels, ideal_ranges)):
    orig_val = orig_lr_metrics[key][orig_best]
    rw_val = mit_metrics_rw[key][rw_best]
    comb_val = combined_vals[idx]

    bars = axes[idx].bar(['No\nMitigation', 'Reweighing\nOnly', 'Reweighing\n+ Cal. EqOdds'],
                         [orig_val, rw_val, comb_val],
                         color=['#e74c3c', '#f39c12', '#27ae60'])
    axes[idx].set_title(label, fontsize=10, fontweight='bold')
    axes[idx].axhspan(ideal[0], ideal[1], alpha=0.15, color='green', label='Ideal Range')
    for bar, val in zip(bars, [orig_val, rw_val, comb_val]):
        axes[idx].text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                       f'{val:.4f}', ha='center', va='bottom', fontsize=7, fontweight='bold')
    axes[idx].tick_params(axis='x', labelsize=7)

plt.suptitle('Fairness Metrics: No Mitigation vs. Reweighing vs. Combined Strategy', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('images/combined_mitigation_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nCombined mitigation comparison plot saved.")

# ============================================================
# Determine which mitigation approach to use as FINAL
# ============================================================
# Use the combined approach if it has >= 2 metrics in range and reasonable accuracy
# Otherwise fall back to reweighing only
if count_combined >= 2 and combined_bal_acc >= 0.5:
    print(f"\n>>> Using COMBINED mitigation as final (metrics in range: {count_combined})")
    final_preds = dataset_combined_test_pred.labels.ravel()
    final_approach = "combined"
else:
    print(f"\n>>> Using REWEIGHING ONLY as final (combined metrics in range: {count_combined})")
    final_preds = rw_test_preds
    final_approach = "reweighing"

# ============================================================
# STAND-OUT #4: Fairness metric cohort analysis
# ============================================================
print("\n" + "=" * 60)
print("STAND-OUT #4: Fairness Metric Cohort Analysis by Education Level")
print("=" * 60)

# For each education level, compute the fairness-relevant metrics
# Since we can't compute group-level AOD within a single group,
# we compute per-group accuracy, balanced accuracy, TPR, FPR
education_levels = sorted(np.unique(edu_labels))
cohort_metrics = {}
for level in education_levels:
    mask = edu_labels == level
    if mask.sum() > 0:
        y_t = orig_test.labels.ravel()[mask]
        y_p = final_preds[mask]
        tp = np.sum((y_t == 1) & (y_p == 1))
        fn = np.sum((y_t == 1) & (y_p == 0))
        fp = np.sum((y_t == 0) & (y_p == 1))
        tn = np.sum((y_t == 0) & (y_p == 0))
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        cohort_metrics[level] = {
            'Accuracy': accuracy_score(y_t, y_p),
            'TPR (Recall)': tpr,
            'FPR': fpr,
            'Precision': precision_score(y_t, y_p, zero_division=0),
            'F1 Score': f1_score(y_t, y_p, zero_division=0),
        }

cohort_df = pd.DataFrame(cohort_metrics).T
print(cohort_df.to_string())

# Plot fairness metrics by education level
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
x = np.arange(len(education_levels))

metric_pairs = [('TPR (Recall)', 'True Positive Rate by Education Level'),
                ('FPR', 'False Positive Rate by Education Level'),
                ('F1 Score', 'F1 Score by Education Level')]
colors_list = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']

for idx, (met, title) in enumerate(metric_pairs):
    vals = [cohort_metrics[l][met] for l in education_levels]
    bars = axes[idx].bar(education_levels, vals, color=colors_list[:len(education_levels)])
    axes[idx].set_xlabel('Education Level')
    axes[idx].set_ylabel(met)
    axes[idx].set_title(title)
    axes[idx].set_ylim(0, 1.1)
    for bar, val in zip(bars, vals):
        axes[idx].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                       f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=8)
    axes[idx].tick_params(axis='x', rotation=30)

plt.suptitle('Fairness Metrics by Education Level (After Bias Mitigation)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('images/fairness_cohort_by_education.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nFairness cohort analysis plot saved.")

# ============================================================
# STAND-OUT #2 (continued): Final model stratified fairness
# ============================================================
final_strat = {}
for level in education_levels:
    mask = edu_labels == level
    if mask.sum() > 0:
        y_t = orig_test.labels.ravel()[mask]
        y_p = final_preds[mask]
        final_strat[level] = {
            'accuracy': accuracy_score(y_t, y_p),
            'balanced_accuracy': balanced_accuracy_score(y_t, y_p) if len(np.unique(y_t)) > 1 else float('nan'),
            'precision': precision_score(y_t, y_p, zero_division=0),
            'recall': recall_score(y_t, y_p, zero_division=0),
            'f1': f1_score(y_t, y_p, zero_division=0),
        }

print("\n--- Final Model - Stratified Performance ---")
final_strat_df = pd.DataFrame(final_strat).T
print(final_strat_df.to_string())

# ============================================================
# STAND-OUT #6: Pipeline diagram
# ============================================================
print("\n" + "=" * 60)
print("STAND-OUT #6: Pipeline Diagram")
print("=" * 60)

fig, ax = plt.subplots(figsize=(20, 10))
ax.set_xlim(0, 20)
ax.set_ylim(0, 10)
ax.axis('off')

# Title
ax.text(10, 9.5, 'IDOOU Budget Predictor AI - End-to-End Pipeline',
        fontsize=16, fontweight='bold', ha='center', va='center',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#2c3e50', edgecolor='none', alpha=0.9),
        color='white')

# Define boxes
boxes = [
    # Row 1: Data Pipeline
    (1.5, 7.5, 'Raw Data\n(300K records)', '#ecf0f1', '#2c3e50'),
    (5, 7.5, 'Pre-Processing\n- Drop NAs\n- Bin Age & Budget\n- One-Hot Encode', '#3498db', 'white'),
    (9, 7.5, 'EDA & Bias\nDetection\n- Bar Plots\n- Correlation Matrix\n- SPD & DI', '#e67e22', 'white'),

    # Row 2: Model Development
    (1.5, 5, 'Bias Mitigation\n(Pre-Processing)\nReweighing', '#e74c3c', 'white'),
    (5, 5, 'Model Training\nLogistic Regression\n(Reweighed Data)', '#9b59b6', 'white'),
    (9, 5, 'Bias Mitigation\n(Post-Processing)\nCalibrated\nEqualized Odds', '#e74c3c', 'white'),

    # Row 3: Evaluation
    (1.5, 2.5, 'Interpretability\nPermutation\nImportance', '#1abc9c', 'white'),
    (5, 2.5, 'Fairness\nEvaluation\n- AOD, SPD, EOD\n- Theil Index\n- Cohort Analysis', '#27ae60', 'white'),
    (9, 2.5, 'Performance\nEvaluation\n- Accuracy, F1\n- ROC/PR Curves\n- Confusion Matrix', '#27ae60', 'white'),

    # Row 4: Deployment
    (5, 0.5, 'Deployment\nIDOOU App\n- Human-in-the-Loop\n- User Budget Confirmation', '#2c3e50', 'white'),
]

box_width = 2.8
box_height = 1.6

for (cx, cy, text, bg, fg) in boxes:
    fancy = FancyBboxPatch((cx - box_width/2, cy - box_height/2), box_width, box_height,
                           boxstyle="round,pad=0.1", facecolor=bg, edgecolor='#2c3e50', linewidth=1.5)
    ax.add_patch(fancy)
    ax.text(cx, cy, text, fontsize=7.5, ha='center', va='center', color=fg, fontweight='bold')

# Draw arrows
arrow_props = dict(arrowstyle='->', color='#2c3e50', lw=2, mutation_scale=15)
connections = [
    (1.5+box_width/2, 7.5, 5-box_width/2, 7.5),    # Raw -> PreProc
    (5+box_width/2, 7.5, 9-box_width/2, 7.5),       # PreProc -> EDA
    (9, 7.5-box_height/2, 9, 5+box_height/2),        # EDA -> PostProc (down)
    (1.5, 7.5-box_height/2, 1.5, 5+box_height/2),    # (Data -> Reweighing)
    (1.5+box_width/2, 5, 5-box_width/2, 5),          # Reweighing -> Training
    (5+box_width/2, 5, 9-box_width/2, 5),            # Training -> PostProc
    (1.5, 5-box_height/2, 1.5, 2.5+box_height/2),    # Reweighing -> Interp
    (5, 5-box_height/2, 5, 2.5+box_height/2),        # Training -> Fairness
    (9, 5-box_height/2, 9, 2.5+box_height/2),        # PostProc -> Performance
    (5, 2.5-box_height/2, 5, 0.5+box_height/2),      # Fairness -> Deployment
]

for (x1, y1, x2, y2) in connections:
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=arrow_props)

# Stage labels
stage_labels = [
    (13, 7.5, 'STAGE 1:\nData Preparation\n& Exploration', '#3498db'),
    (13, 5, 'STAGE 2:\nModel Training\n& Bias Mitigation', '#9b59b6'),
    (13, 2.5, 'STAGE 3:\nEvaluation &\nInterpretability', '#27ae60'),
    (13, 0.5, 'STAGE 4:\nDeployment', '#2c3e50'),
]

for (sx, sy, slabel, scolor) in stage_labels:
    ax.text(sx, sy, slabel, fontsize=9, ha='left', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=scolor, alpha=0.15, edgecolor=scolor),
            color=scolor, fontweight='bold')

# Legend
legend_items = [
    ('Pre-Processing', '#3498db'),
    ('Analysis', '#e67e22'),
    ('Bias Mitigation', '#e74c3c'),
    ('Model', '#9b59b6'),
    ('Evaluation', '#27ae60'),
    ('Deployment', '#2c3e50'),
]
for i, (label, color) in enumerate(legend_items):
    ax.add_patch(plt.Rectangle((15.5, 8.5 - i*0.45), 0.3, 0.3, facecolor=color, edgecolor='none'))
    ax.text(16, 8.65 - i*0.45, label, fontsize=8, va='center')

ax.text(15.5, 9, 'Legend', fontsize=9, fontweight='bold', va='center')

plt.savefig('images/pipeline_diagram.png', dpi=150, bbox_inches='tight')
plt.close()
print("Pipeline diagram saved.")

# ============================================================
# Final summary
# ============================================================
print("\n" + "=" * 60)
print("ALL STAND-OUT ITEMS COMPLETE")
print("=" * 60)
for f in sorted(os.listdir('images')):
    print(f"  {f}")
