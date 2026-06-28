"""
=============================================================
  PROJECT: Employee Attrition Analysis
  Goal   : Understand & predict employee turnover
  Author : Nandani Kumari
  Models : Logistic Regression · Random Forest · Gradient Boosting
  Dataset: Synthetic HR Analytics dataset (IBM-style)
=============================================================
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, accuracy_score,
                             precision_score, recall_score, f1_score)
import os

OUT = "/mnt/user-data/outputs/employee_attrition"
os.makedirs(OUT, exist_ok=True)

SEED = 42
np.random.seed(SEED)

print("=" * 60)
print("  EMPLOYEE ATTRITION ANALYSIS — END-TO-END PROJECT")
print("=" * 60)

# ═══════════════════════════════════════════════════════════
# 1. GENERATE REALISTIC HR DATASET
# ═══════════════════════════════════════════════════════════
print("\n[1/7] Generating synthetic HR dataset …")

N = 1470  # IBM HR Analytics benchmark size

departments   = ["Sales", "Research & Development", "Human Resources"]
dept_weights  = [0.35, 0.55, 0.10]
job_roles = {
    "Sales":                   ["Sales Executive", "Sales Representative", "Manager"],
    "Research & Development":  ["Research Scientist", "Laboratory Technician",
                                "Manufacturing Director", "Research Director", "Healthcare Representative"],
    "Human Resources":         ["Human Resources", "Manager"],
}
education_fields = ["Life Sciences", "Medical", "Marketing",
                    "Technical Degree", "Human Resources", "Other"]
marital_status   = ["Single", "Married", "Divorced"]

dept_col, role_col = [], []
for _ in range(N):
    d = np.random.choice(departments, p=dept_weights)
    dept_col.append(d)
    role_col.append(np.random.choice(job_roles[d]))

age            = np.random.randint(18, 61, N)
monthly_income = np.clip(np.random.normal(6500, 3000, N), 1000, 20000).astype(int)
years_company  = np.random.randint(0, 41, N)
years_role     = np.clip(np.random.randint(0, years_company + 1, N), 0, years_company).astype(int)
job_satisfaction     = np.random.randint(1, 5, N)   # 1=Low 4=Very High
work_life_balance    = np.random.randint(1, 5, N)
environment_sat      = np.random.randint(1, 5, N)
performance_rating   = np.random.choice([3, 4], N, p=[0.85, 0.15])
overtime             = np.random.choice(["Yes", "No"], N, p=[0.28, 0.72])
distance_from_home   = np.random.randint(1, 30, N)
num_companies_worked = np.random.randint(0, 10, N)
training_last_year   = np.random.randint(0, 7, N)
education            = np.random.randint(1, 6, N)
education_field      = np.random.choice(education_fields, N)
marital              = np.random.choice(marital_status, N, p=[0.32, 0.46, 0.22])
job_level            = np.random.randint(1, 6, N)
stock_option         = np.random.randint(0, 4, N)
percent_salary_hike  = np.random.randint(11, 26, N)

# --- Attrition probability (realistic causal factors) ---
attrition_prob = (
    0.10
    + 0.15 * (overtime == "Yes")
    + 0.08 * (job_satisfaction <= 2)
    + 0.06 * (work_life_balance <= 2)
    + 0.05 * (distance_from_home > 20)
    + 0.07 * (monthly_income < 3000)
    + 0.04 * (years_company < 2)
    + 0.03 * (num_companies_worked > 5)
    + 0.04 * (marital == "Single")
    - 0.05 * (job_level >= 4)
    - 0.03 * (stock_option >= 2)
    - 0.02 * (training_last_year >= 4)
)
attrition_prob = np.clip(attrition_prob, 0.02, 0.85)
attrition      = (np.random.rand(N) < attrition_prob).astype(int)

df = pd.DataFrame({
    "Age":                   age,
    "Department":            dept_col,
    "JobRole":               role_col,
    "MonthlyIncome":         monthly_income,
    "YearsAtCompany":        years_company,
    "YearsInCurrentRole":    years_role,
    "JobSatisfaction":       job_satisfaction,
    "WorkLifeBalance":       work_life_balance,
    "EnvironmentSatisfaction": environment_sat,
    "PerformanceRating":     performance_rating,
    "OverTime":              overtime,
    "DistanceFromHome":      distance_from_home,
    "NumCompaniesWorked":    num_companies_worked,
    "TrainingTimesLastYear": training_last_year,
    "Education":             education,
    "EducationField":        education_field,
    "MaritalStatus":         marital,
    "JobLevel":              job_level,
    "StockOptionLevel":      stock_option,
    "PercentSalaryHike":     percent_salary_hike,
    "Attrition":             attrition,
})

print(f"   Dataset shape     : {df.shape}")
print(f"   Attrition rate    : {df['Attrition'].mean()*100:.1f}%")
print(f"   Employees leaving : {df['Attrition'].sum()} / {N}")
print(f"   Departments       : {df['Department'].nunique()}")
print(f"   Job Roles         : {df['JobRole'].nunique()}")

df.to_csv(f"{OUT}/hr_dataset.csv", index=False)
print("   Saved → hr_dataset.csv")

# ═══════════════════════════════════════════════════════════
# 2. EXPLORATORY DATA ANALYSIS
# ═══════════════════════════════════════════════════════════
print("\n[2/7] Exploratory Data Analysis …")

pal = {"Stayed": "#2563EB", "Left": "#EF4444"}
df["AttritionLabel"] = df["Attrition"].map({0: "Stayed", 1: "Left"})

fig = plt.figure(figsize=(20, 16))
fig.suptitle("Employee Attrition Analysis — EDA Dashboard", fontsize=16, fontweight="bold", y=0.98)
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.38)

# 2a. Overall attrition pie
ax0 = fig.add_subplot(gs[0, 0])
counts = df["Attrition"].value_counts()
ax0.pie([counts[0], counts[1]], labels=["Stayed", "Left"],
        colors=["#2563EB", "#EF4444"], autopct="%1.1f%%",
        startangle=90, wedgeprops=dict(edgecolor="white", linewidth=2))
ax0.set_title("Overall Attrition Rate", fontsize=11, fontweight="bold")

# 2b. Attrition by Department
ax1 = fig.add_subplot(gs[0, 1])
dept_att = df.groupby("Department")["Attrition"].mean() * 100
colors_d = ["#3B82F6", "#EF4444", "#10B981"]
bars = ax1.bar(dept_att.index, dept_att.values, color=colors_d, edgecolor="white")
ax1.set_title("Attrition Rate by Department", fontsize=11, fontweight="bold")
ax1.set_ylabel("Attrition Rate (%)")
ax1.tick_params(axis="x", rotation=15)
for bar, val in zip(bars, dept_att.values):
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
             f"{val:.1f}%", ha="center", fontsize=9, fontweight="bold")
ax1.grid(axis="y", alpha=0.3)

# 2c. Age distribution
ax2 = fig.add_subplot(gs[0, 2])
for label, color in pal.items():
    subset = df[df["AttritionLabel"] == label]["Age"]
    ax2.hist(subset, bins=20, alpha=0.65, color=color, label=label, edgecolor="white")
ax2.set_title("Age Distribution by Attrition", fontsize=11, fontweight="bold")
ax2.set_xlabel("Age")
ax2.set_ylabel("Count")
ax2.legend()
ax2.grid(alpha=0.3)

# 2d. Monthly Income
ax3 = fig.add_subplot(gs[1, 0])
stayed = df[df["Attrition"] == 0]["MonthlyIncome"]
left   = df[df["Attrition"] == 1]["MonthlyIncome"]
ax3.boxplot([stayed, left], labels=["Stayed", "Left"],
            patch_artist=True,
            boxprops=dict(facecolor="#DBEAFE"),
            medianprops=dict(color="#1D4ED8", linewidth=2))
ax3.set_title("Monthly Income vs Attrition", fontsize=11, fontweight="bold")
ax3.set_ylabel("Monthly Income ($)")
ax3.grid(axis="y", alpha=0.3)

# 2e. OverTime impact
ax4 = fig.add_subplot(gs[1, 1])
ot_att = df.groupby("OverTime")["Attrition"].mean() * 100
ax4.bar(ot_att.index, ot_att.values,
        color=["#10B981", "#EF4444"], edgecolor="white", width=0.5)
ax4.set_title("Attrition Rate: Overtime vs No Overtime", fontsize=11, fontweight="bold")
ax4.set_ylabel("Attrition Rate (%)")
for i, val in enumerate(ot_att.values):
    ax4.text(i, val + 0.3, f"{val:.1f}%", ha="center", fontweight="bold", fontsize=11)
ax4.grid(axis="y", alpha=0.3)

# 2f. Job Satisfaction
ax5 = fig.add_subplot(gs[1, 2])
sat_labels = {1: "Low", 2: "Medium", 3: "High", 4: "Very High"}
js_att = df.groupby("JobSatisfaction")["Attrition"].mean() * 100
ax5.plot([sat_labels[i] for i in js_att.index], js_att.values,
         marker="o", color="#EF4444", linewidth=2, markersize=8)
ax5.fill_between(range(len(js_att)), js_att.values, alpha=0.15, color="#EF4444")
ax5.set_xticks(range(len(js_att)))
ax5.set_xticklabels([sat_labels[i] for i in js_att.index])
ax5.set_title("Attrition Rate by Job Satisfaction", fontsize=11, fontweight="bold")
ax5.set_ylabel("Attrition Rate (%)")
ax5.grid(alpha=0.3)

# 2g. Years at Company
ax6 = fig.add_subplot(gs[2, 0])
for label, color in pal.items():
    subset = df[df["AttritionLabel"] == label]["YearsAtCompany"]
    ax6.hist(subset, bins=20, alpha=0.65, color=color, label=label, edgecolor="white")
ax6.set_title("Years at Company by Attrition", fontsize=11, fontweight="bold")
ax6.set_xlabel("Years at Company")
ax6.set_ylabel("Count")
ax6.legend()
ax6.grid(alpha=0.3)

# 2h. Marital Status
ax7 = fig.add_subplot(gs[2, 1])
ms_att = df.groupby("MaritalStatus")["Attrition"].mean() * 100
colors_m = ["#3B82F6", "#F59E0B", "#10B981"]
bars = ax7.bar(ms_att.index, ms_att.values, color=colors_m, edgecolor="white")
ax7.set_title("Attrition Rate by Marital Status", fontsize=11, fontweight="bold")
ax7.set_ylabel("Attrition Rate (%)")
for bar, val in zip(bars, ms_att.values):
    ax7.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
             f"{val:.1f}%", ha="center", fontweight="bold")
ax7.grid(axis="y", alpha=0.3)

# 2i. Work Life Balance
ax8 = fig.add_subplot(gs[2, 2])
wlb_labels = {1: "Bad", 2: "Good", 3: "Better", 4: "Best"}
wlb_att = df.groupby("WorkLifeBalance")["Attrition"].mean() * 100
bars = ax8.bar([wlb_labels[i] for i in wlb_att.index], wlb_att.values,
               color=["#EF4444","#F59E0B","#3B82F6","#10B981"], edgecolor="white")
ax8.set_title("Attrition Rate by Work-Life Balance", fontsize=11, fontweight="bold")
ax8.set_ylabel("Attrition Rate (%)")
for bar, val in zip(bars, wlb_att.values):
    ax8.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
             f"{val:.1f}%", ha="center", fontsize=9, fontweight="bold")
ax8.grid(axis="y", alpha=0.3)

plt.savefig(f"{OUT}/01_eda_dashboard.png", dpi=150, bbox_inches="tight")
plt.close()
print("   Saved → 01_eda_dashboard.png")

# ═══════════════════════════════════════════════════════════
# 3. CORRELATION HEATMAP
# ═══════════════════════════════════════════════════════════
print("\n[3/7] Correlation Analysis …")

df_enc = df.copy()
le = LabelEncoder()
for col in ["Department","JobRole","OverTime","EducationField","MaritalStatus","AttritionLabel"]:
    df_enc[col] = le.fit_transform(df_enc[col].astype(str))

numeric_cols = ["Age","MonthlyIncome","YearsAtCompany","YearsInCurrentRole",
                "JobSatisfaction","WorkLifeBalance","EnvironmentSatisfaction",
                "OverTime","DistanceFromHome","NumCompaniesWorked",
                "TrainingTimesLastYear","JobLevel","StockOptionLevel",
                "PercentSalaryHike","Attrition"]

corr = df_enc[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(14, 11))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlBu_r",
            center=0, square=True, linewidths=0.5, ax=ax,
            annot_kws={"size": 8},
            cbar_kws={"shrink": 0.8})
ax.set_title("Feature Correlation Heatmap\n(Focus: Attrition row for key drivers)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT}/02_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("   Saved → 02_correlation_heatmap.png")

# ═══════════════════════════════════════════════════════════
# 4. PREPROCESSING & FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════
print("\n[4/7] Preprocessing & Model Training …")

df_model = df.drop(columns=["AttritionLabel"])
cat_cols  = ["Department","JobRole","OverTime","EducationField","MaritalStatus"]
for col in cat_cols:
    df_model[col] = le.fit_transform(df_model[col].astype(str))

X = df_model.drop(columns=["Attrition"])
y = df_model["Attrition"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED, stratify=y)

scaler  = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

print(f"   Train : {len(X_train)} samples  |  Test : {len(X_test)} samples")
print(f"   Train attrition rate : {y_train.mean()*100:.1f}%")
print(f"   Test  attrition rate : {y_test.mean()*100:.1f}%")

def evaluate_model(name, model, X_tr, X_te, y_tr, y_te, scaled=False):
    Xtr = X_tr if not scaled else X_train_s
    Xte = X_te if not scaled else X_test_s
    model.fit(Xtr, y_tr)
    y_pred     = model.predict(Xte)
    y_prob     = model.predict_proba(Xte)[:, 1]
    acc        = accuracy_score(y_te, y_pred)
    prec       = precision_score(y_te, y_pred, zero_division=0)
    rec        = recall_score(y_te, y_pred, zero_division=0)
    f1         = f1_score(y_te, y_pred, zero_division=0)
    roc        = roc_auc_score(y_te, y_prob)
    cv_scores  = cross_val_score(model, Xtr, y_tr,
                                  cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                                  scoring="roc_auc")
    print(f"   {name:<25}  Acc={acc:.3f}  Prec={prec:.3f}  Rec={rec:.3f}"
          f"  F1={f1:.3f}  ROC-AUC={roc:.3f}  CV={cv_scores.mean():.3f}±{cv_scores.std():.3f}")
    return {"Model": name, "Accuracy": round(acc,3), "Precision": round(prec,3),
            "Recall": round(rec,3), "F1": round(f1,3), "ROC_AUC": round(roc,3),
            "CV_Mean": round(cv_scores.mean(),3)}, y_pred, y_prob, model

results, preds, probs, models = [], {}, {}, {}

r, yp, ypr, m = evaluate_model("Logistic Regression",
    LogisticRegression(max_iter=1000, random_state=SEED, class_weight="balanced"),
    X_train_s, X_test_s, y_train, y_test, scaled=True)
results.append(r); preds["Logistic Regression"] = yp
probs["Logistic Regression"] = ypr; models["Logistic Regression"] = m

r, yp, ypr, m = evaluate_model("Random Forest",
    RandomForestClassifier(n_estimators=200, max_depth=10, class_weight="balanced",
                           random_state=SEED, n_jobs=-1),
    X_train, X_test, y_train, y_test)
results.append(r); preds["Random Forest"] = yp
probs["Random Forest"] = ypr; models["Random Forest"] = m

r, yp, ypr, m = evaluate_model("Gradient Boosting",
    GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                subsample=0.8, random_state=SEED),
    X_train, X_test, y_train, y_test)
results.append(r); preds["Gradient Boosting"] = yp
probs["Gradient Boosting"] = ypr; models["Gradient Boosting"] = m

results_df = pd.DataFrame(results)
results_df.to_csv(f"{OUT}/model_results.csv", index=False)
print(f"\n   Results saved → model_results.csv")

# ═══════════════════════════════════════════════════════════
# 5. MODEL EVALUATION PLOTS
# ═══════════════════════════════════════════════════════════
print("\n[5/7] Generating model evaluation plots …")

COLORS = {"Logistic Regression": "#3B82F6",
          "Random Forest":       "#10B981",
          "Gradient Boosting":   "#8B5CF6"}

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Model Evaluation — Confusion Matrices", fontsize=14, fontweight="bold")

for ax, (name, yp) in zip(axes, preds.items()):
    cm = confusion_matrix(y_test, yp)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Stayed","Left"], yticklabels=["Stayed","Left"],
                linewidths=1, linecolor="white",
                annot_kws={"size": 14, "fontweight": "bold"})
    ax.set_title(f"{name}\nROC-AUC = {results_df[results_df['Model']==name]['ROC_AUC'].values[0]:.3f}",
                 fontsize=11, fontweight="bold")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")

plt.tight_layout()
plt.savefig(f"{OUT}/03_confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close()
print("   Saved → 03_confusion_matrices.png")

# ROC curves
fig, ax = plt.subplots(figsize=(8, 7))
ax.plot([0,1],[0,1],"k--", linewidth=1, label="Random Classifier (AUC=0.50)")
for name, ypr in probs.items():
    fpr, tpr, _ = roc_curve(y_test, ypr)
    auc = results_df[results_df["Model"]==name]["ROC_AUC"].values[0]
    ax.plot(fpr, tpr, linewidth=2.2, color=COLORS[name], label=f"{name} (AUC={auc:.3f})")
ax.fill_between([0,1],[0,1], alpha=0.05, color="gray")
ax.set_title("ROC Curves — All Models", fontsize=13, fontweight="bold")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/04_roc_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("   Saved → 04_roc_curves.png")

# ═══════════════════════════════════════════════════════════
# 6. FEATURE IMPORTANCE & HR DASHBOARD
# ═══════════════════════════════════════════════════════════
print("\n[6/7] Feature importance & HR KPI Dashboard …")

rf_model = models["Random Forest"]
feat_imp  = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle("Employee Attrition — Feature Importance & Model Comparison",
             fontsize=14, fontweight="bold")

# Feature importance
ax = axes[0]
top15 = feat_imp.head(15)
cat_color = lambda f: (
    "#EF4444" if f in ["OverTime","DistanceFromHome","NumCompaniesWorked"] else
    "#2563EB" if f in ["MonthlyIncome","JobLevel","StockOptionLevel","PercentSalaryHike"] else
    "#10B981" if f in ["JobSatisfaction","WorkLifeBalance","EnvironmentSatisfaction"] else
    "#F59E0B"
)
colors_fi = [cat_color(f) for f in top15.index[::-1]]
bars = ax.barh(top15.index[::-1], top15.values[::-1], color=colors_fi, edgecolor="white")
ax.set_title("Top 15 Feature Importances (Random Forest)", fontsize=11, fontweight="bold")
ax.set_xlabel("Importance Score")
for bar, val in zip(bars, top15.values[::-1]):
    ax.text(bar.get_width()+0.001, bar.get_y()+bar.get_height()/2,
            f"{val:.3f}", va="center", fontsize=8)
legend_els = [Patch(facecolor="#EF4444", label="Work factors"),
              Patch(facecolor="#2563EB", label="Compensation"),
              Patch(facecolor="#10B981", label="Satisfaction"),
              Patch(facecolor="#F59E0B", label="Demographics")]
ax.legend(handles=legend_els, fontsize=9)
ax.grid(axis="x", alpha=0.3)

# Model comparison radar-style bar
ax2 = axes[1]
metrics = ["Accuracy","Precision","Recall","F1","ROC_AUC"]
x = np.arange(len(results_df))
width = 0.15
for i, (metric, color) in enumerate(zip(metrics,
        ["#3B82F6","#10B981","#EF4444","#F59E0B","#8B5CF6"])):
    vals = results_df[metric].values
    b = ax2.bar(x + i*width, vals, width, label=metric, color=color,
                alpha=0.85, edgecolor="white")
    for bar, v in zip(b, vals):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
                 f"{v:.2f}", ha="center", fontsize=6.5, rotation=90)
ax2.set_xticks(x + width*2)
ax2.set_xticklabels(results_df["Model"], fontsize=10)
ax2.set_ylim(0, 1.12)
ax2.set_title("Model Comparison (All Metrics)", fontsize=11, fontweight="bold")
ax2.set_ylabel("Score")
ax2.legend(fontsize=9)
ax2.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUT}/05_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("   Saved → 05_feature_importance.png")

# --- HR KPI Dashboard ---
fig = plt.figure(figsize=(18, 10))
fig.suptitle("HR Attrition — Executive KPI Dashboard", fontsize=15, fontweight="bold", y=1.01)
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.38)

# KPI cards (top row as text boxes)
kpis = [
    ("Total Employees", f"{N:,}", "#2563EB"),
    ("Attrition Count", f"{df['Attrition'].sum():,}", "#EF4444"),
    ("Attrition Rate", f"{df['Attrition'].mean()*100:.1f}%", "#F59E0B"),
]
for i, (label, val, color) in enumerate(kpis):
    ax_k = fig.add_subplot(gs[0, i])
    ax_k.set_facecolor(color + "18")
    ax_k.text(0.5, 0.6, val, ha="center", va="center",
              fontsize=36, fontweight="bold", color=color,
              transform=ax_k.transAxes)
    ax_k.text(0.5, 0.2, label, ha="center", va="center",
              fontsize=13, color="#374151", transform=ax_k.transAxes)
    ax_k.set_xticks([]); ax_k.set_yticks([])
    for spine in ax_k.spines.values():
        spine.set_edgecolor(color); spine.set_linewidth(2)

# Attrition by Job Role
ax_r = fig.add_subplot(gs[1, :2])
role_att = (df.groupby("JobRole")["Attrition"].mean() * 100).sort_values(ascending=True)
colors_r = ["#EF4444" if v > 25 else "#F59E0B" if v > 15 else "#10B981"
            for v in role_att.values]
bars = ax_r.barh(role_att.index, role_att.values, color=colors_r, edgecolor="white")
ax_r.set_title("Attrition Rate by Job Role  (Red > 25%  |  Orange > 15%  |  Green ≤ 15%)",
               fontsize=11, fontweight="bold")
ax_r.set_xlabel("Attrition Rate (%)")
for bar, val in zip(bars, role_att.values):
    ax_r.text(bar.get_width()+0.2, bar.get_y()+bar.get_height()/2,
              f"{val:.1f}%", va="center", fontsize=9)
ax_r.grid(axis="x", alpha=0.3)

# Risk segmentation
ax_s = fig.add_subplot(gs[1, 2])
gb_model  = models["Gradient Boosting"]
risk_prob = gb_model.predict_proba(X_test)[:, 1]
risk_bins = pd.cut(risk_prob, bins=[0,0.3,0.6,1.0],
                   labels=["Low Risk\n(<30%)", "Medium Risk\n(30–60%)", "High Risk\n(>60%)"])
risk_counts = risk_bins.value_counts().sort_index()
ax_s.bar(risk_counts.index, risk_counts.values,
         color=["#10B981","#F59E0B","#EF4444"], edgecolor="white", width=0.5)
ax_s.set_title("Employee Risk Segmentation\n(Gradient Boosting Predictions)",
               fontsize=11, fontweight="bold")
ax_s.set_ylabel("Number of Employees")
for i, val in enumerate(risk_counts.values):
    ax_s.text(i, val+1, str(val), ha="center", fontweight="bold", fontsize=11)
ax_s.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUT}/06_hr_kpi_dashboard.png", dpi=150, bbox_inches="tight")
plt.close()
print("   Saved → 06_hr_kpi_dashboard.png")

# ═══════════════════════════════════════════════════════════
# 7. ACTIONABLE INSIGHTS REPORT
# ═══════════════════════════════════════════════════════════
print("\n[7/7] Generating insights report …")

best = results_df.loc[results_df["ROC_AUC"].idxmax()]
high_risk = (risk_prob > 0.60).sum()
mid_risk  = ((risk_prob > 0.30) & (risk_prob <= 0.60)).sum()

report = f"""
====================================================================
  EMPLOYEE ATTRITION ANALYSIS — INSIGHTS REPORT
====================================================================

DATASET SUMMARY
---------------
• Total employees analysed : {N:,}
• Employees who left       : {df['Attrition'].sum():,}
• Overall attrition rate   : {df['Attrition'].mean()*100:.1f}%
• Highest-risk department  : {(df.groupby('Department')['Attrition'].mean()*100).idxmax()}
  ({(df.groupby('Department')['Attrition'].mean()*100).max():.1f}% attrition)

TOP ATTRITION DRIVERS (Feature Importance)
-------------------------------------------
{chr(10).join([f"  {i+1}. {feat}: {val:.4f}" for i, (feat, val) in enumerate(feat_imp.head(8).items())])}

MODEL PERFORMANCE SUMMARY
--------------------------
{results_df.to_string(index=False)}

BEST MODEL : {best['Model']}
  → Accuracy  : {best['Accuracy']:.3f}
  → Precision : {best['Precision']:.3f}
  → Recall    : {best['Recall']:.3f}
  → F1 Score  : {best['F1']:.3f}
  → ROC-AUC   : {best['ROC_AUC']:.3f}

EMPLOYEE RISK SEGMENTATION (Test Set: {len(X_test)} employees)
--------------------------------------------------------------
  🔴 High Risk  (>60% prob) : {high_risk} employees
  🟡 Medium Risk (30–60%)   : {mid_risk} employees
  🟢 Low Risk   (<30%)      : {len(X_test)-high_risk-mid_risk} employees

HR RECOMMENDATIONS
------------------
1. OVERTIME POLICY     — Employees doing overtime leave at significantly higher
                         rates. Consider workload redistribution or comp-off policy.
2. LOW SATISFACTION    — Employees with Job Satisfaction ≤ 2 need immediate
                         attention. Suggest skip-level meetings and career roadmaps.
3. INCOME GAPS         — Employees earning < $3,000/month show elevated risk.
                         Benchmark salaries against market rates.
4. EARLY TENURE RISK   — Employees with < 2 years at company are vulnerable.
                         Strengthen onboarding and buddy programs.
5. SINGLE EMPLOYEES    — Higher attrition among single employees. 
                         Consider social engagement and community-building programs.
6. STOCK OPTIONS       — Employees with stock options (level ≥ 2) show lower
                         attrition. Expand ESOP eligibility.
7. DISTANCE FROM HOME  — Employees commuting > 20 km show higher risk.
                         Consider hybrid/remote work policies.

====================================================================
"""

print(report)
with open(f"{OUT}/insights_report.txt", "w") as f:
    f.write(report)
print("   Saved → insights_report.txt")

print("\n  Output files saved to:", OUT)
print("  • hr_dataset.csv              — Synthetic HR dataset")
print("  • 01_eda_dashboard.png        — 9-panel EDA dashboard")
print("  • 02_correlation_heatmap.png  — Feature correlations")
print("  • 03_confusion_matrices.png   — All 3 model confusion matrices")
print("  • 04_roc_curves.png           — ROC curves comparison")
print("  • 05_feature_importance.png   — Feature importance + model metrics")
print("  • 06_hr_kpi_dashboard.png     — Executive HR dashboard")
print("  • model_results.csv           — Full metrics table")
print("  • insights_report.txt         — HR recommendations")
print("=" * 60)
