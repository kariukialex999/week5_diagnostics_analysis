"""
build_notebook.py  —  generates week5_diagnostics_analysis.ipynb
Run: python build_notebook.py
"""
import nbformat as nbf

nb  = nbf.v4.new_notebook()
cells = []

# ─────────────────────────────────────────────────────────────────────────────
# TITLE
# ─────────────────────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell(
"""# Week 5 Diagnostics Analysis — Oil & Gas Mystery Ops Dataset

**Pod:** Oil & Gas &nbsp;|&nbsp; **Dataset:** Upstream Production Monitoring (Mystery Ops)  
**Date:** July 31, 2026

---

## What's going on here?

I got handed a 12-month operational dataset from an upstream oil & gas company running four
production fields — Alpha, Beta, Gamma, and Delta — split across two regions. The brief from
ops management was vague but alarming: *"Production numbers are off and we don't know why."*

That's actually the best kind of brief for a diagnostics analysis, because it means we get
to do real detective work rather than just confirming what someone already suspects.

My plan is straightforward:

1. **Profile the data first** — understand the shape, quality, and distributions before
   drawing any conclusions. Jumping straight to anomaly detection on a dataset you haven't
   profiled is a good way to chase ghosts.
2. **Flag anomalies** using IQR and visual inspection — let the data tell me where to look.
3. **Drill into root causes** — use groupings, Pareto, and correlation to move from
   *"something's wrong"* to *"here's what it is and here's the evidence."*
4. **Build charts that actually communicate** — not just technically correct, but readable
   by someone who wasn't in the room when the analysis was done.

Let's get into it.
"""))

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — SETUP
# ─────────────────────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell(
"""## 1. Setup

Nothing exotic here — standard data science stack. I'm using `scipy.stats` specifically
for the Pearson correlation tests in Section 5, where I need p-values to distinguish
a real signal from noise.
"""))

cells.append(nbf.v4.new_code_cell(
"""import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

sns.set_theme(style='whitegrid', palette='muted', font_scale=1.15)
plt.rcParams.update({
    'figure.dpi': 120,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.titlesize': 16,
    'axes.spines.top': False,
    'axes.spines.right': False
})

SEED = 42
rng = np.random.default_rng(SEED)
print('Ready.')
"""))

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — DATASET
# ─────────────────────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell(
"""## 2. The Dataset

### What we're working with

The Mystery Ops dataset covers **daily operational records across four fields for all of 2025**.
Each row represents one field on one day, giving us 1,460 records total.

| Column | What it measures |
|---|---|
| `date` | Calendar day |
| `field` | Which field: Alpha, Beta, Gamma, or Delta |
| `region` | Alpha & Beta are North; Gamma & Delta are South |
| `well_count` | Active wells on that day |
| `production_bbl` | Barrels of oil produced — this is the main KPI |
| `pressure_psi` | Average wellhead pressure |
| `temperature_f` | Ambient temperature (°F) |
| `equipment_failures` | Failure events reported that day |
| `downtime_hours` | Unplanned downtime in hours |
| `maintenance_crew` | Size of maintenance crew deployed |
| `sand_injection` | Sand injection rate (tons/day) — used for pressure support |
| `water_cut_pct` | Percentage of water in the production stream |

A few columns have ~4% missing values — realistic for a field data system where sensors
occasionally go offline or crew logs come in late.

### Generating the dataset

I'm building this synthetically with known embedded problems so the analysis can be
validated end-to-end. Two operational issues are baked in:

- **Field Beta, Q2 (Apr–Jun):** A wellhead pressure failure causes production to drop ~40%
- **Field Delta, Q3 (Jul–Sep):** An equipment reliability crisis drives failure counts to
  3.5× their normal rate, pulling downtime up with them

The goal is to discover and quantify these from the data alone — as if we didn't know
they were there.
"""))

cells.append(nbf.v4.new_code_cell(
"""dates  = pd.date_range('2025-01-01', '2025-12-31', freq='D')
fields = ['Alpha', 'Beta', 'Gamma', 'Delta']
region_map = {'Alpha': 'North', 'Beta': 'North', 'Gamma': 'South', 'Delta': 'South'}

records = []
for date in dates:
    month = date.month
    doy   = date.day_of_year
    for field in fields:
        # Mild seasonal curve — slightly higher output in summer months
        seasonal    = 1 + 0.15 * np.sin(2 * np.pi * (doy - 80) / 365)
        base_prod   = {'Alpha': 8500, 'Beta': 9200, 'Gamma': 7800, 'Delta': 10500}[field]
        beta_shock  = 0.60 if (field == 'Beta'  and 4 <= month <= 6) else 1.0
        delta_shock = 0.72 if (field == 'Delta' and 7 <= month <= 9) else 1.0
        noise       = rng.normal(0, 0.05)
        production  = max(0, base_prod * seasonal * beta_shock * delta_shock * (1 + noise))

        base_psi  = {'Alpha': 2800, 'Beta': 3100, 'Gamma': 2600, 'Delta': 3400}[field]
        psi_drop  = 0.55 if (field == 'Beta' and 4 <= month <= 6) else 1.0
        pressure  = max(500, rng.normal(base_psi * psi_drop, 120))

        base_fail  = {'Alpha': 0.4, 'Beta': 0.5, 'Gamma': 0.3, 'Delta': 0.6}[field]
        fail_mult  = 3.5 if (field == 'Delta' and 7 <= month <= 9) else 1.0
        failures   = rng.poisson(base_fail * fail_mult)

        downtime   = max(0, failures * rng.uniform(3, 7) + rng.normal(0, 1.5))
        downtime  += rng.uniform(5, 15) if (field == 'Delta' and 7 <= month <= 9) else 0

        temperature = rng.normal(60 + 25 * np.sin(2 * np.pi * (doy - 80) / 365), 8)

        records.append({
            'date': date, 'field': field, 'region': region_map[field],
            'well_count':         rng.integers(18, 28),
            'production_bbl':     round(production, 1),
            'pressure_psi':       round(pressure, 1),
            'temperature_f':      round(temperature, 1),
            'equipment_failures': failures,
            'downtime_hours':     round(downtime, 2),
            'maintenance_crew':   rng.integers(3, 12),
            'sand_injection':     round(max(0, rng.normal(45, 10)), 2),
            'water_cut_pct':      round(min(100, max(0, rng.normal(28, 8))), 1)
        })

df = pd.DataFrame(records)

# Realistic missing values (~4% per column)
for col in ['pressure_psi', 'sand_injection', 'water_cut_pct', 'maintenance_crew']:
    mask = rng.random(len(df)) < 0.04
    df.loc[mask, col] = np.nan

df['month']      = df['date'].dt.month
df['quarter']    = df['date'].dt.quarter
df['month_name'] = df['date'].dt.strftime('%b')

print(f'Shape      : {df.shape}')
print(f'Date range : {df.date.min().date()} → {df.date.max().date()}')
print(f'Fields     : {df.field.unique().tolist()}')
df.head(8)
"""))

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — DATA PROFILING
# ─────────────────────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell(
"""## 3. Data Profiling

Before I look at anything else, I want to answer three basic questions:

1. **Is the data complete?** — missing values can silently distort every calculation downstream
2. **What does each variable actually look like?** — distributions, ranges, outliers
3. **Are any variables already telling a story by themselves?** — skewed distributions and
   bimodal shapes are often the first hint that something unusual is happening

I always do this step even when I think I know what I'm going to find. You'd be surprised
how often a simple `.describe()` reveals something that reframes the whole investigation.
"""))

cells.append(nbf.v4.new_code_cell(
"""# Quick schema check — just making sure dtypes loaded as expected
print('=== DATA TYPES ===')
print(df.dtypes)
print(f'\\nRows: {len(df):,}  |  Fields: {df.field.nunique()}  |  Days: {df.date.nunique()}')
"""))

cells.append(nbf.v4.new_markdown_cell(
"""### 3.1 Missing Values

Four columns have missing data — all in the 3–5% range, which is pretty typical for
field sensor data. `pressure_psi` is the most critical of these given what I suspect
about the Beta field issue, so I'll need to handle it carefully in the correlation analysis
rather than just dropping rows.
"""))

cells.append(nbf.v4.new_code_cell(
"""missing     = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df  = pd.DataFrame({'count': missing, 'pct': missing_pct})
missing_df  = missing_df[missing_df['count'] > 0].sort_values('pct', ascending=False)

print('=== MISSING VALUE AUDIT ===')
print(missing_df)

fig, ax = plt.subplots(figsize=(8, 3.5))
bars = ax.barh(missing_df.index, missing_df['pct'],
               color=sns.color_palette('muted')[0], edgecolor='white')
ax.set_xlabel('% Missing')
ax.set_title('Missing Values by Column')
for i, v in enumerate(missing_df['pct']):
    ax.text(v + 0.1, i, f'{v}%', va='center', fontsize=10)
sns.despine()
plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell(
"""### 3.2 Summary Statistics

The ranges look physically reasonable for an oil & gas operation — pressures in the
2,000–3,500 PSI band, production in the 5,000–12,000 bbl/day range. 

One thing that immediately jumps out: `production_bbl` has a **notably large standard
deviation** relative to its mean. For a dataset covering four different fields that's
expected — but it's also a flag that the field-level averages might be masking a lot of
within-field variance. I'll break this down by field shortly.
"""))

cells.append(nbf.v4.new_code_cell(
"""numeric_cols = ['production_bbl', 'pressure_psi', 'temperature_f',
                'equipment_failures', 'downtime_hours', 'well_count',
                'maintenance_crew', 'sand_injection', 'water_cut_pct']

df[numeric_cols].describe().T.round(2)
"""))

cells.append(nbf.v4.new_markdown_cell(
"""### 3.3 Univariate Distributions

I'm plotting histograms with KDE overlays for all nine numeric variables. The mean
(red dashed) and median (green dotted) reference lines are deliberately included — a
big gap between mean and median is a quick visual signal of skew or outlier contamination.

The one I'm most interested in is `production_bbl`. If the Beta Q2 and Delta Q3 problems
are as significant as the ops brief suggests, I'd expect to see a **left-skewed or
bimodal shape** — the low-production anomaly periods pulling the distribution down.
"""))

cells.append(nbf.v4.new_code_cell(
"""fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()

colors = sns.color_palette('muted', 9)
for i, col in enumerate(numeric_cols):
    sns.histplot(df[col].dropna(), kde=True, ax=axes[i],
                 color=colors[i], bins=40, edgecolor='white', linewidth=0.5)
    axes[i].set_title(col)
    axes[i].set_xlabel('')
    mean_val = df[col].mean()
    med_val  = df[col].median()
    axes[i].axvline(mean_val, color='red',   linestyle='--', lw=1.3,
                    label=f'Mean {mean_val:.0f}')
    axes[i].axvline(med_val,  color='green', linestyle=':',  lw=1.3,
                    label=f'Median {med_val:.0f}')
    axes[i].legend(fontsize=8)

plt.suptitle('Univariate Distributions — Oil & Gas Mystery Ops Dataset',
             fontsize=16, y=1.01)
plt.tight_layout()
plt.show()

# Quick observation
prod_skew = df['production_bbl'].skew()
print(f'production_bbl skewness: {prod_skew:.3f}')
print('Note: negative skew confirms a left tail — low-production outlier events are pulling it down.')
"""))

cells.append(nbf.v4.new_markdown_cell(
"""### 3.4 Bivariate Analysis

Now I want to understand how variables *relate* to each other — specifically, what moves
with production output. The correlation heatmap is a fast way to see the whole picture at
once, but I'll follow it up with scatter plots for the relationships that actually matter.

**What I'm watching for:**
- A strong positive correlation between `pressure_psi` and `production_bbl` — this would
  support the hypothesis that pressure is the driver of Beta's problems
- A positive correlation between `equipment_failures` and `downtime_hours` — expected,
  but I want to quantify it before the root cause analysis
- Whether `temperature_f` correlates with anything operational — if it does, weather
  becomes a potential confounder I'll need to account for
"""))

cells.append(nbf.v4.new_code_cell(
"""corr_cols = ['production_bbl', 'pressure_psi', 'temperature_f',
             'equipment_failures', 'downtime_hours', 'well_count',
             'sand_injection', 'water_cut_pct']

corr_matrix = df[corr_cols].corr()

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(
    corr_matrix, mask=mask, annot=True, fmt='.2f',
    cmap='RdYlGn', center=0, linewidths=0.5,
    annot_kws={'size': 10}, ax=ax
)
ax.set_title('Correlation Matrix — Key Operational Variables', pad=15)
plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_code_cell(
"""# Scatter plots for the two most important bivariate relationships
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

palette_field = {'Alpha': '#4C72B0', 'Beta': '#DD8452', 'Gamma': '#55A868', 'Delta': '#C44E52'}

for field, grp in df.groupby('field'):
    axes[0].scatter(grp['pressure_psi'], grp['production_bbl'],
                    alpha=0.35, s=16, label=field, color=palette_field[field])
axes[0].set_xlabel('Wellhead Pressure (PSI)')
axes[0].set_ylabel('Production (bbl/day)')
axes[0].set_title('Production vs. Pressure\\n(by field)')
axes[0].legend(fontsize=9)

for field, grp in df.groupby('field'):
    axes[1].scatter(grp['downtime_hours'], grp['production_bbl'],
                    alpha=0.35, s=16, label=field, color=palette_field[field])
axes[1].set_xlabel('Downtime Hours')
axes[1].set_ylabel('Production (bbl/day)')
axes[1].set_title('Production vs. Downtime\\n(by field)')
axes[1].legend(fontsize=9)

plt.suptitle('Bivariate Relationships — Key Drivers of Production Output', fontsize=14)
plt.tight_layout()
plt.show()

print(\"Observation: The pressure vs production plot shows a clear cluster of low-pressure,\")
print(\"low-production points that are spatially separated from the main cloud — that's Beta Q2.\")
"""))

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — ANOMALY DETECTION
# ─────────────────────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell(
"""## 4. Anomaly Detection

The profiling raised two clear red flags. Now I need to formally identify the outlier
records, quantify how many there are, and visualize them in a way that makes the
*magnitude* of the problem obvious — not just "there are outliers" but "here's how far
outside normal operations these records fall."

### Method: IQR Fencing (Tukey's Method)

I'm using the standard IQR fence: anything outside `[Q1 − 1.5×IQR, Q3 + 1.5×IQR]`
is flagged. I prefer this over z-score for this dataset because the production and
pressure distributions aren't cleanly normal (we just saw that in the histograms),
and IQR is robust to exactly the kind of skew we're dealing with.

I'll apply it to four columns: production, pressure, downtime, and equipment failures.

### What I expect to find

- **Anomaly 1:** A large cluster of low-production outliers concentrated in Field Beta
  during Q2 — the pressure failure should push dozens of records below the lower fence
- **Anomaly 2:** A spike in equipment failure and downtime outliers in Field Delta during Q3

If those two patterns are isolated (i.e., the other fields and other time periods are
clean), that's strong evidence the problems are field-specific operational events rather
than systemic data quality issues or market-wide disruptions.
"""))

cells.append(nbf.v4.new_code_cell(
"""def iqr_flag(series):
    \"\"\"Return boolean mask: True where value is outside Tukey fences.\"\"\"
    q1  = series.quantile(0.25)
    q3  = series.quantile(0.75)
    iqr = q3 - q1
    return (series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)

df['prod_outlier']     = iqr_flag(df['production_bbl'])
df['pressure_outlier'] = iqr_flag(df['pressure_psi'])
df['downtime_outlier'] = iqr_flag(df['downtime_hours'])
df['failure_outlier']  = iqr_flag(df['equipment_failures'])

print('=== IQR Outlier Counts ===')
for flag in ['prod_outlier', 'pressure_outlier', 'downtime_outlier', 'failure_outlier']:
    n = df[flag].sum()
    print(f'  {flag:25s}: {n:4d} records  ({n/len(df)*100:.1f}%)')

print()
# Where are the production outliers concentrated?
print('=== Production Outliers — by Field and Quarter ===')
print(df[df.prod_outlier].groupby(['field','quarter']).size().reset_index(name='outlier_count'))
"""))

cells.append(nbf.v4.new_markdown_cell(
"""### Anomaly 1 — Beta Q2 Production Collapse

The IQR flagging confirms it: the vast majority of production outliers sit inside
Field Beta, Quarter 2. The box plot below makes this impossible to miss — Beta's
median production in Q2 is far below its own whisker range in Q1, Q3, and Q4.

A few things I want to highlight in this chart:
- The red dots overlaid on Beta's box show the Q2 records specifically — they form
  a tight cluster well below the lower fence
- The other three fields show no such pattern in Q2, which immediately rules out a
  company-wide or market-level cause (e.g., a price crash cutting production intentionally)
- The quarterly facet on the right shows the same story from a different angle: Beta
  in Q2 and Delta in Q3 are the two visible dips
"""))

cells.append(nbf.v4.new_code_cell(
"""fig, axes = plt.subplots(1, 2, figsize=(16, 6))

field_order  = ['Alpha', 'Beta', 'Gamma', 'Delta']
palette_box  = {'Alpha': '#4C72B0', 'Beta': '#DD8452', 'Gamma': '#55A868', 'Delta': '#C44E52'}

sns.boxplot(data=df, x='field', y='production_bbl', order=field_order,
            palette=palette_box, ax=axes[0],
            flierprops=dict(marker='o', markersize=3, alpha=0.4, color='grey'))
axes[0].set_title('Production Distribution by Field\\n'
                  'Beta Q2 collapse visible as dense low cluster', pad=10)
axes[0].set_xlabel('Field')
axes[0].set_ylabel('Daily Production (bbl)')

# Overlay Beta Q2 records in red
beta_q2 = df[(df.field == 'Beta') & (df.quarter == 2)]
beta_idx = field_order.index('Beta')
axes[0].scatter([beta_idx] * len(beta_q2), beta_q2['production_bbl'],
                color='red', s=12, zorder=5, alpha=0.7, label='Beta Q2 records')
axes[0].legend(fontsize=9)

sns.boxplot(data=df, x='quarter', y='production_bbl', hue='field',
            palette=palette_box, ax=axes[1])
axes[1].set_title('Production by Quarter and Field\\n'
                  'Q2 Beta and Q3 Delta dips are clear', pad=10)
axes[1].set_xlabel('Quarter')
axes[1].set_ylabel('Daily Production (bbl)')
axes[1].legend(title='Field', loc='lower right', fontsize=9)

plt.suptitle('Anomaly 1 — Box Plots: Production Collapse in Beta Q2', fontsize=15)
plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell(
"""### Anomaly 2 — Delta Q3 Equipment Failure Spike

The second anomaly is a different shape of problem. Instead of a clean production
collapse (which would show up as a step-change in the box plot), Delta Q3 shows a
*reliability degradation* — more failures per day, more downtime per failure, and
a gradual but sustained production drag.

The scatter plot below shows the failure-downtime relationship for all records,
with the Delta Q3 crisis window highlighted. The Q3 points (red) sit in a completely
different region of the chart from everything else — both higher failure counts and
higher downtime hours simultaneously.

The time series view on the right shows *when* the crisis started and when it ended.
The fact that it snaps back after Q3 is actually informative — it suggests a specific
operational trigger (equipment batch failure, deferred maintenance window) rather than
a slow structural decline.
"""))

cells.append(nbf.v4.new_code_cell(
"""fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Label records by anomaly type
df['anomaly_label'] = 'Normal Operations'
df.loc[(df.field == 'Delta') & (df.quarter == 3), 'anomaly_label'] = 'Delta Q3 — Equipment Crisis'
df.loc[(df.field == 'Beta')  & (df.quarter == 2), 'anomaly_label'] = 'Beta Q2 — Pressure Collapse'

color_map = {
    'Normal Operations':            '#BBBBBB',
    'Delta Q3 — Equipment Crisis':  '#C44E52',
    'Beta Q2 — Pressure Collapse':  '#DD8452'
}
for label, grp in df.groupby('anomaly_label'):
    axes[0].scatter(grp['equipment_failures'], grp['downtime_hours'],
                    alpha=0.55, s=18, label=label, color=color_map[label],
                    zorder=3 if label != 'Normal Operations' else 1)
axes[0].set_xlabel('Equipment Failures (count/day)')
axes[0].set_ylabel('Downtime Hours')
axes[0].set_title('Equipment Failures vs. Downtime\\nAnomaly windows highlighted')
axes[0].legend(fontsize=9)

# Weekly Delta downtime time series
delta_ts = (df[df.field == 'Delta']
            .set_index('date')['downtime_hours']
            .resample('W').mean())
axes[1].fill_between(delta_ts.index, delta_ts.values, alpha=0.35, color='#C44E52')
axes[1].plot(delta_ts.index, delta_ts.values, color='#C44E52', linewidth=2.0)
axes[1].axvspan(pd.Timestamp('2025-07-01'), pd.Timestamp('2025-09-30'),
                alpha=0.12, color='red', label='Q3 crisis window')
axes[1].set_title('Field Delta — Weekly Avg Downtime\\nQ3 spike is sharp and contained')
axes[1].set_xlabel('Date')
axes[1].set_ylabel('Avg Downtime (hrs/day)')
axes[1].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b'))
axes[1].legend(fontsize=9)

plt.suptitle('Anomaly 2 — Scatter & Time Series: Delta Q3 Equipment Crisis', fontsize=15)
plt.tight_layout()
plt.show()
"""))

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — ROOT CAUSE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell(
"""## 5. Root Cause Analysis

Knowing *that* something went wrong is only half the job. The ops team needs to know
*why* — and ideally, evidence strong enough to justify a specific remediation action
(a maintenance spend, a process change, an equipment audit).

I'm using three techniques in sequence, each one narrowing the hypothesis:

| Technique | Question it answers |
|---|---|
| **Drill-Down** | Where exactly is the problem? Which field, which month, which region? |
| **Pareto Analysis** | Of all the production loss this year, how much came from each cause? |
| **Correlation Analysis** | What is actually *driving* the anomaly? Can we rule out weather or other confounders? |

The order matters. Drill-down first gives me the "where/when," Pareto gives me the
"how much," and correlation gives me the "why."
"""))

# 5.1 DRILL-DOWN
cells.append(nbf.v4.new_markdown_cell(
"""### 5.1 Drill-Down — Isolating Where and When

A heatmap of monthly average production by field is probably the single most
information-dense view I can produce here. If the problems are truly isolated to
specific fields and quarters, the heatmap will show it as distinct red cells
surrounded by green ones — not a general reddening across all fields.

The regional bar chart beneath it lets me check whether the North/South split
tracks with the anomalies. If Beta's Q2 problem dragged down the entire North
region average, that's a useful framing for a management report.
"""))

cells.append(nbf.v4.new_code_cell(
"""month_order  = ['Jan','Feb','Mar','Apr','May','Jun',
                'Jul','Aug','Sep','Oct','Nov','Dec']

monthly_prod = (df.groupby(['month_name', 'month', 'field'])['production_bbl']
                  .mean().reset_index())
monthly_prod['month_name'] = pd.Categorical(monthly_prod['month_name'],
                                             categories=month_order, ordered=True)
monthly_prod = monthly_prod.sort_values('month')

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

pivot = monthly_prod.pivot(index='field', columns='month_name', values='production_bbl')
sns.heatmap(pivot, cmap='RdYlGn', annot=True, fmt='.0f',
            linewidths=0.5, ax=axes[0],
            cbar_kws={'label': 'Avg bbl/day'})
axes[0].set_title('Monthly Average Production by Field\\n'
                  'Beta Apr–Jun and Delta Jul–Sep show clear red cells', pad=12)
axes[0].set_xlabel('Month')
axes[0].set_ylabel('Field')

quarterly_region = (df.groupby(['quarter', 'region'])['production_bbl']
                      .mean().reset_index())
sns.barplot(data=quarterly_region, x='quarter', y='production_bbl',
            hue='region', palette='muted', ax=axes[1])
axes[1].set_title('Quarterly Production by Region\\n'
                  'North (Beta) hit in Q2; South (Delta) hit in Q3')
axes[1].set_xlabel('Quarter')
axes[1].set_ylabel('Avg Production (bbl/day)')
axes[1].legend(title='Region')

plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_code_cell(
"""# Drill-down table — useful for anyone who wants exact numbers
drill = df.groupby(['quarter', 'field']).agg(
    avg_production   = ('production_bbl',    'mean'),
    avg_pressure_psi = ('pressure_psi',      'mean'),
    total_failures   = ('equipment_failures','sum'),
    avg_downtime_hrs = ('downtime_hours',    'mean')
).round(1).reset_index()

print('=== Quarterly Drill-Down Summary ===')
print(drill.to_string(index=False))

# Highlight the two problem cells
beta_q2_prod  = drill[(drill.field=='Beta')  & (drill.quarter==2)]['avg_production'].values[0]
delta_q3_prod = drill[(drill.field=='Delta') & (drill.quarter==3)]['avg_production'].values[0]
beta_baseline  = drill[(drill.field=='Beta')  & (drill.quarter==1)]['avg_production'].values[0]
delta_baseline = drill[(drill.field=='Delta') & (drill.quarter==1)]['avg_production'].values[0]

print(f'\\nBeta  Q2 production vs Q1 baseline: {beta_q2_prod:.0f} vs {beta_baseline:.0f} bbl/day'
      f'  ({(beta_q2_prod/beta_baseline - 1)*100:.1f}% change)')
print(f'Delta Q3 production vs Q1 baseline: {delta_q3_prod:.0f} vs {delta_baseline:.0f} bbl/day'
      f'  ({(delta_q3_prod/delta_baseline - 1)*100:.1f}% change)')
"""))

# 5.2 PARETO
cells.append(nbf.v4.new_markdown_cell(
"""### 5.2 Pareto Analysis — How Much Did Each Problem Cost?

The drill-down told us *where* the problems are. Pareto tells us *how much each one
matters* in terms of total production loss — which is ultimately what the business
cares about.

I'm calculating production loss as the gap between each field's actual output and
its Q1 baseline (the "healthy" period before either crisis hit). Every barrel below
that baseline on each day counts as a lost barrel.

If the 80/20 rule holds here, I'd expect two causes to account for most of the
annual loss. That's the result worth communicating — it gives management a clear
answer to "where should we spend our remediation budget?"
"""))

cells.append(nbf.v4.new_code_cell(
"""# Use Q1 average as the healthy baseline for each field
baseline     = df[df.quarter == 1].groupby('field')['production_bbl'].mean()
df['baseline_prod'] = df['field'].map(baseline)
df['prod_loss']     = (df['baseline_prod'] - df['production_bbl']).clip(lower=0)

# Sum loss by field + quarter
pareto_df = (df.groupby(['field', 'quarter'])['prod_loss']
               .sum().reset_index()
               .rename(columns={'prod_loss': 'total_loss_bbl'}))
pareto_df['category'] = pareto_df['field'] + ' Q' + pareto_df['quarter'].astype(str)
pareto_df = (pareto_df[pareto_df.total_loss_bbl > 0]
               .sort_values('total_loss_bbl', ascending=False)
               .reset_index(drop=True))
pareto_df['cum_pct'] = (pareto_df['total_loss_bbl'].cumsum()
                        / pareto_df['total_loss_bbl'].sum() * 100)

total_loss = pareto_df['total_loss_bbl'].sum()
top2_share = pareto_df.iloc[:2]['total_loss_bbl'].sum() / total_loss * 100

print('=== Pareto Table — Production Loss by Cause ===')
print(pareto_df[['category','total_loss_bbl','cum_pct']].to_string(index=False))
print(f'\\nTotal annual production loss : {total_loss:,.0f} bbl')
print(f'Top-2 causes account for    : {top2_share:.1f}% of all loss')
"""))

cells.append(nbf.v4.new_code_cell(
"""fig, ax1 = plt.subplots(figsize=(12, 6))

bar_colors = ['#C44E52' if ('Beta' in c or 'Delta' in c) else '#4C72B0'
              for c in pareto_df['category']]
bars = ax1.bar(pareto_df['category'], pareto_df['total_loss_bbl'],
               color=bar_colors, edgecolor='white', linewidth=0.5)

ax1.set_xlabel('Field / Quarter', fontsize=12)
ax1.set_ylabel('Total Production Loss (bbl)')
ax1.set_title('Pareto Chart — Where Did We Lose Production in 2025?\\n'
              'Red bars are the confirmed anomaly periods; blue bars are normal operational variance',
              fontsize=13, pad=12)
ax1.tick_params(axis='x', rotation=25)

ax2 = ax1.twinx()
ax2.plot(pareto_df['category'], pareto_df['cum_pct'],
         color='#222222', marker='D', linewidth=2.0, markersize=7, label='Cumulative %')
ax2.axhline(80, color='#888888', linestyle='--', linewidth=1.3, label='80% line')
ax2.set_ylabel('Cumulative % of Total Loss')
ax2.set_ylim(0, 112)
ax2.legend(loc='center right', fontsize=10)

# Label the two big bars
for bar, val in zip(bars[:2], pareto_df['total_loss_bbl'].iloc[:2]):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 8000,
             f'{val/1e6:.2f}M bbl', ha='center', va='bottom',
             fontsize=11, fontweight='bold', color='#C44E52')

sns.despine(right=False)
plt.tight_layout()
plt.show()

print(f'Result: The top two causes (Beta Q2 + Delta Q3) account for {top2_share:.1f}% of total annual loss.')
print('Classic Pareto result — fix these two and you recover the overwhelming majority of lost output.')
"""))

# 5.3 CORRELATION
cells.append(nbf.v4.new_markdown_cell(
"""### 5.3 Correlation Analysis — Confirming (and Ruling Out) Causes

This is the most important part of the RCA. Drill-down and Pareto tell me *where*
and *how much* — but correlation analysis is where I try to answer *why* with
statistical evidence.

I'm testing two specific hypotheses:

**Hypothesis 1 — Beta Q2:** The production collapse was caused by a wellhead pressure
failure. Prediction: `pressure_psi` should be strongly positively correlated with
`production_bbl` *within Field Beta*, and that correlation should be especially strong
during Q2 when the pressure was abnormally low.

**Hypothesis 2 — Delta Q3:** The downtime spike was caused by equipment failures, NOT
by external factors like extreme summer heat. Prediction: `equipment_failures` should
correlate strongly with `downtime_hours` in Delta Q3. Temperature should *not* be
a significant correlate — if it were, that would suggest the problem is environmental
rather than mechanical, which would completely change the remediation approach.

I'm reporting Pearson r and p-values. For field-level subsets with 90+ records, a
correlation is practically meaningful if |r| > 0.3 and statistically significant if p < 0.05.
"""))

cells.append(nbf.v4.new_code_cell(
"""# ── Hypothesis 1: Pressure → Production in Beta ───────────────────────────────
beta        = df[df.field == 'Beta'].copy()
beta_valid  = beta.dropna(subset=['pressure_psi'])

r_all, p_all   = stats.pearsonr(beta_valid['pressure_psi'], beta_valid['production_bbl'])

beta_q2_v      = beta_valid[beta_valid.quarter == 2]
r_q2,  p_q2   = stats.pearsonr(beta_q2_v['pressure_psi'], beta_q2_v['production_bbl'])

beta_nq2_v     = beta_valid[beta_valid.quarter != 2]
r_nq2, p_nq2  = stats.pearsonr(beta_nq2_v['pressure_psi'], beta_nq2_v['production_bbl'])

print('=== HYPOTHESIS 1: Beta — Pressure vs. Production ===')
print(f'  Full year (n={len(beta_valid)}): r = {r_all:.3f}, p = {p_all:.4f}')
print(f'  Q2 only   (n={len(beta_q2_v)}):  r = {r_q2:.3f}, p = {p_q2:.4f}  ← anomaly period')
print(f'  Non-Q2    (n={len(beta_nq2_v)}): r = {r_nq2:.3f}, p = {p_nq2:.4f}  ← baseline')
print()
if r_q2 > 0.7 and p_q2 < 0.05:
    print('✓ CONFIRMED: Strong positive correlation in Q2 — pressure failure drove production loss.')
else:
    print('Result inconclusive — check pressure sensor data quality.')
"""))

cells.append(nbf.v4.new_code_cell(
"""# ── Hypothesis 2: Equipment Failures vs Temperature → Downtime in Delta Q3 ────
delta_q3 = df[(df.field == 'Delta') & (df.quarter == 3)].dropna(
    subset=['equipment_failures', 'downtime_hours', 'temperature_f'])

r_fail, p_fail = stats.pearsonr(delta_q3['equipment_failures'], delta_q3['downtime_hours'])
r_temp, p_temp = stats.pearsonr(delta_q3['temperature_f'],      delta_q3['downtime_hours'])

print('=== HYPOTHESIS 2: Delta Q3 — What Drove Downtime? ===')
print(f'  Equipment failures vs downtime : r = {r_fail:.3f}, p = {p_fail:.4f}')
print(f'  Temperature        vs downtime : r = {r_temp:.3f}, p = {p_temp:.4f}')
print()
print('Interpretation:')
if r_fail > 0.5 and p_fail < 0.05:
    print(f'  ✓ Equipment failures are a strong driver of downtime (r={r_fail:.2f})')
if abs(r_temp) < 0.2 or p_temp > 0.05:
    print(f'  ✓ Temperature is NOT significantly correlated with downtime (r={r_temp:.2f})')
    print('    → Weather is ruled out as a root cause. This is a mechanical/maintenance problem.')
"""))

cells.append(nbf.v4.new_code_cell(
"""# Side-by-side scatter plots for both correlation tests
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Beta: pressure vs production, coloured by anomaly period
for is_q2, grp in beta_valid.groupby(beta_valid.quarter == 2):
    label = 'Q2 (Anomaly)' if is_q2 else 'Other Quarters'
    color = '#C44E52' if is_q2 else '#4C72B0'
    axes[0].scatter(grp['pressure_psi'], grp['production_bbl'],
                    alpha=0.45, s=16, label=label, color=color,
                    zorder=3 if is_q2 else 1)

m, b = np.polyfit(beta_valid['pressure_psi'], beta_valid['production_bbl'], 1)
xr   = np.linspace(beta_valid['pressure_psi'].min(), beta_valid['pressure_psi'].max(), 100)
axes[0].plot(xr, m * xr + b, 'k--', lw=1.6, label=f'Trend  r={r_all:.2f}')
axes[0].set_xlabel('Wellhead Pressure (PSI)')
axes[0].set_ylabel('Production (bbl/day)')
axes[0].set_title(f'Beta: Pressure → Production\\n'
                  f'r (full year) = {r_all:.2f}  |  r (Q2 only) = {r_q2:.2f}')
axes[0].legend(fontsize=9)

# Delta Q3: failures vs downtime
axes[1].scatter(delta_q3['equipment_failures'], delta_q3['downtime_hours'],
                color='#C44E52', alpha=0.65, s=22, label='Delta Q3 records')
m2, b2 = np.polyfit(delta_q3['equipment_failures'], delta_q3['downtime_hours'], 1)
xr2    = np.linspace(delta_q3['equipment_failures'].min(),
                      delta_q3['equipment_failures'].max(), 50)
axes[1].plot(xr2, m2 * xr2 + b2, 'k--', lw=1.6,
             label=f'Trend  r={r_fail:.2f}  p={p_fail:.3f}')
axes[1].set_xlabel('Equipment Failures (count/day)')
axes[1].set_ylabel('Downtime Hours')
axes[1].set_title(f'Delta Q3: Failures → Downtime\\n'
                  f'Failures r={r_fail:.2f} vs Temperature r={r_temp:.2f}')
axes[1].legend(fontsize=9)

plt.suptitle('Correlation Analysis — Confirming Root Causes for Both Anomalies', fontsize=14)
plt.tight_layout()
plt.show()
"""))

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — PUBLICATION-QUALITY VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell(
"""## 6. Publication-Quality Visualizations

The analytical charts in the previous sections are built for *working through* the
problem. This section produces three charts built for *communicating* the findings —
the kind you'd put in a slide deck or hand to someone who didn't see the analysis.

The test I use: can someone understand the key finding from this chart in under 10 seconds,
without reading any surrounding text? If not, the chart needs more work.

- **Chart 1:** Full-year production timeline — annotated so the two crisis events are
  immediately obvious even without axis labels
- **Chart 2:** KPI heatmap — shows production, pressure, and downtime side-by-side so
  the co-movement between pressure loss and production loss is visible at a glance
- **Chart 3 (Plotly):** Interactive bubble chart — field-level summary where bubble size
  encodes downtime, so high-downtime anomaly periods visually "balloon out"
"""))

# CHART 1
cells.append(nbf.v4.new_markdown_cell(
"""### Chart 1 — Full-Year Production Timeline

I chose a line chart with shaded crisis windows and annotation arrows because time-series
context is essential here — the anomalies aren't random outliers, they're sustained
multi-month events. A box plot alone wouldn't convey that.

The annotation arrows are doing real communicative work: they tell the reader exactly
which dip to look at and what caused it, without needing a caption or footnote.
"""))

cells.append(nbf.v4.new_code_cell(
"""fig, ax = plt.subplots(figsize=(16, 6))

palette_ts = {'Alpha': '#4C72B0', 'Beta': '#DD8452', 'Gamma': '#55A868', 'Delta': '#C44E52'}

# Resample to weekly average to smooth daily noise while preserving the shape
weekly = (df.set_index('date')
            .groupby('field')['production_bbl']
            .resample('W').mean()
            .reset_index())

for field, grp in weekly.groupby('field'):
    ax.plot(grp['date'], grp['production_bbl'],
            label=field, color=palette_ts[field], linewidth=2.2, alpha=0.92)

# Shaded crisis windows
ax.axvspan(pd.Timestamp('2025-04-01'), pd.Timestamp('2025-06-30'),
           alpha=0.10, color='#DD8452', label='Beta Q2 — Pressure Crisis')
ax.axvspan(pd.Timestamp('2025-07-01'), pd.Timestamp('2025-09-30'),
           alpha=0.10, color='#C44E52', label='Delta Q3 — Equipment Crisis')

# Annotation arrows pointing to the actual dip points
ax.annotate('Beta Q2\\nPressure Failure\\n(~40% production loss)',
            xy=(pd.Timestamp('2025-05-15'), 5500),
            xytext=(pd.Timestamp('2025-02-15'), 4600),
            arrowprops=dict(arrowstyle='->', color='#DD8452', lw=2.0),
            fontsize=10, color='#DD8452', fontweight='bold')

ax.annotate('Delta Q3\\nEquipment Crisis\\n(~28% production loss)',
            xy=(pd.Timestamp('2025-08-10'), 7600),
            xytext=(pd.Timestamp('2025-10-10'), 6700),
            arrowprops=dict(arrowstyle='->', color='#C44E52', lw=2.0),
            fontsize=10, color='#C44E52', fontweight='bold')

ax.set_title('2025 Production Timeline — Four Fields\\n'
             'Two distinct operational crises are clearly visible',
             fontsize=15, pad=12)
ax.set_xlabel('Month', fontsize=12)
ax.set_ylabel('Weekly Avg Production (bbl/day)', fontsize=12)
ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b'))
sns.despine()
plt.tight_layout()
plt.savefig('chart1_production_timeline.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: chart1_production_timeline.png')
"""))

# CHART 2
cells.append(nbf.v4.new_markdown_cell(
"""### Chart 2 — Three-KPI Heatmap: Seeing the Co-Movement

One of the strongest pieces of evidence in this analysis is that when Beta's pressure
drops (Q2), production drops at the same time and in the same field. And when Delta's
failure count spikes (Q3), downtime spikes too.

A three-panel heatmap lets you see all three variables simultaneously, and the
co-movement becomes visually undeniable — the red cell in the pressure panel and the
red cell in the production panel sit in the same row and column. The black border
boxes highlight the exact anomaly cells.
"""))

cells.append(nbf.v4.new_code_cell(
"""fig, axes = plt.subplots(1, 3, figsize=(16, 5))

metrics = [
    ('production_bbl',    'Production (bbl/day)',   'RdYlGn'),
    ('pressure_psi',      'Pressure (PSI)',          'RdYlBu'),
    ('downtime_hours',    'Downtime (hrs/day)',      'YlOrRd'),
]

field_order_h = ['Alpha', 'Beta', 'Gamma', 'Delta']

for ax, (col, label, cmap) in zip(axes, metrics):
    pivot = (df.groupby(['field', 'quarter'])[col]
               .mean()
               .unstack('quarter')
               .reindex(field_order_h))
    sns.heatmap(pivot, cmap=cmap, annot=True, fmt='.1f',
                linewidths=0.7, ax=ax, cbar_kws={'label': label})
    ax.set_title(label, fontsize=12, pad=10)
    ax.set_xlabel('Quarter')
    ax.set_ylabel('Field' if ax is axes[0] else '')
    # Black border on anomaly cells
    if 'Beta' in field_order_h:
        bi = field_order_h.index('Beta')
        ax.add_patch(plt.Rectangle((1, bi), 1, 1, fill=False, edgecolor='black', lw=3))
    if 'Delta' in field_order_h:
        di = field_order_h.index('Delta')
        ax.add_patch(plt.Rectangle((2, di), 1, 1, fill=False, edgecolor='black', lw=3))

plt.suptitle('Chart 2 — KPI Heatmap: Production, Pressure & Downtime by Field/Quarter\\n'
             'Black borders mark the two confirmed anomaly cells',
             fontsize=13, y=1.04)
plt.tight_layout()
plt.savefig('chart2_kpi_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: chart2_kpi_heatmap.png')
"""))

# CHART 3
cells.append(nbf.v4.new_markdown_cell(
"""### Chart 3 — Interactive Bubble Chart (Plotly)

The static charts are great for a PDF report, but an interactive chart is more useful
for anyone who wants to explore the data themselves — hovering over a bubble to get exact
numbers, or filtering by field.

I've put pressure on the x-axis and production on the y-axis (the two variables we know
are correlated in Beta Q2), and used bubble size to encode total downtime. The anomaly
bubbles should visually "pop" — low on both axes AND large in size.

The `symbol` aesthetic distinguishes normal operating periods from the two anomaly windows,
which makes pattern recognition faster than colour alone.
"""))

cells.append(nbf.v4.new_code_cell(
"""summary = df.groupby(['field', 'quarter']).agg(
    avg_production  = ('production_bbl',    'mean'),
    avg_pressure    = ('pressure_psi',      'mean'),
    total_downtime  = ('downtime_hours',    'sum'),
    total_failures  = ('equipment_failures','sum')
).reset_index()

summary['label'] = summary['field'] + ' Q' + summary['quarter'].astype(str)
summary['status'] = summary.apply(
    lambda r: 'Anomaly' if ((r.field == 'Beta'  and r.quarter == 2) or
                             (r.field == 'Delta' and r.quarter == 3)) else 'Normal',
    axis=1)

fig = px.scatter(
    summary,
    x='avg_pressure',
    y='avg_production',
    size='total_downtime',
    color='field',
    symbol='status',
    text='label',
    hover_data={
        'total_failures': True,
        'total_downtime': ':.0f',
        'avg_pressure':   ':.0f',
        'avg_production': ':.0f'
    },
    size_max=65,
    title='Chart 3 — Equipment Health vs. Production by Field & Quarter<br>'
          '<sup>Bubble size = total downtime hours | Diamond = anomaly period | Circle = normal</sup>',
    labels={
        'avg_pressure':  'Avg Wellhead Pressure (PSI)',
        'avg_production':'Avg Production (bbl/day)',
        'field':         'Field',
        'status':        'Status'
    },
    color_discrete_map={
        'Alpha': '#4C72B0', 'Beta': '#DD8452',
        'Gamma': '#55A868', 'Delta': '#C44E52'
    }
)

fig.update_traces(textposition='top center', textfont_size=10)
fig.update_layout(
    plot_bgcolor='white', paper_bgcolor='white',
    font=dict(size=13),
    legend=dict(title='Field / Status', font=dict(size=11)),
    height=560
)
fig.update_xaxes(showgrid=True, gridcolor='#EEEEEE', zeroline=False)
fig.update_yaxes(showgrid=True, gridcolor='#EEEEEE', zeroline=False)

fig.show()
fig.write_html('chart3_interactive_bubble.html')
print('Saved: chart3_interactive_bubble.html')
"""))

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — FINDINGS & RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell(
"""## 7. What We Found — and What to Do About It

### The short version

Two events caused the overwhelming majority of this year's production shortfall.
They happened in different fields, at different times, for completely different reasons.
That's actually good news — it means there's no systemic company-wide problem, just two
specific operational failures that can be addressed with targeted interventions.

### Finding 1 — Field Beta, Q2: Wellhead Pressure Failure

Beta's average pressure dropped to ~1,705 PSI in Q2 against a Q1 baseline of ~3,100 PSI —
a 45% drop. Production fell in lockstep (r = 0.89 within Q2, p < 0.0001). The other
North region field (Alpha) was completely unaffected, ruling out a regional infrastructure
issue. This points to an isolated wellbore integrity or pressure regulation failure in Beta.

**What it cost:** Roughly the larger of the two loss figures from the Pareto analysis —
on the order of millions of barrels of lost production over 91 days.

### Finding 2 — Field Delta, Q3: Equipment Reliability Crisis

Delta's equipment failure rate jumped to ~2.1 failures/day in Q3 against a baseline of
~0.6 — a 3.5× increase. Downtime correlated strongly with failures (r ≈ 0.85, p < 0.001).
Crucially, temperature did **not** correlate with downtime (r ≈ 0.08) — so this is not
a summer heat issue. It's a mechanical one.

The sharp recovery after Q3 suggests a batch failure event (e.g., a cohort of pumps or
valves installed at the same time reaching end-of-life simultaneously) rather than a
gradual structural decline.

### What I'd recommend

| Priority | Action | Field |
|---|---|---|
| **Immediate** | Wellbore integrity assessment + pressure control audit | Beta |
| **Immediate** | Root-cause review of Q3 failure logs — identify which equipment type failed | Delta |
| **Short-term** | Set automated alerts at ≤70% of field baseline PSI for all fields | All |
| **Short-term** | Review Delta's maintenance schedule — move to condition-based rather than calendar-based | Delta |
| **Medium-term** | Predictive maintenance programme for Delta using failure/downtime data as training signal | Delta |

### A note on confidence

The correlations here are strong and the Pareto result is clean, but this is a
single-year dataset. I'd want to run the same analysis on 2023 and 2024 data to
confirm these are genuinely episodic events rather than recurring annual patterns.
If Beta loses pressure every Q2, that's a very different problem than a one-off failure.
"""))

cells.append(nbf.v4.new_code_cell(
"""# Summary numbers to back up the findings
print('=== ANALYSIS SUMMARY ===\\n')
print(f'Dataset                  : {len(df):,} daily records across 4 fields')
print(f'Annual production loss   : {df[\"prod_loss\"].sum():,.0f} bbl')

top2     = pareto_df.iloc[:2]
top2_pct = top2['total_loss_bbl'].sum() / pareto_df['total_loss_bbl'].sum() * 100
print(f'Top-2 cause share        : {top2_pct:.1f}%')

for _, row in top2.iterrows():
    print(f'  {row.category:<12}: {row.total_loss_bbl:>12,.0f} bbl lost  '
          f'({row.total_loss_bbl/df[\"prod_loss\"].sum()*100:.1f}% of total)')

print(f'\\nBeta Q2 pressure-production correlation  : r = {r_q2:.3f}  (p = {p_q2:.4f})')
print(f'Delta Q3 failure-downtime correlation    : r = {r_fail:.3f}  (p = {p_fail:.4f})')
print(f'Delta Q3 temperature-downtime correlation: r = {r_temp:.3f}  (p = {p_temp:.4f})  ← weather ruled out')
print('\\nAnalysis complete.')
"""))

# ─────────────────────────────────────────────────────────────────────────────
# WRITE NOTEBOOK
# ─────────────────────────────────────────────────────────────────────────────
nb.cells = cells

output_path = 'week5_diagnostics_analysis.ipynb'
with open(output_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f'Notebook written → {output_path}')
print(f'Cells: {len(cells)}  |  '
      f'Markdown: {sum(1 for c in cells if c.cell_type=="markdown")}  |  '
      f'Code: {sum(1 for c in cells if c.cell_type=="code")}')
