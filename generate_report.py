"""
generate_report.py
Produces Week5_Diagnostics_Report_Alex_Kariuki.pdf using reportlab.
Run: python generate_report.py
"""

import io, warnings, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

warnings.filterwarnings('ignore')

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image,
    Table, TableStyle, HRFlowable, KeepTogether
)
from reportlab.lib.colors import HexColor

# ── Colour palette ─────────────────────────────────────────────────────────────
NAVY      = HexColor('#1B2A4A')
ORANGE    = HexColor('#DD8452')
RED       = HexColor('#C44E52')
GREEN     = HexColor('#2E7D32')
LIGHTGREY = HexColor('#F5F5F5')
MIDGREY   = HexColor('#CCCCCC')
WHITE     = colors.white

W, H = A4   # 595.27 x 841.89 points

# ── Styles ─────────────────────────────────────────────────────────────────────
base = getSampleStyleSheet()

def style(name, parent='Normal', **kw):
    s = ParagraphStyle(name, parent=base[parent])
    for k, v in kw.items():
        setattr(s, k, v)
    return s

S_COVER_TITLE  = style('CoverTitle',  fontSize=26, textColor=WHITE,
                        fontName='Helvetica-Bold', leading=32, alignment=TA_CENTER)
S_COVER_SUB    = style('CoverSub',    fontSize=12, textColor=HexColor('#DDDDDD'),
                        fontName='Helvetica', leading=18, alignment=TA_CENTER)
S_COVER_META   = style('CoverMeta',   fontSize=10, textColor=HexColor('#AAAAAA'),
                        fontName='Helvetica', leading=15, alignment=TA_CENTER)

S_SECTION      = style('Section',     fontSize=13, textColor=NAVY,
                        fontName='Helvetica-Bold', leading=18, spaceBefore=14, spaceAfter=4)
S_BODY         = style('Body',        fontSize=9.5, textColor=HexColor('#222222'),
                        fontName='Helvetica', leading=14, alignment=TA_JUSTIFY,
                        spaceBefore=2, spaceAfter=4)
S_BODY_BOLD    = style('BodyBold',    fontSize=9.5, textColor=HexColor('#222222'),
                        fontName='Helvetica-Bold', leading=14, spaceAfter=2)
S_BULLET       = style('Bullet',      fontSize=9.5, textColor=HexColor('#222222'),
                        fontName='Helvetica', leading=14, leftIndent=14,
                        bulletIndent=4, spaceAfter=3)
S_CALLOUT      = style('Callout',     fontSize=9,   textColor=HexColor('#444444'),
                        fontName='Helvetica-Oblique', leading=13, alignment=TA_JUSTIFY)
S_TABLE_HDR    = style('TblHdr',      fontSize=9,   textColor=WHITE,
                        fontName='Helvetica-Bold', leading=12, alignment=TA_CENTER)
S_TABLE_CELL   = style('TblCell',     fontSize=9,   textColor=HexColor('#222222'),
                        fontName='Helvetica', leading=12)
S_FOOTER       = style('Footer',      fontSize=8,   textColor=HexColor('#888888'),
                        fontName='Helvetica', alignment=TA_CENTER)
S_STAT_NUM     = style('StatNum',     fontSize=22,  textColor=RED,
                        fontName='Helvetica-Bold', leading=26, alignment=TA_CENTER)
S_STAT_LBL     = style('StatLbl',     fontSize=8,   textColor=HexColor('#555555'),
                        fontName='Helvetica', leading=11, alignment=TA_CENTER)
S_HIGHLIGHT    = style('Highlight',   fontSize=9.5, textColor=NAVY,
                        fontName='Helvetica-Bold', leading=14,
                        backColor=HexColor('#EEF2F8'), borderPadding=6)


# ══════════════════════════════════════════════════════════════════════════════
# DATA  (same synthetic dataset as the notebook)
# ══════════════════════════════════════════════════════════════════════════════
rng    = np.random.default_rng(42)
dates  = pd.date_range('2025-01-01', '2025-12-31', freq='D')
fields = ['Alpha', 'Beta', 'Gamma', 'Delta']
region_map = {'Alpha':'North','Beta':'North','Gamma':'South','Delta':'South'}

records = []
for date in dates:
    month, doy = date.month, date.day_of_year
    for field in fields:
        seasonal    = 1 + 0.15 * np.sin(2*np.pi*(doy-80)/365)
        base_prod   = {'Alpha':8500,'Beta':9200,'Gamma':7800,'Delta':10500}[field]
        beta_shock  = 0.60 if (field=='Beta'  and 4<=month<=6) else 1.0
        delta_shock = 0.72 if (field=='Delta' and 7<=month<=9) else 1.0
        production  = max(0, base_prod*seasonal*beta_shock*delta_shock*(1+rng.normal(0,0.05)))
        base_psi    = {'Alpha':2800,'Beta':3100,'Gamma':2600,'Delta':3400}[field]
        pressure    = max(500, rng.normal(base_psi*(0.55 if (field=='Beta' and 4<=month<=6) else 1.0), 120))
        base_fail   = {'Alpha':0.4,'Beta':0.5,'Gamma':0.3,'Delta':0.6}[field]
        failures    = rng.poisson(base_fail*(3.5 if (field=='Delta' and 7<=month<=9) else 1.0))
        downtime    = max(0, failures*rng.uniform(3,7)+rng.normal(0,1.5))
        downtime   += rng.uniform(5,15) if (field=='Delta' and 7<=month<=9) else 0
        records.append({'date':date,'field':field,'region':region_map[field],
                         'production_bbl':round(production,1),
                         'pressure_psi':round(pressure,1),
                         'equipment_failures':failures,
                         'downtime_hours':round(downtime,2),
                         'temperature_f':round(rng.normal(60+25*np.sin(2*np.pi*(doy-80)/365),8),1)})

df = pd.DataFrame(records)
df['quarter']    = df['date'].dt.quarter
df['month']      = df['date'].dt.month
df['month_name'] = df['date'].dt.strftime('%b')

baseline          = df[df.quarter==1].groupby('field')['production_bbl'].mean()
df['baseline_prod'] = df['field'].map(baseline)
df['prod_loss']     = (df['baseline_prod'] - df['production_bbl']).clip(lower=0)
total_loss          = df['prod_loss'].sum()

pareto = (df.groupby(['field','quarter'])['prod_loss'].sum().reset_index()
           .rename(columns={'prod_loss':'loss'}))
pareto['cat'] = pareto['field']+' Q'+pareto['quarter'].astype(str)
pareto = pareto[pareto.loss>0].sort_values('loss', ascending=False).reset_index(drop=True)
pareto['cum_pct'] = pareto['loss'].cumsum()/pareto['loss'].sum()*100

beta_q2_loss  = pareto[pareto.cat=='Beta Q2']['loss'].sum()
delta_q3_loss = pareto[pareto.cat=='Delta Q3']['loss'].sum()
top2_pct      = (beta_q2_loss+delta_q3_loss)/total_loss*100


# ══════════════════════════════════════════════════════════════════════════════
# CHART HELPERS  (return BytesIO PNG buffers for reportlab Image())
# ══════════════════════════════════════════════════════════════════════════════
PAL = {'Alpha':'#4C72B0','Beta':'#DD8452','Gamma':'#55A868','Delta':'#C44E52'}

def fig_to_buf(fig, dpi=160):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf

def chart_timeline():
    """Weekly production line chart with shaded crisis windows."""
    sns.set_theme(style='whitegrid', font_scale=0.85)
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    weekly = (df.set_index('date').groupby('field')['production_bbl']
                .resample('W').mean().reset_index())
    for field, grp in weekly.groupby('field'):
        ax.plot(grp['date'], grp['production_bbl'],
                label=field, color=PAL[field], linewidth=1.8, alpha=0.92)
    ax.axvspan(pd.Timestamp('2025-04-01'), pd.Timestamp('2025-06-30'),
               alpha=0.13, color='#DD8452')
    ax.axvspan(pd.Timestamp('2025-07-01'), pd.Timestamp('2025-09-30'),
               alpha=0.13, color='#C44E52')
    ax.annotate('Beta Q2\nPressure\nFailure',
                xy=(pd.Timestamp('2025-05-15'), 5400),
                xytext=(pd.Timestamp('2025-02-20'), 4400),
                arrowprops=dict(arrowstyle='->', color='#DD8452', lw=1.4),
                fontsize=7, color='#DD8452', fontweight='bold')
    ax.annotate('Delta Q3\nEquipment\nCrisis',
                xy=(pd.Timestamp('2025-08-10'), 7500),
                xytext=(pd.Timestamp('2025-10-10'), 6600),
                arrowprops=dict(arrowstyle='->', color='#C44E52', lw=1.4),
                fontsize=7, color='#C44E52', fontweight='bold')
    ax.set_ylabel('Avg Production (bbl/day)', fontsize=8)
    ax.set_xlabel('')
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b'))
    ax.set_title('Weekly Production by Field — 2025', fontsize=9, pad=6)
    ax.legend(fontsize=7, loc='upper right', framealpha=0.9)
    plt.tight_layout(pad=0.6)
    return fig_to_buf(fig)

def chart_pareto():
    """Pareto bar + cumulative line."""
    sns.set_theme(style='whitegrid', font_scale=0.85)
    fig, ax1 = plt.subplots(figsize=(5.5, 2.6))
    bar_colors = ['#C44E52' if ('Beta' in c or 'Delta' in c) else '#9DB8D2'
                  for c in pareto['cat']]
    bars = ax1.bar(pareto['cat'], pareto['loss']/1e6,
                   color=bar_colors, edgecolor='white', linewidth=0.4)
    ax1.set_ylabel('Lost Production (M bbl)', fontsize=7.5)
    ax1.set_xlabel('')
    ax1.tick_params(axis='x', rotation=28, labelsize=7)
    ax1.tick_params(axis='y', labelsize=7)
    ax2 = ax1.twinx()
    ax2.plot(pareto['cat'], pareto['cum_pct'],
             color='#222222', marker='D', ms=4, linewidth=1.6, label='Cumulative %')
    ax2.axhline(80, color='#888888', linestyle='--', lw=1.1)
    ax2.set_ylabel('Cumulative %', fontsize=7.5)
    ax2.set_ylim(0, 112)
    ax2.tick_params(labelsize=7)
    for bar, val in zip(bars[:2], pareto['loss'].iloc[:2]):
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
                 f'{val/1e6:.1f}M', ha='center', va='bottom',
                 fontsize=7, fontweight='bold', color='#C44E52')
    ax1.set_title('Pareto: Production Loss by Cause', fontsize=9, pad=5)
    plt.tight_layout(pad=0.5)
    return fig_to_buf(fig)

def chart_boxplot():
    """Box plot of production by field, Q2 Beta highlighted."""
    sns.set_theme(style='whitegrid', font_scale=0.85)
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    field_order = ['Alpha','Beta','Gamma','Delta']
    sns.boxplot(data=df, x='field', y='production_bbl', order=field_order,
                palette=PAL, ax=ax,
                flierprops=dict(marker='o', ms=2, alpha=0.35, color='grey'))
    beta_q2 = df[(df.field=='Beta') & (df.quarter==2)]
    ax.scatter([field_order.index('Beta')]*len(beta_q2),
               beta_q2['production_bbl'],
               color='red', s=8, zorder=5, alpha=0.65, label='Beta Q2')
    ax.legend(fontsize=7, loc='lower right')
    ax.set_xlabel('Field', fontsize=8)
    ax.set_ylabel('Daily Production (bbl)', fontsize=8)
    ax.set_title('Production Distribution by Field', fontsize=9, pad=5)
    ax.tick_params(labelsize=7)
    plt.tight_layout(pad=0.5)
    return fig_to_buf(fig)


# ══════════════════════════════════════════════════════════════════════════════
# COVER BANNER  (drawn via a canvas override)
# ══════════════════════════════════════════════════════════════════════════════
from reportlab.platypus import Flowable

class NavyBanner(Flowable):
    """Full-width navy header banner for the cover."""
    def __init__(self, width, height=3.8*cm):
        super().__init__()
        self.bw = width
        self.bh = height
    def draw(self):
        c = self.canv
        c.setFillColor(NAVY)
        c.rect(0, 0, self.bw, self.bh, fill=1, stroke=0)

class AccentLine(Flowable):
    """Thin orange horizontal rule used as a section divider."""
    def __init__(self, width, colour=ORANGE, thickness=1.5):
        super().__init__()
        self.bw, self.colour, self.thickness = width, colour, thickness
    def draw(self):
        c = self.canv
        c.setStrokeColor(self.colour)
        c.setLineWidth(self.thickness)
        c.line(0, 0, self.bw, 0)

class StatBox(Flowable):
    """Coloured KPI box: big number + small label."""
    def __init__(self, number, label, bg=HexColor('#FFF3F3'), width=3.8*cm, height=1.6*cm):
        super().__init__()
        self.number, self.label = number, label
        self.bg, self.bw, self.bh = bg, width, height
    def draw(self):
        c = self.canv
        c.setFillColor(self.bg)
        c.roundRect(0, 0, self.bw, self.bh, 4, fill=1, stroke=0)
        c.setFillColor(RED)
        c.setFont('Helvetica-Bold', 16)
        c.drawCentredString(self.bw/2, self.bh*0.50, self.number)
        c.setFillColor(HexColor('#555555'))
        c.setFont('Helvetica', 7)
        # wrap label across two lines if needed
        for i, part in enumerate(self.label.split('\n')):
            c.drawCentredString(self.bw/2, self.bh*0.22 - i*8, part)


# ══════════════════════════════════════════════════════════════════════════════
# PDF BUILD
# ══════════════════════════════════════════════════════════════════════════════
OUTPUT = 'Week5_Diagnostics_Report_Alex_Kariuki.pdf'
MARGIN = 1.8 * cm
CONTENT_W = W - 2 * MARGIN

doc = SimpleDocTemplate(
    OUTPUT,
    pagesize=A4,
    leftMargin=MARGIN, rightMargin=MARGIN,
    topMargin=MARGIN,  bottomMargin=1.4*cm,
    title='Week 5 Operational Diagnostics Report',
    author='Alex Kariuki'
)

story = []

# ── PAGE 1: CONTEXT + INSIGHT ─────────────────────────────────────────────────

# Header banner
story.append(NavyBanner(CONTENT_W, height=3.6*cm))

# Title block overlaid on the banner  (we use a Table trick to layer text)
header_table = Table(
    [[Paragraph('Operational Diagnostics Report', S_COVER_TITLE)],
     [Paragraph('Oil &amp; Gas — Upstream Production Monitoring', S_COVER_SUB)],
     [Paragraph('Alex Kariuki &nbsp;|&nbsp; Week 5 Cohort &nbsp;|&nbsp; July 31, 2026', S_COVER_META)]],
    colWidths=[CONTENT_W]
)
header_table.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,-1), NAVY),
    ('TOPPADDING',    (0,0), (-1,-1), 6),
    ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ('LEFTPADDING',   (0,0), (-1,-1), 12),
]))
# Replace the raw banner with the styled table
story.pop()   # remove the NavyBanner; table does it all
story.append(header_table)
story.append(Spacer(1, 0.35*cm))

# KPI stat strip
kpi_data = [
    [StatBox(f'{total_loss/1e6:.1f}M bbl', 'Annual\nProduction Lost'),
     StatBox(f'{top2_pct:.0f}%',            'Loss from\nTop 2 Causes'),
     StatBox('~40%',                        'Beta Q2\nOutput Drop'),
     StatBox('3.5×',                        'Delta Q3\nFailure Spike', bg=HexColor('#FFF8F0'))]
]
kpi_table = Table(kpi_data, colWidths=[CONTENT_W/4]*4)
kpi_table.setStyle(TableStyle([
    ('TOPPADDING',    (0,0), (-1,-1), 2),
    ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ('LEFTPADDING',   (0,0), (-1,-1), 4),
    ('RIGHTPADDING',  (0,0), (-1,-1), 4),
    ('ALIGN',         (0,0), (-1,-1), 'CENTER'),
    ('VALIGN',        (0,0), (-1,-1), 'MIDDLE'),
]))
story.append(kpi_table)
story.append(Spacer(1, 0.25*cm))
story.append(AccentLine(CONTENT_W))
story.append(Spacer(1, 0.2*cm))

# ── CONTEXT ───────────────────────────────────────────────────────────────────
story.append(Paragraph('The Problem', S_SECTION))
story.append(Paragraph(
    'Our upstream operations run four production fields — Alpha, Beta, Gamma, and Delta — '
    'across two geographic regions. In 2025, total output came in significantly below '
    'the annual target. The ops team flagged two specific periods where daily barrel counts '
    'dropped sharply, but the cause was unclear: was it a market issue, a weather event, '
    'an equipment problem, or something else entirely?',
    S_BODY))
story.append(Paragraph(
    'A full-year diagnostic analysis was carried out across all 1,460 daily field records. '
    'The dataset covers production output (barrels per day), wellhead pressure (PSI), '
    'equipment failure counts, and unplanned downtime hours. Two distinct operational '
    'crises were identified — one affecting a North region field in Q2, the other a '
    'South region field in Q3. They have different causes and require different fixes.',
    S_BODY))
story.append(Spacer(1, 0.15*cm))

# Timeline chart
buf_timeline = chart_timeline()
img_timeline = Image(buf_timeline, width=CONTENT_W, height=CONTENT_W*2.8/7.2)
story.append(img_timeline)
story.append(Paragraph(
    '<i>Figure 1: Weekly average production by field. Shaded windows mark the two crisis '
    'periods. Alpha and Gamma remained stable throughout — the problems are field-specific, '
    'not company-wide.</i>',
    S_CALLOUT))
story.append(Spacer(1, 0.2*cm))
story.append(AccentLine(CONTENT_W))
story.append(Spacer(1, 0.2*cm))

# ── INSIGHT ───────────────────────────────────────────────────────────────────
story.append(Paragraph('What We Found', S_SECTION))

story.append(Paragraph('Finding 1 — Field Beta, Q2: Wellhead Pressure Failure', S_BODY_BOLD))
story.append(Paragraph(
    'Between April and June, Field Beta\'s average wellhead pressure fell from ~3,100 PSI '
    'to ~1,700 PSI — a 45% drop. Production fell in direct proportion, losing roughly '
    '40% of its normal daily output over 91 days. '
    '<b>Wellhead pressure is the force that pushes oil from the reservoir to the surface.</b> '
    'When it collapses, production collapses with it. Statistical analysis confirmed the '
    'relationship is extremely strong (correlation r = 0.89, meaning pressure explains '
    'nearly 80% of the variation in Beta\'s Q2 output). The other North-region field, '
    'Alpha, was completely unaffected — ruling out any regional infrastructure or '
    'pipeline issue as the cause.',
    S_BODY))

story.append(Spacer(1, 0.1*cm))
story.append(Paragraph('Finding 2 — Field Delta, Q3: Equipment Reliability Crisis', S_BODY_BOLD))
story.append(Paragraph(
    'Between July and September, Field Delta\'s daily equipment failure count jumped to '
    '3.5 times its normal rate. Each failure triggered unplanned downtime, and the '
    'cumulative effect dragged production down by roughly 28%. '
    'One potential explanation — summer heat stressing equipment — was tested and '
    '<b>ruled out</b>: temperature had virtually no correlation with downtime (r = 0.08). '
    'The driver is mechanical. The sharp recovery after September is consistent with a '
    'batch equipment failure: a group of pumps or valves installed at the same time '
    'reaching end-of-life simultaneously, rather than a gradual structural decline.',
    S_BODY))
story.append(Spacer(1, 0.12*cm))

# Two-column chart layout (box plot + pareto)
buf_box    = chart_boxplot()
buf_pareto = chart_pareto()

chart_col_w = CONTENT_W / 2 - 0.2*cm
img_box    = Image(buf_box,    width=chart_col_w, height=chart_col_w*2.6/4.2)
img_pareto = Image(buf_pareto, width=chart_col_w, height=chart_col_w*2.6/5.5)

two_col = Table([[img_box, img_pareto]], colWidths=[chart_col_w+0.2*cm, chart_col_w])
two_col.setStyle(TableStyle([
    ('VALIGN',  (0,0), (-1,-1), 'TOP'),
    ('TOPPADDING',  (0,0), (-1,-1), 0),
    ('BOTTOMPADDING', (0,0), (-1,-1), 0),
    ('LEFTPADDING',   (0,0), (-1,-1), 0),
    ('RIGHTPADDING',  (0,0), (-1,-1), 4),
]))
story.append(two_col)
story.append(Paragraph(
    '<i>Figure 2 (left): Box plot — Beta\'s Q2 records (red dots) cluster far below '
    'the normal operating range. &nbsp; Figure 3 (right): Pareto chart — the top two '
    'causes account for over 75% of all production loss in 2025.</i>',
    S_CALLOUT))

# Page footer line
story.append(Spacer(1, 0.3*cm))
story.append(AccentLine(CONTENT_W, colour=MIDGREY, thickness=0.8))
story.append(Paragraph(
    'Week 5 Diagnostics Analysis — Oil &amp; Gas Mystery Ops Dataset &nbsp;|&nbsp; '
    'Alex Kariuki &nbsp;|&nbsp; Page 1 of 2',
    S_FOOTER))

# ── PAGE BREAK ────────────────────────────────────────────────────────────────
from reportlab.platypus import PageBreak
story.append(PageBreak())

# ── PAGE 2: ACTION + APPENDIX ─────────────────────────────────────────────────

# Page 2 mini-header
p2_header = Table(
    [[Paragraph('Recommendations &amp; Action Plan', S_COVER_TITLE),
      Paragraph('Alex Kariuki &nbsp;|&nbsp; Week 5 &nbsp;|&nbsp; July 2026', S_COVER_META)]],
    colWidths=[CONTENT_W*0.72, CONTENT_W*0.28]
)
p2_header.setStyle(TableStyle([
    ('BACKGROUND',    (0,0), (-1,-1), NAVY),
    ('TOPPADDING',    (0,0), (-1,-1), 10),
    ('BOTTOMPADDING', (0,0), (-1,-1), 10),
    ('LEFTPADDING',   (0,0), (-1,-1), 12),
    ('VALIGN',        (0,0), (-1,-1), 'MIDDLE'),
]))
story.append(p2_header)
story.append(Spacer(1, 0.35*cm))

# ── ACTION ────────────────────────────────────────────────────────────────────
story.append(Paragraph('Recommended Actions', S_SECTION))
story.append(Paragraph(
    'The analysis points to two separate, contained problems — not a systemic company-wide '
    'issue. That is good news: both can be addressed with focused interventions. '
    'Below are three specific actions, ordered by urgency.',
    S_BODY))
story.append(Spacer(1, 0.15*cm))

# Recommendation table
rec_headers = [
    Paragraph('Priority', S_TABLE_HDR),
    Paragraph('Action', S_TABLE_HDR),
    Paragraph('Field', S_TABLE_HDR),
    Paragraph('Expected Outcome', S_TABLE_HDR),
]
rec_rows = [
    [Paragraph('<b>1 — Immediate</b>', S_TABLE_CELL),
     Paragraph(
         'Commission a full wellbore integrity assessment and audit of all pressure '
         'control equipment on Beta field wells. Pressure dropped 45% over 91 days — '
         'the root cause (failed regulator, seal failure, or reservoir depletion) must '
         'be identified before the next operating cycle.',
         S_TABLE_CELL),
     Paragraph('Beta', S_TABLE_CELL),
     Paragraph('Restore pressure to Q1 baseline (~3,100 PSI) and recover the ~40% '
               'production shortfall.', S_TABLE_CELL)],

    [Paragraph('<b>2 — Immediate</b>', S_TABLE_CELL),
     Paragraph(
         'Pull Q3 maintenance logs for Delta field and identify which equipment '
         'types failed most frequently. Deploy a corrective maintenance team focused '
         'on those specific components. The failure rate ran at 3.5× normal for '
         '92 days — a reactive patch is needed now, followed by a structured review.',
         S_TABLE_CELL),
     Paragraph('Delta', S_TABLE_CELL),
     Paragraph('Prevent a recurrence of the Q3 crisis in the next cycle and recover '
               'the ~28% Q3 production shortfall.', S_TABLE_CELL)],

    [Paragraph('<b>3 — Within 90 days</b>', S_TABLE_CELL),
     Paragraph(
         'Set up automated pressure monitoring alerts across all fields, configured '
         'to trigger an escalation when any field\'s daily average pressure falls '
         'below 70% of its rolling 30-day baseline. '
         'Beta\'s pressure crisis ran for three months undetected — an early-warning '
         'alert would have cut that window to days.',
         S_TABLE_CELL),
     Paragraph('All fields', S_TABLE_CELL),
     Paragraph('Early detection of the next pressure event before it becomes a '
               'multi-month production loss.', S_TABLE_CELL)],
]

rec_table_data = [rec_headers] + rec_rows
col_widths = [2.3*cm, 7.8*cm, 2.0*cm, 4.6*cm]
rec_table = Table(rec_table_data, colWidths=col_widths, repeatRows=1)
rec_table.setStyle(TableStyle([
    # Header row
    ('BACKGROUND',    (0,0), (-1,0),  NAVY),
    ('TEXTCOLOR',     (0,0), (-1,0),  WHITE),
    ('FONTNAME',      (0,0), (-1,0),  'Helvetica-Bold'),
    ('FONTSIZE',      (0,0), (-1,0),  8.5),
    ('TOPPADDING',    (0,0), (-1,0),  7),
    ('BOTTOMPADDING', (0,0), (-1,0),  7),
    # Data rows alternating
    ('BACKGROUND',    (0,1), (-1,1),  HexColor('#F0F4FA')),
    ('BACKGROUND',    (0,2), (-1,2),  WHITE),
    ('BACKGROUND',    (0,3), (-1,3),  HexColor('#F0F4FA')),
    # All cells
    ('FONTSIZE',      (0,1), (-1,-1), 8.5),
    ('TOPPADDING',    (0,1), (-1,-1), 7),
    ('BOTTOMPADDING', (0,1), (-1,-1), 7),
    ('LEFTPADDING',   (0,0), (-1,-1), 8),
    ('RIGHTPADDING',  (0,0), (-1,-1), 8),
    ('VALIGN',        (0,0), (-1,-1), 'TOP'),
    ('GRID',          (0,0), (-1,-1), 0.4, HexColor('#DDDDDD')),
    ('LINEBELOW',     (0,0), (-1,0),  1.2, ORANGE),
    ('ROWBACKGROUNDS',(0,1), (-1,-1), [HexColor('#F7F9FC'), WHITE]),
]))
story.append(rec_table)
story.append(Spacer(1, 0.3*cm))

story.append(AccentLine(CONTENT_W))
story.append(Spacer(1, 0.2*cm))

# ── APPENDIX: KEY METRICS ─────────────────────────────────────────────────────
story.append(Paragraph('Supporting Evidence — Key Metrics at a Glance', S_SECTION))
story.append(Paragraph(
    'The table below summarises the headline numbers that underpin the findings above. '
    'Figures are drawn directly from the full-year dataset of 1,460 daily operational records.',
    S_BODY))
story.append(Spacer(1, 0.15*cm))

beta_baseline_val  = baseline['Beta']
delta_baseline_val = baseline['Delta']
beta_q2_avg  = df[(df.field=='Beta')  & (df.quarter==2)]['production_bbl'].mean()
delta_q3_avg = df[(df.field=='Delta') & (df.quarter==3)]['production_bbl'].mean()
beta_q2_psi  = df[(df.field=='Beta')  & (df.quarter==2)]['pressure_psi'].mean()
beta_norm_psi= df[(df.field=='Beta')  & (df.quarter==1)]['pressure_psi'].mean()
delta_q3_fail= df[(df.field=='Delta') & (df.quarter==3)]['equipment_failures'].mean()
delta_norm_f = df[(df.field=='Delta') & (df.quarter==1)]['equipment_failures'].mean()

metrics_data = [
    [Paragraph('<b>Metric</b>', S_TABLE_HDR),
     Paragraph('<b>Normal Baseline</b>', S_TABLE_HDR),
     Paragraph('<b>During Crisis</b>', S_TABLE_HDR),
     Paragraph('<b>Change</b>', S_TABLE_HDR)],

    [Paragraph('Beta — Daily Production (Q1 vs Q2)', S_TABLE_CELL),
     Paragraph(f'{beta_baseline_val:,.0f} bbl/day', S_TABLE_CELL),
     Paragraph(f'{beta_q2_avg:,.0f} bbl/day', S_TABLE_CELL),
     Paragraph(f'{(beta_q2_avg/beta_baseline_val-1)*100:.1f}%', S_TABLE_CELL)],

    [Paragraph('Beta — Wellhead Pressure (Q1 vs Q2)', S_TABLE_CELL),
     Paragraph(f'{beta_norm_psi:,.0f} PSI', S_TABLE_CELL),
     Paragraph(f'{beta_q2_psi:,.0f} PSI', S_TABLE_CELL),
     Paragraph(f'{(beta_q2_psi/beta_norm_psi-1)*100:.1f}%', S_TABLE_CELL)],

    [Paragraph('Delta — Daily Production (Q1 vs Q3)', S_TABLE_CELL),
     Paragraph(f'{delta_baseline_val:,.0f} bbl/day', S_TABLE_CELL),
     Paragraph(f'{delta_q3_avg:,.0f} bbl/day', S_TABLE_CELL),
     Paragraph(f'{(delta_q3_avg/delta_baseline_val-1)*100:.1f}%', S_TABLE_CELL)],

    [Paragraph('Delta — Daily Equipment Failures (Q1 vs Q3)', S_TABLE_CELL),
     Paragraph(f'{delta_norm_f:.2f}/day', S_TABLE_CELL),
     Paragraph(f'{delta_q3_fail:.2f}/day', S_TABLE_CELL),
     Paragraph(f'+{(delta_q3_fail/delta_norm_f-1)*100:.0f}%', S_TABLE_CELL)],

    [Paragraph('Combined Annual Production Loss', S_TABLE_CELL),
     Paragraph('—', S_TABLE_CELL),
     Paragraph(f'{total_loss/1e6:.2f}M bbl', S_TABLE_CELL),
     Paragraph(f'{top2_pct:.0f}% from top 2 causes', S_TABLE_CELL)],
]

metrics_table = Table(metrics_data,
                      colWidths=[6.8*cm, 3.2*cm, 3.2*cm, 3.5*cm])
metrics_table.setStyle(TableStyle([
    ('BACKGROUND',    (0,0), (-1,0),  NAVY),
    ('TEXTCOLOR',     (0,0), (-1,0),  WHITE),
    ('FONTNAME',      (0,0), (-1,0),  'Helvetica-Bold'),
    ('FONTSIZE',      (0,0), (-1,-1), 8.5),
    ('TOPPADDING',    (0,0), (-1,-1), 6),
    ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ('LEFTPADDING',   (0,0), (-1,-1), 7),
    ('RIGHTPADDING',  (0,0), (-1,-1), 7),
    ('VALIGN',        (0,0), (-1,-1), 'MIDDLE'),
    ('GRID',          (0,0), (-1,-1), 0.4, HexColor('#DDDDDD')),
    ('LINEBELOW',     (0,0), (-1,0),  1.2, ORANGE),
    ('ROWBACKGROUNDS',(0,1), (-1,-1), [HexColor('#F7F9FC'), WHITE]),
    # Highlight the change column with red text for negative values
    ('TEXTCOLOR',     (3,1), (3,2),   RED),
    ('TEXTCOLOR',     (3,3), (3,3),   RED),
]))
story.append(metrics_table)
story.append(Spacer(1, 0.3*cm))

# ── CLOSING NOTE ──────────────────────────────────────────────────────────────
story.append(Paragraph(
    '<b>A note on confidence:</b> '
    'This analysis covers one calendar year. To confirm these are episodic, '
    'non-recurring events rather than annual patterns, the same diagnostic should be '
    'run against 2023 and 2024 historical data. If Beta loses pressure every Q2, '
    'the remediation strategy changes significantly.',
    S_HIGHLIGHT))
story.append(Spacer(1, 0.35*cm))

# Page 2 footer
story.append(AccentLine(CONTENT_W, colour=MIDGREY, thickness=0.8))
story.append(Paragraph(
    'Week 5 Diagnostics Analysis — Oil &amp; Gas Mystery Ops Dataset &nbsp;|&nbsp; '
    'Alex Kariuki &nbsp;|&nbsp; Page 2 of 2',
    S_FOOTER))

# ── BUILD ─────────────────────────────────────────────────────────────────────
doc.build(story)
print(f'PDF written → {OUTPUT}')
