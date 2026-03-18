"""HTML report generation — light-theme design with full interactivity.

Generates comprehensive single-page HTML reports with:
- Light theme with gradient header
- Sticky top navigation bar (horizontal pills, scroll-spy)
- Sub-tab system (Basic / Advanced with ADV badges)
- Drag-to-scroll tables with momentum
- Image zoom/pan modal (wheel, pinch, double-click reset)
- Metric tooltips (100+ definitions)
- Language selector (6 languages)
- Quality gauge bars
- Inline base64 charts (32 chart types)
"""

from __future__ import annotations

import base64
import html as html_mod
import io
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from f2a.report.i18n import SUPPORTED_LANGUAGES, TRANSLATIONS, t

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    import pandas as pd
    HAS_PD = True
except ImportError:
    HAS_PD = False

logger = logging.getLogger("f2a.report")


# =====================================================================
#  Metric tooltip descriptions
# =====================================================================

_METRIC_TIPS: dict[str, str] = {
    "type": "Inferred data type of the column.",
    "count": "Number of non-null values.",
    "missing": "Number of missing (null / NaN) values.",
    "missing_%": "Percentage of missing values.",
    "unique": "Number of distinct values.",
    "mean": "Arithmetic mean.",
    "median": "Middle value (50th percentile).",
    "std": "Standard deviation — spread around the mean.",
    "se": "Standard error of the mean = std / sqrt(n).",
    "cv": "Coefficient of variation = std / |mean|.",
    "mad": "Median Absolute Deviation.",
    "min": "Minimum value.", "max": "Maximum value.",
    "range": "Range = max - min.",
    "p5": "5th percentile.", "q1": "25th percentile (Q1).",
    "q3": "75th percentile (Q3).", "p95": "95th percentile.",
    "iqr": "Interquartile Range = Q3 - Q1.",
    "skewness": "Distribution asymmetry. 0 = symmetric, >0 = right-skewed.",
    "kurtosis": "Tail heaviness. 0 = normal, >0 = heavy tails.",
    "top": "Most frequent value.", "freq": "Frequency of most common value.",
    "n": "Number of non-null observations.",
    "normality_p": "p-value of normality test. p<0.05 → likely non-normal.",
    "shapiro_p": "Shapiro-Wilk p-value.",
    "missing_count": "Number of missing values.",
    "missing_ratio": "Fraction of missing values.",
    "dtype": "Data type of the column.",
    "lower_bound": "IQR lower fence.",
    "upper_bound": "IQR upper fence.",
    "outlier_count": "Number of outlier values.",
    "outlier_%": "Percentage of outliers.",
    "top_value": "Most frequent category.",
    "top_frequency": "Count of most frequent category.",
    "entropy": "Shannon entropy (bits). Higher = more uniform.",
    "variance": "Variance of the column.",
    "mean_abs_corr": "Mean absolute Pearson correlation.",
    "VIF": "Variance Inflation Factor. >5 = moderate, >10 = severe multicollinearity.",
    "variance_ratio": "Proportion of variance explained by this PC.",
    "cumulative_ratio": "Cumulative variance explained up to this PC.",
    "total_rows": "Total number of rows.",
    "duplicate_rows": "Number of duplicate rows.",
    "duplicate_ratio": "Fraction of duplicate rows.",
    "completeness": "Fraction of non-missing values.",
    "uniqueness": "Ratio of unique to total values.",
    "consistency": "Type consistency score.",
    "validity": "Fraction of valid values.",
    "overall": "Weighted quality score.",
    "quality_score": "Per-column quality score.",
    "column": "Column name.", "value": "Value.",
    "percentage": "Percentage share.",
    "optimal_k": "Best number of clusters (silhouette).",
    "best_silhouette": "Highest mean silhouette score (-1 to 1).",
    "noise_ratio": "Points labelled as noise by DBSCAN.",
    "anomaly_ratio": "Fraction of detected anomalies.",
    "consensus_ratio": "Fraction agreed anomalous by ≥2/3 methods.",
}


# =====================================================================
#  Helpers
# =====================================================================

def _esc(text: str) -> str:
    return html_mod.escape(str(text))


def _fmt(val: Any) -> str:
    if val is None:
        return "—"
    if isinstance(val, float):
        if abs(val) >= 1e6:
            return f"{val:.2e}"
        return f"{val:.4f}"
    if isinstance(val, bool):
        return "Yes" if val else "No"
    return str(val)


_RATIO_KEYS = frozenset({
    "anomaly_ratio", "noise_ratio", "consensus_ratio", "missing_ratio",
    "duplicate_ratio", "duplicate_row_ratio", "numeric_ratio",
    "categorical_ratio", "total_variance_explained",
    "completeness", "uniqueness", "consistency", "validity",
    "overall", "overall_score",
})


def _dict_to_cards(d: dict[str, Any]) -> str:
    """Convert a flat dict into stat-card HTML."""
    cards: list[str] = []
    for key, val in d.items():
        if isinstance(val, (dict, list)):
            continue
        if isinstance(val, float):
            display = (f"{val * 100:.1f}%" if key in _RATIO_KEYS and 0 <= val <= 1
                       else f"{val:,.4f}" if abs(val) < 1e6 else f"{val:.2e}")
        elif isinstance(val, int):
            display = f"{val:,}"
        elif isinstance(val, bool):
            display = "Yes" if val else "No"
        else:
            display = str(val)
        label = key.replace("_", " ").title()
        tip = _METRIC_TIPS.get(key, "")
        tip_attr = f' data-tip="{_esc(tip)}"' if tip else ""
        cards.append(
            f'<div class="card"{tip_attr}>'
            f'<div class="value">{_esc(display)}</div>'
            f'<div class="label">{_esc(label)}</div></div>'
        )
    return "\n".join(cards)


def _json_table(data: Any, max_rows: int = 100) -> str:
    """Render a list-of-dicts or dict as an HTML table."""
    if isinstance(data, dict):
        rows = ""
        for k, v in data.items():
            if isinstance(v, (dict, list)):
                continue
            tip = _METRIC_TIPS.get(k, "")
            tip_attr = f' data-tip="{_esc(tip)}"' if tip else ""
            rows += (f'<tr><td{tip_attr}><strong>{_esc(k)}</strong></td>'
                     f'<td>{_esc(_fmt(v))}</td></tr>')
        if not rows:
            return ""
        return (f'<div class="table-wrapper"><table class="table"><tbody>'
                f'{rows}</tbody></table><div class="scroll-hint"></div></div>')

    if isinstance(data, list) and data and isinstance(data[0], dict):
        keys = list(data[0].keys())
        header = "".join(f'<th>{_esc(k)}</th>' for k in keys)
        body = ""
        for item in data[:max_rows]:
            cells = ""
            for k in keys:
                v = item.get(k)
                tip = _METRIC_TIPS.get(k, "")
                tip_attr = f' data-tip="{_esc(tip)}"' if tip else ""
                cells += f'<td{tip_attr}>{_esc(_fmt(v))}</td>'
            body += f'<tr>{cells}</tr>'
        return (f'<div class="table-wrapper"><table class="table">'
                f'<thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>'
                f'<div class="scroll-hint"></div></div>')
    return ""


def _quality_bars(data: dict) -> str:
    """Build quality gauge bars."""
    dims_data = data.get("dimensions", [])
    overall = data.get("overall_score", 0)
    parts: list[str] = []
    pct = overall * 100
    cls = "good" if pct >= 90 else ("fair" if pct >= 70 else "poor")
    parts.append(
        f'<div class="qbar"><div class="qbar-label">Overall</div>'
        f'<div class="qbar-track"><div class="qbar-fill {cls}" '
        f'style="width:{pct:.0f}%">{pct:.1f}%</div></div></div>'
    )
    for dim in dims_data:
        name = dim.get("name", "")
        score = dim.get("score", 0)
        pct = score * 100
        cls = "good" if pct >= 90 else ("fair" if pct >= 70 else "poor")
        parts.append(
            f'<div class="qbar"><div class="qbar-label">{_esc(name)}</div>'
            f'<div class="qbar-track"><div class="qbar-fill {cls}" '
            f'style="width:{pct:.0f}%">{pct:.1f}%</div></div></div>'
        )
    return '<div class="quality-bars">' + "".join(parts) + "</div>" if parts else ""


# =====================================================================
#  CSS — light theme with gradient header
# =====================================================================

_CSS = r"""
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    line-height: 1.6; color: #333; background: #f5f7fa; margin: 0;
}
.header {
    background: linear-gradient(135deg, #2c3e50, #3498db);
    color: #fff; padding: 30px 40px; position: relative;
}
.header h1 { font-size: 1.8em; margin-bottom: 4px; }
.header p  { font-size: 1.05em; opacity: 0.9; }
.analysis-meta { font-size: 0.88em; opacity: 0.8; margin-top: 4px; }
.topnav {
    background: #fff; border-bottom: 1px solid #dde; padding: 8px 20px;
    position: sticky; top: 0; z-index: 100;
    display: flex; flex-wrap: wrap; gap: 4px; align-items: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.topnav a {
    padding: 5px 14px; border-radius: 20px; text-decoration: none;
    color: #666; font-size: 0.82em; transition: all 0.2s; white-space: nowrap;
}
.topnav a:hover, .topnav a.active { background: #3498db; color: #fff; }
.main { max-width: 1400px; margin: 0 auto; padding: 20px; }
section {
    background: #fff; border-radius: 12px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    margin: 20px 0; padding: 25px;
}
.section-title {
    font-size: 1.25em; color: #2c3e50;
    border-bottom: 2px solid #3498db;
    padding-bottom: 8px; margin-bottom: 18px;
}
.section-subtitle { font-size: 1em; color: #555; margin: 18px 0 10px 0; }
.cards {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 12px; margin: 15px 0;
}
.card {
    background: #f8f9fa; border-radius: 8px; padding: 14px; text-align: center;
}
.card .value { font-size: 1.7em; font-weight: bold; color: #3498db; }
.card .label { font-size: 0.82em; color: #888; margin-top: 2px; }
/* Drag-scroll tables */
.table-wrapper {
    position: relative; overflow-x: auto; overflow-y: visible;
    margin: 12px 0; border: 1px solid #e0e0e0; border-radius: 8px;
    cursor: grab; -webkit-user-select: none; user-select: none;
}
.table-wrapper.dragging { cursor: grabbing; }
.table-wrapper .scroll-hint {
    position: absolute; top: 0; right: 0; bottom: 0; width: 40px;
    pointer-events: none;
    background: linear-gradient(to right, transparent, rgba(0,0,0,0.06));
    border-radius: 0 8px 8px 0; transition: opacity 0.3s;
}
.table-wrapper .scroll-hint.hidden { opacity: 0; }
.table {
    width: max-content; min-width: 100%; border-collapse: collapse; font-size: 0.85em;
}
.table th, .table td {
    padding: 7px 11px; text-align: left; border-bottom: 1px solid #eee; white-space: nowrap;
}
.table th {
    background: #f8f9fa; font-weight: 600; position: sticky; top: 0; z-index: 1;
}
.table th:first-child { position: sticky; left: 0; z-index: 2; background: #eef2f5; }
.table td:first-child {
    position: sticky; left: 0; background: #fff; z-index: 1;
    font-weight: 500; border-right: 2px solid #e0e0e0;
}
.table tr:hover td { background: #f1f3f5; }
.table tr:hover td:first-child { background: #e8ecf0; }
/* Charts */
.charts-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
    gap: 15px; margin: 15px 0;
}
.chart-card {
    background: #fafafa; border-radius: 8px; padding: 12px; text-align: center;
}
.chart-card img { max-width: 100%; border-radius: 6px; cursor: zoom-in; transition: opacity 0.15s; }
.chart-card img:hover { opacity: 0.85; }
.chart-card h4 { font-size: 0.9em; color: #555; margin-bottom: 8px; }
.chart-full { text-align: center; margin: 15px 0; }
.chart-full img {
    max-width: 100%; border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08); cursor: zoom-in;
}
/* Image-viewer modal */
.f2a-img-overlay {
    position: fixed; inset: 0; z-index: 10001;
    background: rgba(0,0,0,0.82); backdrop-filter: blur(4px);
    display: flex; align-items: center; justify-content: center;
    opacity: 0; pointer-events: none; transition: opacity 0.2s; cursor: grab;
}
.f2a-img-overlay.visible { opacity: 1; pointer-events: auto; }
.f2a-img-overlay.dragging { cursor: grabbing; }
.f2a-img-overlay .img-viewport {
    position: relative; width: 100%; height: 100%; overflow: hidden;
}
.f2a-img-overlay .img-viewport img {
    position: absolute; top: 0; left: 0; transform-origin: 0 0;
    max-width: none; max-height: none; user-select: none; -webkit-user-drag: none;
}
.f2a-img-overlay .img-close {
    position: fixed; top: 18px; right: 24px; z-index: 10002;
    background: rgba(255,255,255,0.15); border: none; color: #fff; font-size: 2em;
    cursor: pointer; width: 48px; height: 48px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
}
.f2a-img-overlay .img-close:hover { background: rgba(255,255,255,0.3); }
.f2a-img-overlay .img-title {
    position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%); z-index: 10002;
    color: #fff; font-size: 0.9em; background: rgba(0,0,0,0.5); padding: 6px 18px;
    border-radius: 20px; pointer-events: none;
}
.f2a-img-overlay .img-zoom-info {
    position: fixed; top: 24px; left: 50%; transform: translateX(-50%); z-index: 10002;
    color: #fff; font-size: 0.82em; background: rgba(0,0,0,0.45); padding: 4px 14px;
    border-radius: 14px; pointer-events: none; opacity: 0; transition: opacity 0.25s;
}
.f2a-img-overlay .img-zoom-info.show { opacity: 1; }
/* Quality gauges */
.quality-bars { display: flex; flex-wrap: wrap; gap: 20px; margin: 15px 0; }
.qbar { flex: 1; min-width: 120px; }
.qbar-label { font-size: 0.85em; color: #555; margin-bottom: 4px; }
.qbar-track { background: #eee; border-radius: 6px; height: 22px; position: relative; overflow: hidden; }
.qbar-fill {
    height: 100%; border-radius: 6px; transition: width 0.4s;
    display: flex; align-items: center; justify-content: flex-end;
    padding-right: 6px; font-size: 0.75em; color: #fff; font-weight: 600;
}
.qbar-fill.good { background: #27ae60; }
.qbar-fill.fair { background: #f39c12; }
.qbar-fill.poor { background: #e74c3c; }
/* Sub-tabs */
.sub-tab-bar {
    display: flex; flex-wrap: wrap; gap: 3px;
    border-bottom: 2px solid #d5dce4; margin: 18px 0 0 0;
}
.sub-tab-btn {
    padding: 7px 16px; border: 1px solid transparent; border-bottom: none;
    background: transparent; cursor: pointer; border-radius: 6px 6px 0 0;
    font-size: 0.84em; color: #888; transition: all 0.15s; white-space: nowrap;
}
.sub-tab-btn:hover { background: #edf2f7; color: #555; }
.sub-tab-btn.active {
    background: #fff; border-color: #d5dce4; border-bottom: 2px solid #fff;
    margin-bottom: -2px; font-weight: 600; color: #2980b9;
}
.sub-tab-btn.adv { color: #8e44ad; }
.sub-tab-btn.adv.active { color: #8e44ad; }
.sub-tab-content { padding: 18px 0; display: none; }
.sub-tab-content.active { display: block; }
.adv-badge {
    display: inline-block; background: #8e44ad; color: #fff; font-size: 0.7em;
    padding: 1px 7px; border-radius: 10px; margin-left: 8px; vertical-align: middle;
}
/* Tooltip */
.f2a-tooltip {
    position: fixed; z-index: 9999;
    max-width: 340px; padding: 10px 14px;
    background: #2c3e50; color: #fff; font-size: 0.82em; line-height: 1.5;
    border-radius: 8px; pointer-events: none;
    box-shadow: 0 4px 16px rgba(0,0,0,0.25);
    opacity: 0; transition: opacity 0.15s;
}
.f2a-tooltip.visible { opacity: 1; }
.f2a-tooltip .tip-header {
    font-weight: 700; color: #5dade2; margin-bottom: 4px;
    border-bottom: 1px solid rgba(255,255,255,0.15); padding-bottom: 3px;
}
.f2a-tooltip .tip-value { color: #f9e79f; font-weight: 600; }
[data-tip] { cursor: help; }
th[data-tip] { text-decoration: underline dotted rgba(0,0,0,0.25); text-underline-offset: 3px; }
/* Language selector */
.lang-selector {
    position: absolute; top: 24px; right: 30px;
    display: flex; align-items: center; gap: 8px;
}
.lang-selector label { font-size: 0.85em; opacity: 0.85; color: #fff; }
.lang-selector select {
    background: rgba(255,255,255,0.2); color: #fff;
    border: 1px solid rgba(255,255,255,0.4);
    border-radius: 6px; padding: 4px 10px; font-size: 0.85em; cursor: pointer;
}
.lang-selector select option { color: #333; background: #fff; }
/* Warnings */
.warnings {
    background: #fff3cd; border: 1px solid #ffc107; border-radius: 8px;
    padding: 14px; margin: 15px 0;
}
.warnings li { margin: 4px 0 4px 20px; font-size: 0.92em; }
/* Preprocessing log */
.log-list { list-style: none; padding: 0; }
.log-list li { padding: 4px 0; font-size: 0.9em; color: #555; }
.log-list li::before { content: "\2192  "; color: #3498db; font-weight: bold; }
/* Insight items */
.insight-item {
    border-left: 4px solid #3498db; padding: 12px 16px; margin-bottom: 10px;
    background: #fff; border-radius: 0 6px 6px 0;
}
/* Fallback JSON */
details { margin: 12px 0; }
details summary { cursor: pointer; color: #3498db; font-size: 0.9em; }
pre.json-pre {
    background: #f8f9fa; border: 1px solid #e0e0e0; border-radius: 8px;
    padding: 14px; overflow-x: auto; font-size: 0.8em;
    max-height: 400px; overflow-y: auto; font-family: 'Consolas', monospace;
}
footer { text-align: center; margin-top: 40px; padding: 20px; color: #aaa; font-size: 0.85em; }
@media (max-width: 768px) {
    .topnav { gap: 2px; padding: 6px 10px; }
    .topnav a { font-size: 0.75em; padding: 4px 8px; }
    .main { padding: 10px; }
    section { padding: 15px; }
    .cards { grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); }
    .charts-grid { grid-template-columns: 1fr; }
    .header { padding: 20px; }
    .lang-selector { position: static; margin-top: 10px; }
}
"""

# =====================================================================
#  JavaScript modules
# =====================================================================

_JS_DRAG_SCROLL = """
(function(){
document.querySelectorAll('.table-wrapper').forEach(function(w){
    var d=false,sX,sL,vX=0,mId;
    function upH(){var h=w.querySelector('.scroll-hint');if(!h)return;
    h.classList.toggle('hidden',w.scrollLeft+w.clientWidth>=w.scrollWidth-2);}
    w.addEventListener('mousedown',function(e){d=true;w.classList.add('dragging');
    sX=e.pageX-w.offsetLeft;sL=w.scrollLeft;vX=0;cancelAnimationFrame(mId);e.preventDefault();});
    w.addEventListener('mouseleave',function(){if(d){d=false;w.classList.remove('dragging');mm();}});
    w.addEventListener('mouseup',function(){if(d){d=false;w.classList.remove('dragging');mm();}});
    w.addEventListener('mousemove',function(e){if(!d)return;var x=e.pageX-w.offsetLeft,wk=(x-sX)*1.5;
    vX=w.scrollLeft;w.scrollLeft=sL-wk;vX=vX-w.scrollLeft;upH();});
    w.addEventListener('scroll',upH);
    function mm(){cancelAnimationFrame(mId);(function s(){vX*=0.92;
    if(Math.abs(vX)>0.5){w.scrollLeft-=vX;upH();mId=requestAnimationFrame(s);}})();}
    var tSX,tSL;
    w.addEventListener('touchstart',function(e){tSX=e.touches[0].pageX;tSL=w.scrollLeft;},{passive:true});
    w.addEventListener('touchmove',function(e){w.scrollLeft=tSL-(e.touches[0].pageX-tSX);upH();},{passive:true});
    upH();
});
})();
"""

_JS_NAV_SCROLL = """
(function(){
var lks=document.querySelectorAll('.topnav a[href^="#"]'),secs=[];
lks.forEach(function(a){var id=a.getAttribute('href').slice(1),el=document.getElementById(id);
if(el)secs.push({el:el,link:a});});
function hl(){var sy=window.scrollY+120,act=null;
secs.forEach(function(s){if(s.el.offsetTop<=sy)act=s;});
lks.forEach(function(a){a.classList.remove('active');});if(act)act.link.classList.add('active');}
window.addEventListener('scroll',hl);hl();
lks.forEach(function(a){a.addEventListener('click',function(e){e.preventDefault();
var t=document.getElementById(a.getAttribute('href').slice(1));
if(t)t.scrollIntoView({behavior:'smooth',block:'start'});});});
})();
"""

_JS_TOOLTIP = """
(function(){
var tip=document.createElement('div');tip.className='f2a-tooltip';document.body.appendChild(tip);
var sT=null,hT=null;
function show(el,ev){var d=el.getAttribute('data-tip');if(!d)return;
var h='';if(el.tagName==='TH'){h='<div class="tip-header">'+el.textContent.trim()+'</div>'+d;}
else{var ci=Array.prototype.indexOf.call(el.parentNode.children,el);
var th=el.closest('table');th=th?th.querySelector('thead'):null;
var cn=th?th.querySelectorAll('tr:first-child th')[ci]:null;cn=cn?cn.textContent.trim():'';
var rl=el.parentNode.children[0]?el.parentNode.children[0].textContent.trim():'';
if(rl)h+='<div class="tip-header">'+rl+' → '+cn+'</div>';
else if(cn)h+='<div class="tip-header">'+cn+'</div>';
h+=d;var cv=el.textContent.trim();
if(cv&&cv!=='NaN'&&cv!=='\u2014')h+='<br><span class="tip-value">Value: '+cv+'</span>';}
tip.innerHTML=h;tip.classList.add('visible');pos(ev);}
function pos(ev){var x=ev.clientX+14,y=ev.clientY+14,tw=tip.offsetWidth,th2=tip.offsetHeight;
if(x+tw>window.innerWidth-10)x=ev.clientX-tw-10;if(y+th2>window.innerHeight-10)y=ev.clientY-th2-10;
if(x<4)x=4;if(y<4)y=4;tip.style.left=x+'px';tip.style.top=y+'px';}
function hide(){tip.classList.remove('visible');}
document.addEventListener('mouseover',function(e){var el=e.target.closest('[data-tip]');
if(!el)return;clearTimeout(hT);sT=setTimeout(function(){show(el,e);},250);});
document.addEventListener('mousemove',function(e){if(tip.classList.contains('visible'))pos(e);});
document.addEventListener('mouseout',function(e){var el=e.target.closest('[data-tip]');
if(!el)return;clearTimeout(sT);hT=setTimeout(hide,120);});
})();
"""

_JS_SUB_TAB = """
function openSubTab(evt,subTabId,groupId){
var g=document.getElementById(groupId);if(!g)return;
g.querySelectorAll('.sub-tab-content').forEach(function(el){el.classList.remove('active');});
g.querySelectorAll('.sub-tab-btn').forEach(function(el){el.classList.remove('active');});
var t=document.getElementById(subTabId);if(t)t.classList.add('active');
evt.currentTarget.classList.add('active');}
"""

_JS_IMG_MODAL = """
(function(){
var ov=document.createElement('div');ov.className='f2a-img-overlay';
ov.innerHTML='<div class="img-viewport"><img></div>'
+'<button class="img-close" aria-label="Close">&times;</button>'
+'<div class="img-title"></div><div class="img-zoom-info"></div>';
document.body.appendChild(ov);
var vp=ov.querySelector('.img-viewport'),img=vp.querySelector('img'),
titleEl=ov.querySelector('.img-title'),zoomInfo=ov.querySelector('.img-zoom-info'),
closeBtn=ov.querySelector('.img-close');
var scale=1,panX=0,panY=0,dragging=false,dSX=0,dSY=0,pSX=0,pSY=0,zT=null;
function ap(){img.style.transform='translate('+panX+'px,'+panY+'px) scale('+scale+')';}
function sz(){zoomInfo.textContent=Math.round(scale*100)+'%';zoomInfo.classList.add('show');
clearTimeout(zT);zT=setTimeout(function(){zoomInfo.classList.remove('show');},900);}
function rv(){var vw=vp.clientWidth,vh=vp.clientHeight,nw=img.naturalWidth,nh=img.naturalHeight;
if(!nw||!nh){scale=1;panX=0;panY=0;ap();return;}
scale=Math.min(vw*0.92/nw,vh*0.88/nh,1);panX=(vw-nw*scale)/2;panY=(vh-nh*scale)/2;ap();}
function openI(src,alt){img.src=src;titleEl.textContent=alt||'';
ov.classList.add('visible');document.body.style.overflow='hidden';
requestAnimationFrame(function(){if(img.naturalWidth){rv();}
else{img.onload=function(){rv();img.onload=null;};}});}
function closeI(){ov.classList.remove('visible');document.body.style.overflow='';}
closeBtn.addEventListener('click',closeI);
document.addEventListener('keydown',function(e){if(e.key==='Escape'&&ov.classList.contains('visible'))closeI();});
ov.addEventListener('click',function(e){if(e.target===ov||e.target===vp)closeI();});
vp.addEventListener('wheel',function(e){e.preventDefault();
var r=vp.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top;
var f=e.deltaY<0?1.12:1/1.12,ns=Math.min(Math.max(scale*f,0.2),12),ra=ns/scale;
panX=mx-ra*(mx-panX);panY=my-ra*(my-panY);scale=ns;ap();sz();},{passive:false});
vp.addEventListener('mousedown',function(e){if(e.button!==0)return;e.preventDefault();
dragging=true;ov.classList.add('dragging');dSX=e.clientX;dSY=e.clientY;pSX=panX;pSY=panY;});
window.addEventListener('mousemove',function(e){if(!dragging)return;
panX=pSX+(e.clientX-dSX);panY=pSY+(e.clientY-dSY);ap();});
window.addEventListener('mouseup',function(){if(dragging){dragging=false;ov.classList.remove('dragging');}});
vp.addEventListener('dblclick',function(e){e.preventDefault();
if(Math.abs(scale-1)<0.01){rv();}else{var r=vp.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top;
var ra=1/scale;panX=mx-ra*(mx-panX);panY=my-ra*(my-panY);scale=1;}ap();sz();});
document.addEventListener('click',function(e){var t=e.target;
if(t.tagName==='IMG'&&(t.closest('.chart-card')||t.closest('.chart-full'))){
e.stopPropagation();openI(t.src,t.alt||'');}});
})();
"""

_JS_I18N = """
var _F2A_I18N={i18n_data};var _f2aLang='{default_lang}';
function f2aSetLang(lang){{if(!_F2A_I18N[lang])lang='en';_f2aLang=lang;
document.querySelectorAll('[data-i18n]').forEach(function(el){{var k=el.getAttribute('data-i18n');
var t=_F2A_I18N[lang][k]||_F2A_I18N['en'][k]||k;
if(el.hasAttribute('data-i18n-html')){{el.innerHTML=t;}}else{{el.textContent=t;}}}});
var sel=document.getElementById('f2a-lang-select');if(sel)sel.value=lang;}}
document.addEventListener('DOMContentLoaded',function(){{var sel=document.getElementById('f2a-lang-select');
if(sel){{sel.addEventListener('change',function(){{f2aSetLang(this.value);}});}}}});
"""


# =====================================================================
#  Chart helper
# =====================================================================

def _chart_img(plot_func, *args, title: str = "", **kwargs) -> str:
    """Call a plot function, convert to base64 <img>. Returns empty on error."""
    if not HAS_MPL:
        return ""
    try:
        fig = plot_func(*args, **kwargs)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                    facecolor="#ffffff", edgecolor="none")
        plt.close(fig)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("utf-8")
        alt = _esc(title) if title else "chart"
        return (f'<div class="chart-full">'
                f'<img src="data:image/png;base64,{b64}" alt="{alt}"></div>')
    except Exception as exc:
        logger.debug("Chart generation failed for %s: %s", title, exc)
        plt.close("all")
        return ""


def _chart_card(plot_func, *args, title: str = "", **kwargs) -> str:
    """Chart inside a styled card."""
    img = _chart_img(plot_func, *args, title=title, **kwargs)
    if not img:
        return ""
    return (f'<div class="chart-card"><h4>{_esc(title)}</h4>'
            f'{img}</div>')


def _extract_numeric(df, max_cols: int = 20) -> dict[str, list[float]]:
    """Extract numeric column data from a pandas DataFrame."""
    if df is None or not HAS_PD:
        return {}
    numeric = df.select_dtypes(include=["number"])
    cols = list(numeric.columns)[:max_cols]
    result: dict[str, list[float]] = {}
    for col in cols:
        vals = numeric[col].dropna().tolist()
        if vals:
            result[col] = vals
    return result


# =====================================================================
#  Section builders — work with JSON dicts from Rust + optional DataFrame
# =====================================================================

def _section_overview(schema: dict, duration: float | None) -> str:
    n_rows = schema.get("n_rows", 0)
    n_cols = schema.get("n_cols", 0)
    mem = schema.get("memory_usage_bytes", 0)
    mem_str = (f"{mem / 1024 / 1024:.1f} MB" if mem > 1024 * 1024
               else f"{mem / 1024:.1f} KB" if mem > 1024 else f"{mem} B")
    columns = schema.get("columns", [])
    type_counts: dict[str, int] = {}
    for c in columns:
        tp = c.get("inferred_type", "unknown")
        type_counts[tp] = type_counts.get(tp, 0) + 1
    cards_data = {"rows": n_rows, "columns": n_cols, "memory": mem_str}
    cards_data.update(type_counts)
    if duration:
        cards_data["duration"] = f"{duration:.2f}s"
    return '<div class="cards">' + _dict_to_cards(cards_data) + "</div>"


def _section_schema(schema: dict) -> str:
    columns = schema.get("columns", [])
    if not columns:
        return ""
    rows = ""
    for col in columns:
        missing_pct = col.get("missing_ratio", 0) * 100
        bar_color = "#27ae60" if missing_pct < 5 else "#f39c12" if missing_pct < 30 else "#e74c3c"
        rows += (
            f'<tr><td><strong>{_esc(col["name"])}</strong></td>'
            f'<td>{_esc(col.get("dtype", ""))}</td>'
            f'<td>{_esc(col.get("inferred_type", ""))}</td>'
            f'<td>{col.get("n_unique", "")}</td>'
            f'<td>{col.get("n_missing", 0)} ({missing_pct:.1f}%)'
            f'<div style="height:3px;border-radius:2px;margin-top:3px;'
            f'width:{min(missing_pct,100):.0f}%;background:{bar_color}"></div></td></tr>'
        )
    return (
        '<div class="table-wrapper"><table class="table">'
        '<thead><tr><th>Column</th><th>DType</th><th>Inferred</th>'
        '<th>Unique</th><th>Missing</th></tr></thead>'
        f'<tbody>{rows}</tbody></table><div class="scroll-hint"></div></div>'
    )


def _section_preprocessing(pp: dict) -> str:
    if not pp:
        return ""
    items = [
        f"Rows: {pp.get('rows_before', '?')} → {pp.get('rows_after', '?')}",
        f"Cols: {pp.get('cols_before', '?')} → {pp.get('cols_after', '?')}",
        f"Duplicates: {pp.get('duplicate_row_count', 0)} "
        f"({pp.get('duplicate_row_ratio', 0)*100:.1f}%)",
    ]
    const_cols = pp.get("constant_columns", [])
    if const_cols:
        items.append(f"Constant columns: {', '.join(const_cols)}")
    id_cols = pp.get("id_like_columns", [])
    if id_cols:
        items.append(f"ID-like columns: {', '.join(id_cols)}")
    li = "".join(f"<li>{_esc(i)}</li>" for i in items)
    return f'<ul class="log-list">{li}</ul>'


# ── Fallback renderer ────────────────────────────────────────────

def _section_generic(_key: str, data: Any) -> str:
    """Render any analysis section from JSON data."""
    if isinstance(data, dict):
        parts: list[str] = []
        scalars = {k: v for k, v in data.items() if not isinstance(v, (dict, list))}
        if scalars:
            parts.append('<div class="cards">' + _dict_to_cards(scalars) + "</div>")
        for sub_key, sub_val in data.items():
            if isinstance(sub_val, list) and sub_val and isinstance(sub_val[0], dict):
                parts.append(f'<h3 class="section-subtitle">{_esc(sub_key.replace("_"," ").title())}</h3>')
                parts.append(_json_table(sub_val))
            elif isinstance(sub_val, dict):
                nested_scalars = {k: v for k, v in sub_val.items() if not isinstance(v, (dict, list))}
                if nested_scalars:
                    parts.append(f'<h3 class="section-subtitle">{_esc(sub_key.replace("_"," ").title())}</h3>')
                    parts.append('<div class="cards">' + _dict_to_cards(nested_scalars) + "</div>")
                for nk, nv in sub_val.items():
                    if isinstance(nv, list) and nv and isinstance(nv[0], dict):
                        parts.append(f'<h3 class="section-subtitle">{_esc(nk.replace("_"," ").title())}</h3>')
                        parts.append(_json_table(nv))
        return "\n".join(parts) if parts else _json_table(data)
    if isinstance(data, list):
        return _json_table(data)
    json_str = json.dumps(data, indent=2, default=str, ensure_ascii=False)
    if len(json_str) > 50_000:
        json_str = json_str[:50_000] + "\n... (truncated)"
    return f'<details><summary>Raw data</summary><pre class="json-pre">{_esc(json_str)}</pre></details>'


# ── Dedicated section renderers (with charts) ────────────────────

def _section_quality(data: dict, df=None) -> str:
    from f2a.viz.plots import plot_quality_radar
    parts: list[str] = [_quality_bars(data)]
    dims = data.get("dimensions", [])
    if dims:
        parts.append(_chart_img(plot_quality_radar, dims, title="Data Quality"))
    parts.append(_section_generic("quality", data))
    return "\n".join(parts)


def _section_descriptive(data: dict, df=None) -> str:
    from f2a.viz.plots import plot_distribution_grid, plot_boxplots, plot_violin
    parts: list[str] = []
    numeric = data.get("numeric", [])
    categorical = data.get("categorical", [])
    if numeric:
        parts.append('<h3 class="section-subtitle">Numeric Columns</h3>')
        parts.append(_json_table(numeric))
    if categorical:
        parts.append('<h3 class="section-subtitle">Categorical Columns</h3>')
        parts.append(_json_table(categorical))
    # Charts from raw DataFrame values
    col_data = _extract_numeric(df)
    if col_data:
        parts.append('<div class="charts-grid">')
        parts.append(_chart_card(plot_distribution_grid, col_data, title="Distribution Histograms"))
        parts.append(_chart_card(plot_boxplots, col_data, title="Boxplots"))
        parts.append(_chart_card(plot_violin, col_data, title="Violin Plots"))
        parts.append('</div>')
    return "\n".join(parts) if parts else _section_generic("descriptive", data)


def _section_distribution(data: dict, df=None) -> str:
    from f2a.viz.plots import plot_distribution_grid, plot_boxplots, plot_violin, plot_qq
    parts: list[str] = []
    # Summary cards
    scalars = {k: v for k, v in data.items() if not isinstance(v, (dict, list))}
    if scalars:
        parts.append('<div class="cards">' + _dict_to_cards(scalars) + "</div>")
    # Table for column stats
    columns = data.get("columns", [])
    if columns:
        parts.append(_json_table(columns))
    # Charts
    col_data = _extract_numeric(df)
    if col_data:
        parts.append('<h3 class="section-subtitle">Distribution Charts</h3>')
        parts.append(_chart_img(plot_distribution_grid, col_data, title="Distribution Histograms"))
        parts.append('<div class="charts-grid">')
        parts.append(_chart_card(plot_boxplots, col_data, title="Boxplots"))
        parts.append(_chart_card(plot_violin, col_data, title="Violin Plots"))
        parts.append('</div>')
        try:
            parts.append(_chart_img(plot_qq, col_data, title="Q-Q Plots"))
        except ImportError:
            pass
    return "\n".join(parts) if parts else _section_generic("distribution", data)


def _section_correlation(data: dict, df=None) -> str:
    from f2a.viz.plots import plot_correlation_heatmap
    parts: list[str] = []
    cols = data.get("columns", [])
    pearson = data.get("pearson", [])
    spearman = data.get("spearman", [])
    if pearson and cols:
        parts.append(_chart_img(plot_correlation_heatmap, pearson, cols,
                                title="Pearson Correlation"))
    if spearman and cols:
        parts.append(_chart_img(plot_correlation_heatmap, spearman, cols,
                                title="Spearman Correlation"))
    # High correlation pairs table
    pairs = data.get("high_correlation_pairs", [])
    if pairs:
        parts.append('<h3 class="section-subtitle">High Correlation Pairs</h3>')
        parts.append(_json_table(pairs))
    # VIF
    vif = data.get("vif", [])
    if vif:
        parts.append('<h3 class="section-subtitle">VIF</h3>')
        parts.append(_json_table(vif))
    # Cramér's V
    cramers = data.get("cramers_v", {})
    if isinstance(cramers, dict) and cramers:
        parts.append('<h3 class="section-subtitle">Cramér\'s V</h3>')
        parts.append(_section_generic("cramers_v", cramers))
    if not parts:
        parts.append(_section_generic("correlation", data))
    return "\n".join(parts)


def _section_missing(data: dict, df=None) -> str:
    from f2a.viz.plots import plot_missing_bar, plot_missing_matrix
    per_col = data.get("per_column", [])
    if not per_col:
        return "<p>No missing values detected.</p>"
    missing = [c for c in per_col if c.get("n_missing", 0) > 0]
    if not missing:
        return "<p>No missing values detected.</p>"
    parts: list[str] = [_json_table(missing)]
    # Missing bar chart
    parts.append(_chart_img(plot_missing_bar, per_col, title="Missing Data"))
    # Missing matrix
    matrix = data.get("missing_matrix", [])
    col_names = data.get("columns", [c.get("column", "") for c in per_col])
    if matrix:
        parts.append(_chart_img(plot_missing_matrix, matrix, col_names,
                                title="Missing Data Matrix"))
    return "\n".join(parts)


def _section_outlier(data: dict, df=None) -> str:
    from f2a.viz.plots import plot_outlier_summary
    parts: list[str] = []
    # Cards / table
    scalars = {k: v for k, v in data.items() if not isinstance(v, (dict, list))}
    if scalars:
        parts.append('<div class="cards">' + _dict_to_cards(scalars) + "</div>")
    iqr_data = data.get("iqr", [])
    if iqr_data:
        parts.append('<h3 class="section-subtitle">IQR Method</h3>')
        parts.append(_json_table(iqr_data))
    zscore_data = data.get("zscore", [])
    if zscore_data:
        parts.append('<h3 class="section-subtitle">Z-Score Method</h3>')
        parts.append(_json_table(zscore_data))
    # Chart with raw data
    col_data = _extract_numeric(df)
    if col_data:
        # Build outlier masks from IQR results
        masks: dict[str, list[bool]] = {}
        for item in iqr_data:
            col = item.get("column", "")
            lb = item.get("lower_bound", float("-inf"))
            ub = item.get("upper_bound", float("inf"))
            if col in col_data:
                masks[col] = [v < lb or v > ub for v in col_data[col]]
        parts.append(_chart_img(plot_outlier_summary, col_data, masks,
                                title="Outlier Detection"))
    if not parts:
        parts.append(_section_generic("outlier", data))
    return "\n".join(parts)


def _section_categorical(data: dict, df=None) -> str:
    parts: list[str] = []
    if isinstance(data, list):
        parts.append(_json_table(data))
    elif isinstance(data, dict):
        for key, val in data.items():
            if isinstance(val, list) and val and isinstance(val[0], dict):
                parts.append(f'<h3 class="section-subtitle">{_esc(key.replace("_"," ").title())}</h3>')
                parts.append(_json_table(val))
            elif isinstance(val, dict):
                scalars = {k: v for k, v in val.items() if not isinstance(v, (dict, list))}
                if scalars:
                    parts.append(f'<h3 class="section-subtitle">{_esc(key.replace("_"," ").title())}</h3>')
                    parts.append('<div class="cards">' + _dict_to_cards(scalars) + "</div>")
    return "\n".join(parts) if parts else _section_generic("categorical", data)


def _section_feature_importance(data: dict, df=None) -> str:
    from f2a.viz.plots import plot_feature_importance
    parts: list[str] = []
    mean_abs = data.get("mean_abs_correlation", [])
    variance = data.get("variance_ranking", [])
    if mean_abs or variance:
        parts.append(_chart_img(plot_feature_importance, mean_abs, variance,
                                title="Feature Importance"))
    if mean_abs:
        parts.append('<h3 class="section-subtitle">Mean Absolute Correlation</h3>')
        parts.append(_json_table(mean_abs))
    if variance:
        parts.append('<h3 class="section-subtitle">Variance Ranking</h3>')
        parts.append(_json_table(variance))
    if not parts:
        parts.append(_section_generic("feature_importance", data))
    return "\n".join(parts)


def _section_pca(data: dict, df=None) -> str:
    from f2a.viz.plots import plot_pca_scree, plot_pca_loadings
    parts: list[str] = []
    components = data.get("components", [])
    loadings = data.get("loadings", [])
    feature_names = data.get("feature_names", [])
    if components:
        var_ratio = [c.get("variance_ratio", 0) for c in components]
        cum_ratio = [c.get("cumulative_ratio", 0) for c in components]
        parts.append(_chart_img(plot_pca_scree, var_ratio, cum_ratio,
                                title="PCA — Scree Plot"))
        parts.append(_json_table(components))
    if loadings and feature_names:
        parts.append(_chart_img(plot_pca_loadings, loadings, feature_names,
                                title="PCA Loadings"))
    if not parts:
        parts.append(_section_generic("pca", data))
    return "\n".join(parts)


def _section_duplicates(data: dict, df=None) -> str:
    parts: list[str] = []
    scalars = {k: v for k, v in data.items() if not isinstance(v, (dict, list))}
    if scalars:
        parts.append('<div class="cards">' + _dict_to_cards(scalars) + "</div>")
    per_col = data.get("per_column_uniqueness", [])
    if per_col:
        parts.append('<h3 class="section-subtitle">Per-Column Uniqueness</h3>')
        parts.append(_json_table(per_col))
    return "\n".join(parts) if parts else _section_generic("duplicates", data)


def _section_insights(data: dict, df=None) -> str:
    from f2a.viz.plots import plot_insight_severity, plot_top_insights
    summary = data.get("summary", {})
    insights = data.get("insights", [])
    parts: list[str] = []
    if summary:
        parts.append('<div class="cards">' + _dict_to_cards(summary) + "</div>")
        parts.append('<div class="charts-grid">')
        parts.append(_chart_card(plot_insight_severity, summary,
                                 title="Insight Severity Distribution"))
        if insights:
            parts.append(_chart_card(plot_top_insights, insights,
                                     title="Top Insights"))
        parts.append('</div>')
    for ins in insights[:30]:
        sev = ins.get("severity", "info")
        color = {"critical": "#e74c3c", "warning": "#f39c12", "info": "#3498db"}.get(sev.lower(), "#95a5a6")
        title = _esc(ins.get("message", ins.get("title", "")))
        col = _esc(ins.get("column", "\u2014"))
        rec = _esc(ins.get("recommendation", ""))
        parts.append(
            f'<div class="insight-item" style="border-left-color:{color}">'
            f'<div style="display:flex;justify-content:space-between;align-items:center">'
            f'<strong>{title}</strong>'
            f'<span style="font-size:0.8em;color:{color};font-weight:bold">{sev.upper()}</span></div>'
            f'<div style="font-size:0.85em;color:#666">Column: {col}</div>'
            + (f'<p style="margin:6px 0 0;font-size:0.9em">{rec}</p>' if rec else "")
            + "</div>"
        )
    return "\n".join(parts)


def _section_ml_readiness(data: dict, df=None) -> str:
    grade = data.get("grade", "?")
    score = data.get("overall_score", 0)
    color = {"A": "#27ae60", "B": "#84cc16", "C": "#f39c12", "D": "#f97316", "F": "#e74c3c"}.get(grade, "#888")
    parts = [
        f'<div style="display:inline-block;background:{color};color:#fff;padding:8px 20px;'
        f'border-radius:8px;font-size:1.3em;font-weight:700;margin-bottom:15px">'
        f'{grade} ({score*100:.0f}%)</div>'
    ]
    dims = data.get("dimensions", [])
    if dims:
        parts.append(_json_table(dims))
    recs = data.get("recommendations", [])
    if recs:
        li = "".join(f"<li>{_esc(r)}</li>" for r in recs)
        parts.append(f'<h3 class="section-subtitle">Recommendations</h3><ul>{li}</ul>')
    return "\n".join(parts)


# ── Advanced section renderers ────────────────────────────────────

def _section_advanced_distribution(data: dict, df=None) -> str:
    from f2a.viz.plots import (plot_best_fit_overlay, plot_ecdf,
                                plot_power_transform, plot_jarque_bera)
    parts: list[str] = []
    col_data = _extract_numeric(df)
    # Best-fit
    best_fit = data.get("best_fit", [])
    if best_fit:
        parts.append('<h3 class="section-subtitle">Best-Fit Distribution</h3>')
        parts.append(_json_table(best_fit))
        if col_data:
            try:
                parts.append(_chart_img(plot_best_fit_overlay, col_data, best_fit,
                                        title="Best-Fit Distribution Overlay"))
            except Exception:
                pass
    # ECDF
    if col_data:
        parts.append(_chart_img(plot_ecdf, col_data, title="ECDF Plot"))
    # Power transforms
    power_transforms = data.get("power_transforms", data.get("power_transform", []))
    if power_transforms:
        parts.append('<h3 class="section-subtitle">Power Transforms</h3>')
        parts.append(_json_table(power_transforms))
        if col_data:
            try:
                parts.append(_chart_img(plot_power_transform, col_data, power_transforms,
                                        title="Power Transform Comparison"))
            except Exception:
                pass
    # Jarque-Bera
    normality = data.get("normality_tests", data.get("jarque_bera", []))
    if normality:
        parts.append('<h3 class="section-subtitle">Normality Tests</h3>')
        parts.append(_json_table(normality))
        parts.append(_chart_img(plot_jarque_bera, normality,
                                title="Jarque-Bera Normality Test"))
    if not parts:
        parts.append(_section_generic("advanced_distribution", data))
    return "\n".join(parts)


def _section_advanced_correlation(data: dict, df=None) -> str:
    from f2a.viz.plots import (plot_partial_correlation, plot_mutual_information,
                                plot_bootstrap_ci, plot_correlation_network,
                                plot_distance_correlation)
    parts: list[str] = []
    # Partial correlation
    partial = data.get("partial_correlation", [])
    if partial:
        parts.append('<h3 class="section-subtitle">Partial Correlation</h3>')
        parts.append(_json_table(partial))
        parts.append(_chart_img(plot_partial_correlation, partial,
                                title="Partial Correlation Heatmap"))
    # Mutual information
    mi = data.get("mutual_information", [])
    if mi:
        parts.append('<h3 class="section-subtitle">Mutual Information</h3>')
        parts.append(_json_table(mi))
        parts.append(_chart_img(plot_mutual_information, mi,
                                title="Mutual Information Heatmap"))
    # Bootstrap CI
    bootstrap = data.get("bootstrap_ci", data.get("bootstrap", []))
    if bootstrap:
        parts.append('<h3 class="section-subtitle">Bootstrap Correlation CI</h3>')
        parts.append(_json_table(bootstrap))
        parts.append(_chart_img(plot_bootstrap_ci, bootstrap,
                                title="Bootstrap Correlation CI"))
    # Correlation network
    network = data.get("correlation_network", data.get("network", []))
    if network:
        parts.append(_chart_img(plot_correlation_network, network,
                                title="Correlation Network"))
    # Distance correlation
    dist_corr = data.get("distance_correlation", [])
    if dist_corr:
        parts.append('<h3 class="section-subtitle">Distance Correlation</h3>')
        parts.append(_json_table(dist_corr))
        parts.append(_chart_img(plot_distance_correlation, dist_corr,
                                title="Distance Correlation Heatmap"))
    if not parts:
        parts.append(_section_generic("advanced_correlation", data))
    return "\n".join(parts)


def _section_clustering(data: dict, df=None) -> str:
    from f2a.viz.plots import (plot_elbow_silhouette, plot_cluster_scatter,
                                plot_dendrogram, plot_cluster_profile)
    parts: list[str] = []
    kmeans = data.get("kmeans", {})
    dbscan = data.get("dbscan", {})
    # K-means
    if kmeans:
        scalars = {k: v for k, v in kmeans.items() if not isinstance(v, (dict, list))}
        if scalars:
            parts.append('<h3 class="section-subtitle">K-Means</h3>')
            parts.append('<div class="cards">' + _dict_to_cards(scalars) + "</div>")
        elbow = kmeans.get("elbow_data", [])
        if elbow:
            parts.append(_chart_img(plot_elbow_silhouette, elbow,
                                    title="Elbow & Silhouette"))
        cluster_sizes = kmeans.get("cluster_sizes", [])
        labels = kmeans.get("labels", [])
        if cluster_sizes:
            parts.append(_chart_img(plot_cluster_profile, cluster_sizes,
                                    title="Cluster Profiles"))
        if labels:
            # Try to get 2D embedding for scatter
            dimred = data.get("_dimred", {})
            tsne_emb = None
            if isinstance(dimred, dict):
                tsne_emb = dimred.get("tsne", {}).get("embedding", [])
            parts.append(_chart_img(plot_cluster_scatter, labels, tsne_emb,
                                    title="Cluster Scatter"))
            parts.append(_chart_img(plot_dendrogram, labels,
                                    title="Dendrogram"))
    # DBSCAN
    if dbscan:
        scalars = {k: v for k, v in dbscan.items() if not isinstance(v, (dict, list))}
        if scalars:
            parts.append('<h3 class="section-subtitle">DBSCAN</h3>')
            parts.append('<div class="cards">' + _dict_to_cards(scalars) + "</div>")
        db_sizes = dbscan.get("cluster_sizes", [])
        if db_sizes:
            parts.append(_chart_img(plot_cluster_profile, db_sizes,
                                    title="DBSCAN Cluster Sizes"))
    if not parts:
        parts.append(_section_generic("clustering", data))
    return "\n".join(parts)


def _section_advanced_dimreduction(data: dict, df=None) -> str:
    from f2a.viz.plots import (plot_pca_biplot, plot_explained_variance_curve,
                                plot_factor_loadings, plot_tsne)
    parts: list[str] = []
    # PCA biplot
    feature_contrib = data.get("feature_contributions", [])
    if feature_contrib:
        # Build loadings matrix + names from feature contributions
        feature_names = sorted({d.get("feature", "") for d in feature_contrib})
        comp_names = sorted({d.get("component", "") for d in feature_contrib})
        if feature_names and comp_names:
            loadings_map: dict[str, dict[str, float]] = {}
            for d in feature_contrib:
                feat = d.get("feature", "")
                comp = d.get("component", "")
                loadings_map.setdefault(feat, {})[comp] = d.get("loading", 0)
            loadings = [[loadings_map.get(f, {}).get(c, 0) for c in comp_names]
                        for f in feature_names]
            var_ratios = data.get("pca_variance_ratio",
                                  [1.0 / max(len(comp_names), 1)] * len(comp_names))
            parts.append(_chart_img(plot_pca_biplot, loadings, feature_names,
                                    var_ratios, title="PCA Biplot"))
            parts.append(_chart_img(plot_explained_variance_curve,
                                    var_ratios, list(np.cumsum(var_ratios)),
                                    title="Explained Variance Curve"))
    # Factor analysis
    fa = data.get("factor_analysis", {})
    if fa:
        fa_loadings = fa.get("loadings", [])
        fa_names = fa.get("feature_names", [])
        n_factors = fa.get("n_factors", 0)
        if fa_loadings and fa_names:
            parts.append(_chart_img(plot_factor_loadings, fa_loadings, fa_names,
                                    n_factors, title="Factor Loadings Heatmap"))
    # t-SNE
    tsne_data = data.get("tsne", {})
    if tsne_data:
        embedding = tsne_data.get("embedding", [])
        if embedding:
            parts.append(_chart_img(plot_tsne, embedding, title="t-SNE Embedding"))
    if not parts:
        parts.append(_section_generic("advanced_dimreduction", data))
    return "\n".join(parts)


def _section_advanced_anomaly(data: dict, df=None) -> str:
    from f2a.viz.plots import plot_anomaly_scatter, plot_consensus_comparison
    parts: list[str] = []
    # Individual methods
    for method_key, method_name in [
        ("isolation_forest", "Isolation Forest"),
        ("local_outlier_factor", "Local Outlier Factor"),
        ("mahalanobis", "Mahalanobis"),
    ]:
        method_data = data.get(method_key, {})
        if method_data:
            scalars = {k: v for k, v in method_data.items()
                       if not isinstance(v, (dict, list))}
            if scalars:
                parts.append(f'<h3 class="section-subtitle">{method_name}</h3>')
                parts.append('<div class="cards">' + _dict_to_cards(scalars) + "</div>")
            scores = method_data.get("scores", [])
            labels = method_data.get("labels", [])
            if scores and labels:
                parts.append(_chart_img(plot_anomaly_scatter, scores, labels,
                                        method_name, title=f"Anomaly Scatter ({method_name})"))
    # Consensus
    consensus = data.get("consensus", {})
    if consensus:
        scalars = {k: v for k, v in consensus.items() if not isinstance(v, (dict, list))}
        if scalars:
            parts.append('<h3 class="section-subtitle">Consensus</h3>')
            parts.append('<div class="cards">' + _dict_to_cards(scalars) + "</div>")
    # Consensus comparison chart
    parts.append(_chart_img(plot_consensus_comparison, data,
                            title="Consensus Anomaly Comparison"))
    if not parts:
        parts.append(_section_generic("advanced_anomaly", data))
    return "\n".join(parts)


def _section_statistical_tests(data: dict, df=None) -> str:
    parts: list[str] = []
    for sub_key, sub_val in data.items():
        title = sub_key.replace("_", " ").title()
        if isinstance(sub_val, list) and sub_val and isinstance(sub_val[0], dict):
            parts.append(f'<h3 class="section-subtitle">{_esc(title)}</h3>')
            parts.append(_json_table(sub_val))
        elif isinstance(sub_val, dict):
            scalars = {k: v for k, v in sub_val.items() if not isinstance(v, (dict, list))}
            if scalars:
                parts.append(f'<h3 class="section-subtitle">{_esc(title)}</h3>')
                parts.append('<div class="cards">' + _dict_to_cards(scalars) + "</div>")
    return "\n".join(parts) if parts else _section_generic("statistical_tests", data)


def _section_cross_analysis(data: dict, df=None) -> str:
    return _section_generic("cross_analysis", data)


def _section_column_role(data: dict, df=None) -> str:
    return _section_generic("column_role", data)


def _section_feature_insights(data: dict, df=None) -> str:
    return _section_generic("feature_insights", data)


# =====================================================================
#  Section registry
# =====================================================================

_BASIC_SECTIONS = [
    "descriptive", "distribution", "correlation", "missing",
    "outlier", "categorical", "feature_importance", "pca",
    "duplicates", "quality",
]

_ADVANCED_SECTIONS = [
    ("insight_engine", "Key Insights"),
    ("advanced_distribution", "Distribution+"),
    ("advanced_correlation", "Correlation+"),
    ("clustering", "Clustering"),
    ("advanced_dimreduction", "Dim. Reduction"),
    ("feature_insights", "Feature Insights"),
    ("cross_analysis", "Cross Analysis"),
    ("advanced_anomaly", "Anomaly+"),
    ("statistical_tests", "Stat Tests"),
    ("column_role", "Column Roles"),
    ("ml_readiness", "ML Readiness"),
]

# All renderers accept (data, df=None)
_SECTION_RENDERERS: dict[str, Any] = {
    "quality": _section_quality,
    "descriptive": _section_descriptive,
    "distribution": _section_distribution,
    "correlation": _section_correlation,
    "missing": _section_missing,
    "outlier": _section_outlier,
    "categorical": _section_categorical,
    "feature_importance": _section_feature_importance,
    "pca": _section_pca,
    "duplicates": _section_duplicates,
    "insight_engine": _section_insights,
    "advanced_distribution": _section_advanced_distribution,
    "advanced_correlation": _section_advanced_correlation,
    "clustering": _section_clustering,
    "advanced_dimreduction": _section_advanced_dimreduction,
    "advanced_anomaly": _section_advanced_anomaly,
    "statistical_tests": _section_statistical_tests,
    "cross_analysis": _section_cross_analysis,
    "column_role": _section_column_role,
    "feature_insights": _section_feature_insights,
    "ml_readiness": _section_ml_readiness,
}


# =====================================================================
#  ReportGenerator
# =====================================================================

class ReportGenerator:
    """Generates a self-contained HTML report from an AnalysisReport."""

    def __init__(self, lang: str = "en"):
        self.lang = lang

    @staticmethod
    def _get_version() -> str:
        from f2a._version import __version__
        return __version__

    def save_html(self, output_path: Path, report: Any) -> None:
        """Write the report as a single HTML file."""
        html = self._build_html(report)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

    # ── Main builder ─────────────────────────────────────────────

    def _build_html(self, report: Any) -> str:
        results = report.results
        schema = report.schema
        pp = report.preprocessing
        duration = report.analysis_duration_sec
        source = report.source
        df = getattr(report, "_dataframe", None)

        # Basic sections
        basic_parts: list[str] = []
        basic_parts.append(self._wrap_section(
            "overview", t("overview", self.lang),
            _section_overview(schema, duration)))
        basic_parts.append(self._wrap_section(
            "schema", t("schema", self.lang),
            _section_schema(schema)))
        if pp:
            basic_parts.append(self._wrap_section(
                "preprocessing", t("preprocessing", self.lang),
                _section_preprocessing(pp)))

        nav_ids = [("overview", t("overview", self.lang)),
                   ("schema", t("schema", self.lang))]
        if pp:
            nav_ids.append(("preprocessing", t("preprocessing", self.lang)))

        for key in _BASIC_SECTIONS:
            if key not in results:
                continue
            title = t(key, self.lang)
            renderer = _SECTION_RENDERERS.get(key)
            if renderer:
                body = renderer(results[key], df)
            else:
                body = _section_generic(key, results[key])
            if body.strip():
                basic_parts.append(self._wrap_section(key, title, body))
                nav_ids.append((key, title))

        basic_html = "\n".join(basic_parts)

        # Advanced sub-tabs
        adv_tabs: list[tuple[str, str, str, str]] = []
        for key, tab_label in _ADVANCED_SECTIONS:
            if key not in results:
                continue
            renderer = _SECTION_RENDERERS.get(key)
            if renderer:
                body = renderer(results[key], df)
            else:
                body = _section_generic(key, results[key])
            if body.strip():
                adv_tabs.append((key, tab_label, t(key, self.lang), body))

        if adv_tabs:
            group_id = "stg-main"
            buttons = [
                '<button class="sub-tab-btn active" '
                f'onclick="openSubTab(event,\'main-basic\',\'{group_id}\')">Basic</button>'
            ]
            contents = [f'<div id="main-basic" class="sub-tab-content active">{basic_html}</div>']

            for key, tab_label, title, body in adv_tabs:
                tab_id = f"main-{key}"
                buttons.append(
                    f'<button class="sub-tab-btn adv" '
                    f'onclick="openSubTab(event,\'{tab_id}\',\'{group_id}\')">'
                    f'{_esc(tab_label)}<span class="adv-badge">ADV</span></button>'
                )
                wrapped = (f'<section><h2 class="section-title">{_esc(title)}'
                           f'<span class="adv-badge">ADV</span></h2>{body}</section>')
                contents.append(f'<div id="{tab_id}" class="sub-tab-content">{wrapped}</div>')

            sections_html = (
                f'<div id="{group_id}">'
                f'<div class="sub-tab-bar">{"".join(buttons)}</div>'
                f'{"".join(contents)}</div>'
            )
        else:
            sections_html = basic_html

        # Navigation links
        nav_links = "".join(
            f'<a href="#{sid}" data-i18n="{sid}">{_esc(label)}</a>'
            for sid, label in nav_ids
        )

        # Language selector
        lang_options = "".join(
            f'<option value="{lg["code"]}"'
            f'{"selected" if lg["code"] == self.lang else ""}>'
            f'{lg.get("name", lg["code"])}</option>'
            for lg in SUPPORTED_LANGUAGES
        )
        lang_selector = (
            '<div class="lang-selector">'
            f'<label>Language</label>'
            f'<select id="f2a-lang-select">{lang_options}</select></div>'
        )

        dataset_name = Path(source).stem
        n_rows = schema.get("n_rows", 0)
        n_cols = schema.get("n_cols", 0)
        meta_html = f'<div class="analysis-meta">Duration: {duration:.2f}s</div>' if duration else ""

        i18n_data = json.dumps(TRANSLATIONS, ensure_ascii=False)
        i18n_js = _JS_I18N.format(i18n_data=i18n_data, default_lang=self.lang)
        version = self._get_version()

        return f"""<!DOCTYPE html>
<html lang="{self.lang}">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>f2a Report \u2014 {_esc(dataset_name)}</title>
<style>{_CSS}</style>
</head>
<body>
<div class="header">
    {lang_selector}
    <h1 data-i18n="report_title">f2a Analysis Report</h1>
    <p>{_esc(dataset_name)} \u2014 {n_rows:,} rows \u00d7 {n_cols} columns</p>
    {meta_html}
</div>
<nav class="topnav">{nav_links}</nav>
<div class="main">
{sections_html}
</div>
<footer>Generated by <strong>f2a</strong> v{version} (File to Analysis)</footer>
<script>{i18n_js}</script>
<script>{_JS_SUB_TAB}</script>
<script>{_JS_DRAG_SCROLL}</script>
<script>{_JS_NAV_SCROLL}</script>
<script>{_JS_TOOLTIP}</script>
<script>{_JS_IMG_MODAL}</script>
</body>
</html>"""

    @staticmethod
    def _wrap_section(section_id: str, title: str, body: str) -> str:
        if not body.strip():
            return ""
        return (
            f'<section id="{section_id}">'
            f'<h2 class="section-title" data-i18n="{section_id}">{_esc(title)}</h2>'
            f'{body}</section>'
        )

    @staticmethod
    def fig_to_base64(fig) -> str:
        """Convert a matplotlib figure to a base64-encoded PNG."""
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)
        return f"data:image/png;base64,{b64}"
