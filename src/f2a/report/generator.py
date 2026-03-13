"""HTML report generation module."""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from f2a.utils.logging import get_logger

logger = get_logger(__name__)


def _fig_to_base64(fig: plt.Figure) -> str:
    """Convert a matplotlib Figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


def _df_to_html(df: pd.DataFrame, max_rows: int = 50) -> str:
    """Convert a DataFrame to an HTML table string."""
    return df.head(max_rows).to_html(classes="table", border=0, float_format="%.4f")


# Shared drag-to-scroll JavaScript (inserted at the end of <body>)
_DRAG_SCROLL_JS = """    <script>
    (function() {
        document.querySelectorAll('.table-wrapper').forEach(function(wrapper) {
            var isDown = false, startX, scrollLeft, velX = 0, momentumId;

            function updateHint() {
                var hint = wrapper.querySelector('.scroll-hint');
                if (!hint) return;
                var atEnd = wrapper.scrollLeft + wrapper.clientWidth >= wrapper.scrollWidth - 2;
                hint.classList.toggle('hidden', atEnd);
            }

            wrapper.addEventListener('mousedown', function(e) {
                isDown = true;
                wrapper.classList.add('dragging');
                startX = e.pageX - wrapper.offsetLeft;
                scrollLeft = wrapper.scrollLeft;
                velX = 0;
                cancelAnimationFrame(momentumId);
                e.preventDefault();
            });
            wrapper.addEventListener('mouseleave', function() {
                if (isDown) { isDown = false; wrapper.classList.remove('dragging'); startMomentum(); }
            });
            wrapper.addEventListener('mouseup', function() {
                if (isDown) { isDown = false; wrapper.classList.remove('dragging'); startMomentum(); }
            });
            wrapper.addEventListener('mousemove', function(e) {
                if (!isDown) return;
                var x = e.pageX - wrapper.offsetLeft;
                var walk = (x - startX) * 1.5;
                velX = wrapper.scrollLeft;
                wrapper.scrollLeft = scrollLeft - walk;
                velX = velX - wrapper.scrollLeft;
                updateHint();
            });
            wrapper.addEventListener('scroll', updateHint);

            function startMomentum() {
                cancelAnimationFrame(momentumId);
                (function step() {
                    velX *= 0.92;
                    if (Math.abs(velX) > 0.5) {
                        wrapper.scrollLeft -= velX;
                        updateHint();
                        momentumId = requestAnimationFrame(step);
                    }
                })();
            }

            // Touch support
            var touchStartX, touchScrollLeft;
            wrapper.addEventListener('touchstart', function(e) {
                touchStartX = e.touches[0].pageX;
                touchScrollLeft = wrapper.scrollLeft;
            }, {passive: true});
            wrapper.addEventListener('touchmove', function(e) {
                var dx = e.touches[0].pageX - touchStartX;
                wrapper.scrollLeft = touchScrollLeft - dx;
                updateHint();
            }, {passive: true});

            // Init hint visibility
            updateHint();
        });
    })();
    </script>"""


class ReportGenerator:
    """Generate HTML reports from analysis results."""

    def generate_html(
        self,
        dataset_name: str,
        schema_summary: dict[str, Any],
        stats_df: pd.DataFrame,
        figures: dict[str, plt.Figure],
        warnings: list[str] | None = None,
    ) -> str:
        """Generate an HTML report string.

        Args:
            dataset_name: Dataset name.
            schema_summary: Schema summary dictionary.
            stats_df: Summary statistics DataFrame.
            figures: Name → Figure mapping.
            warnings: List of warning messages.

        Returns:
            HTML string.
        """
        warnings = warnings or []

        # Figure를 base64로 변환
        figure_html_parts: list[str] = []
        for name, fig in figures.items():
            b64 = _fig_to_base64(fig)
            figure_html_parts.append(
                f'<div class="chart-container">'
                f'<h3>{name}</h3>'
                f'<img src="data:image/png;base64,{b64}" alt="{name}" />'
                f"</div>"
            )

        figures_section = "\n".join(figure_html_parts)
        stats_html = _df_to_html(stats_df) if not stats_df.empty else "<p>No statistics data</p>"
        warnings_html = "".join(f"<li>{w}</li>" for w in warnings)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>f2a Analysis Report — {dataset_name}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
               line-height: 1.6; color: #333; max-width: 1200px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; margin-bottom: 20px; }}
        h2 {{ color: #34495e; margin-top: 30px; margin-bottom: 15px; }}
        h3 {{ color: #7f8c8d; margin-bottom: 10px; }}
        .overview {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                     gap: 15px; margin: 20px 0; }}
        .stat-card {{ background: #f8f9fa; border-radius: 8px; padding: 15px; text-align: center; }}
        .stat-card .value {{ font-size: 2em; font-weight: bold; color: #3498db; }}
        .stat-card .label {{ font-size: 0.9em; color: #7f8c8d; }}
        .table-wrapper {{ position: relative; overflow-x: auto; overflow-y: visible;
                        margin: 15px 0; border: 1px solid #e0e0e0; border-radius: 8px;
                        cursor: grab; -webkit-user-select: none; user-select: none; }}
        .table-wrapper.dragging {{ cursor: grabbing; }}
        .table-wrapper .scroll-hint {{ position: absolute; top: 0; right: 0; bottom: 0; width: 40px;
                        pointer-events: none; background: linear-gradient(to right, transparent, rgba(0,0,0,0.06));
                        border-radius: 0 8px 8px 0; transition: opacity 0.3s; }}
        .table-wrapper .scroll-hint.hidden {{ opacity: 0; }}
        .table {{ width: max-content; min-width: 100%; border-collapse: collapse; font-size: 0.9em; }}
        .table th, .table td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #ddd; white-space: nowrap; }}
        .table th {{ background: #f8f9fa; font-weight: 600; position: sticky; top: 0; z-index: 1; }}
        .table th:first-child {{ position: sticky; left: 0; z-index: 2; background: #eef2f5; }}
        .table td:first-child {{ position: sticky; left: 0; background: #fff; z-index: 1;
                                 font-weight: 500; border-right: 2px solid #e0e0e0; }}
        .table tr:hover td {{ background: #f1f3f5; }}
        .table tr:hover td:first-child {{ background: #e8ecf0; }}
        .chart-container {{ margin: 20px 0; text-align: center; }}
        .chart-container img {{ max-width: 100%; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .warnings {{ background: #fff3cd; border: 1px solid #ffc107; border-radius: 8px;
                     padding: 15px; margin: 20px 0; }}
        .warnings li {{ margin-left: 20px; }}
        footer {{ text-align: center; margin-top: 40px; padding: 20px; color: #aaa; font-size: 0.85em; }}
    </style>
</head>
<body>
    <h1>f2a Analysis Report</h1>
    <p>Dataset: <strong>{dataset_name}</strong></p>

    <h2>Overview</h2>
    <div class="overview">
        <div class="stat-card">
            <div class="value">{schema_summary.get('rows', 'N/A'):,}</div>
            <div class="label">Rows</div>
        </div>
        <div class="stat-card">
            <div class="value">{schema_summary.get('columns', 'N/A')}</div>
            <div class="label">Columns</div>
        </div>
        <div class="stat-card">
            <div class="value">{schema_summary.get('numeric', 0)}</div>
            <div class="label">Numeric</div>
        </div>
        <div class="stat-card">
            <div class="value">{schema_summary.get('categorical', 0)}</div>
            <div class="label">Categorical</div>
        </div>
        <div class="stat-card">
            <div class="value">{schema_summary.get('memory_mb', 0)} MB</div>
            <div class="label">Memory</div>
        </div>
    </div>

    <h2>Summary Statistics</h2>
    <div class="table-wrapper">
        {stats_html}
        <div class="scroll-hint"></div>
    </div>

    <h2>Visualizations</h2>
    {figures_section}

    {"<h2>Warnings</h2><div class='warnings'><ul>" + warnings_html + "</ul></div>" if warnings else ""}

    <footer>
        Generated by <strong>f2a</strong> (File to Analysis)
    </footer>
{_DRAG_SCROLL_JS}
</body>
</html>"""
        return html

    def save_html(
        self,
        output_path: str | Path,
        **kwargs: Any,
    ) -> Path:
        """Save an HTML report to a file.

        Args:
            output_path: Output file path.
            **kwargs: Arguments passed to ``generate_html``.

        Returns:
            Path to the saved file.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        html = self.generate_html(**kwargs)
        path.write_text(html, encoding="utf-8")
        logger.info("Report saved: %s", path)
        return path

    # ── Multi-subset report ───────────────────────────────────

    def generate_html_multi(
        self,
        dataset_name: str,
        sections: list[dict[str, Any]],
    ) -> str:
        """Generate an HTML report with multiple subset/split sections.

        Args:
            dataset_name: Dataset name.
            sections: List of dicts, each containing ``subset``, ``split``,
                ``schema_summary``, ``stats_df``, ``figures``, ``warnings``.

        Returns:
            HTML string.
        """
        # Build per-section HTML
        tab_buttons: list[str] = []
        tab_contents: list[str] = []

        for idx, sec in enumerate(sections):
            tab_id = f"tab-{idx}"
            label = f"{sec['subset']} / {sec['split']}"
            active = "active" if idx == 0 else ""

            tab_buttons.append(
                f'<button class="tab-btn {active}" onclick="openTab(event, \'{tab_id}\')">'
                f"{label}</button>"
            )

            schema = sec["schema_summary"]
            stats_html = (
                _df_to_html(sec["stats_df"])
                if not sec["stats_df"].empty
                else "<p>No statistics data</p>"
            )

            figure_parts: list[str] = []
            for name, fig in sec.get("figures", {}).items():
                b64 = _fig_to_base64(fig)
                figure_parts.append(
                    f'<div class="chart-container">'
                    f"<h3>{name}</h3>"
                    f'<img src="data:image/png;base64,{b64}" alt="{name}" />'
                    f"</div>"
                )
            figures_section = "\n".join(figure_parts)

            warnings = sec.get("warnings", [])
            warnings_html = ""
            if warnings:
                items = "".join(f"<li>{w}</li>" for w in warnings)
                warnings_html = (
                    f"<h3>Warnings</h3><div class='warnings'><ul>{items}</ul></div>"
                )

            tab_contents.append(f"""
    <div id="{tab_id}" class="tab-content" style="display: {'block' if idx == 0 else 'none'};">
        <h2>{label}</h2>
        <div class="overview">
            <div class="stat-card">
                <div class="value">{schema.get('rows', 'N/A'):,}</div>
                <div class="label">Rows</div>
            </div>
            <div class="stat-card">
                <div class="value">{schema.get('columns', 'N/A')}</div>
                <div class="label">Columns</div>
            </div>
            <div class="stat-card">
                <div class="value">{schema.get('numeric', 0)}</div>
                <div class="label">Numeric</div>
            </div>
            <div class="stat-card">
                <div class="value">{schema.get('categorical', 0)}</div>
                <div class="label">Categorical</div>
            </div>
            <div class="stat-card">
                <div class="value">{schema.get('memory_mb', 0)} MB</div>
                <div class="label">Memory</div>
            </div>
        </div>
        <h3>Summary Statistics</h3>
        <div class="table-wrapper">
            {stats_html}
            <div class="scroll-hint"></div>
        </div>
        {f'<h3>Visualizations</h3>{figures_section}' if figures_section else ''}
        {warnings_html}
    </div>""")

        total_rows = sum(s["schema_summary"].get("rows", 0) for s in sections)
        tabs_html = "\n".join(tab_buttons)
        content_html = "\n".join(tab_contents)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>f2a Analysis Report — {dataset_name}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
               line-height: 1.6; color: #333; max-width: 1200px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; margin-bottom: 20px; }}
        h2 {{ color: #34495e; margin-top: 20px; margin-bottom: 15px; }}
        h3 {{ color: #7f8c8d; margin: 15px 0 10px 0; }}
        .overview {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                     gap: 15px; margin: 20px 0; }}
        .stat-card {{ background: #f8f9fa; border-radius: 8px; padding: 15px; text-align: center; }}
        .stat-card .value {{ font-size: 2em; font-weight: bold; color: #3498db; }}
        .stat-card .label {{ font-size: 0.9em; color: #7f8c8d; }}
        .table-wrapper {{ position: relative; overflow-x: auto; overflow-y: visible;
                        margin: 15px 0; border: 1px solid #e0e0e0; border-radius: 8px;
                        cursor: grab; -webkit-user-select: none; user-select: none; }}
        .table-wrapper.dragging {{ cursor: grabbing; }}
        .table-wrapper .scroll-hint {{ position: absolute; top: 0; right: 0; bottom: 0; width: 40px;
                        pointer-events: none; background: linear-gradient(to right, transparent, rgba(0,0,0,0.06));
                        border-radius: 0 8px 8px 0; transition: opacity 0.3s; }}
        .table-wrapper .scroll-hint.hidden {{ opacity: 0; }}
        .table {{ width: max-content; min-width: 100%; border-collapse: collapse; font-size: 0.9em; }}
        .table th, .table td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #ddd; white-space: nowrap; }}
        .table th {{ background: #f8f9fa; font-weight: 600; position: sticky; top: 0; z-index: 1; }}
        .table th:first-child {{ position: sticky; left: 0; z-index: 2; background: #eef2f5; }}
        .table td:first-child {{ position: sticky; left: 0; background: #fff; z-index: 1;
                                 font-weight: 500; border-right: 2px solid #e0e0e0; }}
        .table tr:hover td {{ background: #f1f3f5; }}
        .table tr:hover td:first-child {{ background: #e8ecf0; }}
        .chart-container {{ margin: 20px 0; text-align: center; }}
        .chart-container img {{ max-width: 100%; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .warnings {{ background: #fff3cd; border: 1px solid #ffc107; border-radius: 8px;
                     padding: 15px; margin: 15px 0; }}
        .warnings li {{ margin-left: 20px; }}
        footer {{ text-align: center; margin-top: 40px; padding: 20px; color: #aaa; font-size: 0.85em; }}
        /* Tabs */
        .tab-bar {{ display: flex; flex-wrap: wrap; gap: 4px; border-bottom: 2px solid #e0e0e0;
                     margin: 20px 0 0 0; padding-bottom: 0; }}
        .tab-btn {{ padding: 10px 20px; border: 1px solid #ddd; border-bottom: none;
                    background: #f8f9fa; cursor: pointer; border-radius: 8px 8px 0 0;
                    font-size: 0.95em; transition: background 0.15s; }}
        .tab-btn:hover {{ background: #e9ecef; }}
        .tab-btn.active {{ background: #fff; border-bottom: 2px solid #fff; margin-bottom: -2px;
                           font-weight: 600; color: #3498db; }}
        .tab-content {{ padding: 20px 0; }}
        .summary-bar {{ background: #eaf3fb; border-radius: 8px; padding: 12px 20px;
                        margin: 10px 0 20px 0; font-size: 1.05em; }}
    </style>
</head>
<body>
    <h1>f2a Analysis Report</h1>
    <p>Dataset: <strong>{dataset_name}</strong></p>
    <div class="summary-bar">
        Total: <strong>{total_rows:,}</strong> rows across <strong>{len(sections)}</strong> subsets / splits
    </div>

    <div class="tab-bar">
        {tabs_html}
    </div>
    {content_html}

    <footer>
        Generated by <strong>f2a</strong> (File to Analysis)
    </footer>

    <script>
    function openTab(evt, tabId) {{
        document.querySelectorAll('.tab-content').forEach(el => el.style.display = 'none');
        document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
        document.getElementById(tabId).style.display = 'block';
        evt.currentTarget.classList.add('active');
    }}
    </script>
{_DRAG_SCROLL_JS}
</body>
</html>"""
        return html

    def save_html_multi(
        self,
        output_path: str | Path,
        **kwargs: Any,
    ) -> Path:
        """Save a multi-subset HTML report to a file.

        Args:
            output_path: Output file path.
            **kwargs: Arguments passed to ``generate_html_multi``.

        Returns:
            Path to the saved file.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        html = self.generate_html_multi(**kwargs)
        path.write_text(html, encoding="utf-8")
        logger.info("Report saved: %s", path)
        return path
