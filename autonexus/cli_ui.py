"""Rich terminal presentation for the AutoNexus human-facing CLI."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
from rich import box
from rich.align import Align
from rich.console import Console, Group
from rich.padding import Padding
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.theme import Theme
from rich.terminal_theme import TerminalTheme


NAVY = "#06131d"
CYAN = "#42e8e0"
BLUE = "#5aa7ff"
GREEN = "#74f0a7"
AMBER = "#ffc857"
MUTED = "#7895a5"
WHITE = "#e8f5f7"

THEME = Theme(
    {
        "nexus.title": f"bold {WHITE}",
        "nexus.cyan": f"bold {CYAN}",
        "nexus.blue": f"bold {BLUE}",
        "nexus.green": f"bold {GREEN}",
        "nexus.amber": f"bold {AMBER}",
        "nexus.muted": MUTED,
        "nexus.label": f"bold {MUTED}",
        "nexus.value": WHITE,
    }
)


def _console(*, record: bool = False, width: int | None = None) -> Console:
    force_color = os.getenv("AUTONEXUS_FORCE_COLOR") == "1"
    return Console(
        theme=THEME,
        highlight=False,
        record=record,
        width=width,
        force_terminal=True if force_color or record else None,
        color_system="truecolor" if force_color or record else "auto",
    )


console = _console()


def _gradient_title() -> Text:
    title = Text(justify="center")
    colors = (
        "#46e8dc",
        "#47dfe4",
        "#49d6ec",
        "#4bcdf3",
        "#51befa",
        "#5aafff",
        "#65a0ff",
        "#7194ff",
        "#7e8aff",
        "#8b82ff",
    )
    for index, character in enumerate("Auto Nexus"):
        title.append(character, style=f"bold {colors[index]}")
    return title


def render_banner(target: Console = console) -> None:
    identity = Group(
        Align.center(_gradient_title()),
        Align.center(Text("A D A P T I V E   M O D E L   S Y S T E M", style="nexus.muted")),
        Text(""),
        Align.center(
            Text.assemble(
                (" INGEST ", f"bold {NAVY} on {CYAN}"),
                ("  //  ", "nexus.muted"),
                (" SEARCH ", f"bold {NAVY} on {BLUE}"),
                ("  //  ", "nexus.muted"),
                (" VERIFY ", f"bold {NAVY} on {GREEN}"),
                ("  //  ", "nexus.muted"),
                (" SHIP ", f"bold {NAVY} on {AMBER}"),
            )
        ),
    )
    target.print()
    target.print(
        Panel(
            Padding(identity, (1, 3)),
            title=f"[nexus.cyan]AUTO NEXUS[/] [nexus.muted]// v0.3.1[/]",
            subtitle="[nexus.muted]INTELLIGENCE, WITH EVIDENCE[/]",
            border_style=CYAN,
            box=box.HEAVY,
            style=f"on {NAVY}",
        )
    )


def ask(label: str) -> str:
    console.print(
        f"[nexus.cyan]>[/] [nexus.title]{label}[/]: ",
        end="",
    )
    return input("")


def render_launch(config: Any, target: Console = console) -> None:
    grid = Table.grid(expand=True, padding=(0, 1))
    grid.add_column(style="nexus.label", ratio=1)
    grid.add_column(style="nexus.value", ratio=3)
    modality = "VISION" if config.dataset.is_dir() else "TABULAR"
    task = (config.problem_type or "AUTO DETECT").upper()
    grid.add_row("DATA SOURCE", str(config.dataset))
    grid.add_row("MODE", f"{modality}  /  {task}")
    grid.add_row("OUTPUT", str(config.output_dir))
    grid.add_row("MEMORY", "CONTRIBUTE" if config.contribute_memory else "PRIVATE RUN")
    target.print(
        Panel(
            grid,
            title="[nexus.blue]MISSION CONTROL[/]",
            border_style=BLUE,
            box=box.ROUNDED,
        )
    )


def phase(number: int, title: str, detail: str, target: Console = console) -> None:
    target.print()
    target.rule(
        Text.assemble(
            (f" 0{number} ", f"bold {NAVY} on {CYAN}"),
            (f"  {title.upper()}  ", "nexus.title"),
            (f"{detail} ", "nexus.muted"),
        ),
        style=MUTED,
        align="left",
    )


def event(label: str, message: str, *, tone: str = "cyan") -> None:
    style = {
        "cyan": "nexus.cyan",
        "blue": "nexus.blue",
        "green": "nexus.green",
        "amber": "nexus.amber",
    }.get(tone, "nexus.cyan")
    line = Text()
    line.append(f":: {label.upper():<14}  ", style=style)
    line.append(str(message), style="nexus.value")
    console.print(line)


def _format(value: Any, decimals: int = 4) -> str:
    if value is None:
        return "N/A"
    try:
        numeric = float(value)
        if pd.isna(numeric):
            return "N/A"
        return f"{numeric:.{decimals}f}"
    except (TypeError, ValueError):
        return str(value)


def render_results(
    results: pd.DataFrame,
    problem_type: str,
    target: Console = console,
) -> None:
    preferred = (
        ["model", "accuracy", "precision", "recall", "f1", "roc_auc"]
        if problem_type == "classification"
        else ["model", "rmse", "mae", "r2"]
    )
    columns = [column for column in preferred if column in results.columns]
    table = Table(
        box=box.SIMPLE_HEAD,
        border_style=MUTED,
        header_style="nexus.cyan",
        expand=True,
        pad_edge=False,
    )
    for column in columns:
        table.add_column(
            column.upper().replace("_", " "),
            justify="left" if column == "model" else "right",
        )
    for _, row in results[columns].iterrows():
        table.add_row(
            *[
                str(row[column]) if column == "model" else _format(row[column])
                for column in columns
            ],
            style="nexus.value",
        )
    target.print(
        Panel(
            table,
            title="[nexus.cyan]MODEL LEADERBOARD[/]",
            border_style=CYAN,
            box=box.ROUNDED,
        )
    )


def _key_value_table(rows: list[tuple[str, Any, str]]) -> Table:
    table = Table.grid(expand=True, padding=(0, 1))
    table.add_column(style="nexus.label", ratio=2)
    table.add_column(justify="right", ratio=1)
    for label, value, style in rows:
        table.add_row(label, Text(str(value), style=style))
    return table


def render_final_dashboard(
    *,
    best_model: str,
    metric_rows: list[tuple[str, Any, str]],
    timing_rows: list[tuple[str, Any, str]],
    resource_rows: list[tuple[str, Any, str]],
    artifacts: list[tuple[str, str]],
    representation: str | None = None,
    elapsed_seconds: float,
    target: Console = console,
) -> None:
    winner = Text.assemble(
        ("SELECTED MODEL  ", "nexus.muted"),
        (best_model.upper(), "nexus.green"),
    )
    if representation:
        winner.append("   //   ", style="nexus.muted")
        winner.append(representation.upper(), style="nexus.blue")
    target.print(
        Panel(
            Align.center(winner),
            border_style=GREEN,
            box=box.HEAVY,
            style=f"on {NAVY}",
        )
    )

    metrics = Panel(
        _key_value_table(metric_rows),
        title="[nexus.green]GENERALIZATION[/]",
        border_style=GREEN,
        box=box.ROUNDED,
    )
    timings = Panel(
        _key_value_table(timing_rows),
        title="[nexus.blue]STAGE TELEMETRY[/]",
        border_style=BLUE,
        box=box.ROUNDED,
    )
    target.print(metrics)
    target.print(timings)
    target.print(
        Panel(
            _key_value_table(resource_rows),
            title="[nexus.amber]RESOURCE ENVELOPE[/]",
            border_style=AMBER,
            box=box.ROUNDED,
        )
    )

    artifact_table = Table.grid(expand=True, padding=(0, 1))
    artifact_table.add_column(style="nexus.label", ratio=1)
    artifact_table.add_column(style="nexus.value", ratio=4, overflow="fold")
    for label, path in artifacts:
        artifact_table.add_row(label, path)
    target.print(
        Panel(
            artifact_table,
            title="[nexus.cyan]ARTIFACT MATRIX[/]",
            border_style=CYAN,
            box=box.ROUNDED,
        )
    )
    target.print(
        Align.center(
            Text.assemble(
                ("RUN SEALED", "nexus.green"),
                ("  //  ", "nexus.muted"),
                (f"{elapsed_seconds:.1f}s", "nexus.title"),
                ("  //  reproducible artifacts online", "nexus.muted"),
            )
        )
    )
    target.print()


def save_design_preview(path: str | Path) -> Path:
    """Render a representative dashboard to SVG using the real UI code."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    preview = _console(record=True, width=112)
    render_banner(preview)
    phase(1, "Input Matrix", "dataset recognized / split firewall armed", preview)
    event_preview = Text.assemble(
        (":: DATASET        ", "nexus.cyan"),
        ("12,450 images / 10 classes / grouped split", "nexus.value"),
    )
    preview.print(event_preview)
    phase(2, "Representation Search", "four backbones / successive halving", preview)
    preview.print(
        "[nexus.blue]:: BACKBONE[/]      DINOv2 finalist  "
        "[nexus.green]CV 0.9548[/]  [nexus.muted]NLL 0.1821[/]"
    )
    phase(3, "Generalization Gate", "selection finalized / test unlocked", preview)
    render_results(
        pd.DataFrame(
            [
                {
                    "model": "diverse_ensemble",
                    "accuracy": 0.9624,
                    "precision": 0.9631,
                    "recall": 0.9624,
                    "f1": 0.9622,
                    "roc_auc": 0.9971,
                }
            ]
        ),
        "classification",
        preview,
    )
    render_final_dashboard(
        best_model="diverse_ensemble",
        representation="frozen-dinov2",
        metric_rows=[
            ("FITTED TRAIN", "0.9741", "nexus.value"),
            ("CROSS-VALIDATED", "0.9587", "nexus.green"),
            ("HELD-OUT TEST", "0.9624", "nexus.cyan"),
            ("FIT / CV GAP", "+0.0154", "nexus.amber"),
            ("CV / TEST GAP", "-0.0037", "nexus.green"),
        ],
        timing_rows=[
            ("INPUT PREPARATION", "42.8s", "nexus.value"),
            ("BACKBONE SEARCH", "184.2s", "nexus.value"),
            ("DOWNSTREAM AUTOML", "96.7s", "nexus.value"),
            ("REPORT + NOTEBOOK", "11.4s", "nexus.value"),
        ],
        resource_rows=[
            ("RAM  CURRENT / PEAK", "1,842 / 2,316 MiB", "nexus.amber"),
            ("VRAM CURRENT / PEAK", "16 / 1,584 MiB", "nexus.blue"),
        ],
        artifacts=[
            ("MODEL", "artifacts/model.pkl"),
            ("MANIFEST", "artifacts/run.json"),
            ("ANALYTICS", "artifacts/analysis.ipynb"),
            ("REPORT", "artifacts/report/explanation.md"),
            ("SEARCH", "artifacts/search_profile.json"),
        ],
        elapsed_seconds=335.1,
        target=preview,
    )
    preview.save_svg(
        str(output),
        title="Auto Nexus CLI",
        theme=TerminalTheme(
            (3, 12, 18),
            (232, 245, 247),
            [(120, 149, 165)] * 8,
            [(66, 232, 224)] * 8,
        ),
    )
    return output
