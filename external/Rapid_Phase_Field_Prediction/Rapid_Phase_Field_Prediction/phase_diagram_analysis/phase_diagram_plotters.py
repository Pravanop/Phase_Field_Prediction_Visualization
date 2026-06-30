from pathlib import Path
from typing import Optional

import itertools
import numpy as np
import matplotlib.pyplot as plt

from pycalphad import Database, binplot, ternplot
from pycalphad import variables as v



MODULE_ROOT = Path(__file__).resolve().parents[1]


def _resolve_tdb_path(
    composition: str,
    tdb_dir: Optional[str | Path] = None,
    tdb_path: Optional[str | Path] = None,
    allow_tdb_generation: bool = False,
) -> Path:
    """
    Resolve the TDB path for a given composition.

    Priority:
    1. Explicit tdb_path
    2. Explicit tdb_dir / composition.tdb
    3. Common local repo folders
    4. Optional TDBHandler fallback
    """

    if tdb_path is not None:
        path = Path(tdb_path)
        if not path.exists():
            raise FileNotFoundError(f"TDB path does not exist: {path}")
        return path

    candidates = []

    if tdb_dir is not None:
        candidates.append(Path(tdb_dir) / f"{composition}.tdb")

    candidates.extend(
        [
            MODULE_ROOT / "tdbs" / f"{composition}.tdb",
            MODULE_ROOT / "tdb_db" / f"{composition}.tdb",
            MODULE_ROOT / "tdb_version" / "tdb_db_reg" / f"{composition}.tdb",
        ]
    )

    for path in candidates:
        if path.exists():
            return path

        generated_path = MODULE_ROOT / "tdb_db" / f"{composition}.tdb"
        if generated_path.exists():
            return generated_path

    raise FileNotFoundError(
        f"No TDB found for {composition}. Checked:\n"
        + "\n".join(str(path) for path in candidates)
    )


def _get_database(
    composition: str,
    tdb_dir: Optional[str | Path] = None,
    tdb_path: Optional[str | Path] = None,
    allow_tdb_generation: bool = False,
) -> Database:
    path = _resolve_tdb_path(
        composition=composition,
        tdb_dir=tdb_dir,
        tdb_path=tdb_path,
        allow_tdb_generation=allow_tdb_generation,
    )
    return Database(str(path))


def _filter_phases(phases: list[str], include_intermetallics: bool = True) -> list[str]:
    if include_intermetallics:
        return phases

    return [phase for phase in phases if "MP" not in phase]


def generate_binary_phase_diagram(
    composition: str,
    ax=None,
    tdb_dir: Optional[str | Path] = None,
    tdb_path: Optional[str | Path] = None,
    include_intermetallics: bool = True,
    x_element: Optional[str] = None,
    temperature_min: float = 300,
    temperature_max: float = 3000,
    temperature_step: float = 10,
    composition_step: float = 0.02,
    pressure: float = 101325,
    invert_x_axis: bool = True,
    allow_tdb_generation: bool = False,
):
    """
    Generate a binary temperature-composition phase diagram.

    Parameters
    ----------
    composition
        Binary system label, e.g. "Cr-Mo".

    ax
        Optional matplotlib axis.

    tdb_dir / tdb_path
        TDB location.

    include_intermetallics
        Whether to include phases containing "MP".

    x_element
        Element to use on the composition x-axis.
        Defaults to the second element.

    Returns
    -------
    fig, ax
    """

    elements = composition.split("-")

    if len(elements) != 2:
        raise ValueError(
            f"Binary phase diagram requires 2 elements. Got {elements}."
        )

    dbf = _get_database(
        composition=composition,
        tdb_dir=tdb_dir,
        tdb_path=tdb_path,
        allow_tdb_generation=allow_tdb_generation,
    )

    comps = [el.upper() for el in elements] + ["VA"]
    phases = _filter_phases(
        list(dbf.phases.keys()),
        include_intermetallics=include_intermetallics,
    )

    if x_element is None:
        x_element = elements[1]

    if x_element not in elements:
        raise ValueError(
            f"x_element={x_element} is not in composition={composition}."
        )

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.figure

    conditions = {
        v.X(x_element.upper()): (0, 1, composition_step),
        v.T: (temperature_min, temperature_max, temperature_step),
        v.P: pressure,
        v.N: 1,
    }

    binplot(
        dbf,
        comps,
        phases,
        conditions,
        plot_kwargs={
            "ax": ax,
            "tieline_color": (0.8, 0.8, 0.8, 0.9),
            "zorder": 0,
        },
    )

    legend = ax.get_legend()
    if legend is not None:
        legend.remove()

    ax.set_ylabel("Temperature (K)")
    ax.set_xlabel(f"X({x_element})")
    ax.set_ylim(temperature_min, temperature_max)

    if invert_x_axis:
        ax.set_xlim(1, 0)
    else:
        ax.set_xlim(0, 1)

    fig.tight_layout()

    return fig, ax


def generate_ternary_phase_diagram(
    composition: str,
    temperature: float,
    order: int = 0,
    ax=None,
    tdb_dir: Optional[str | Path] = None,
    tdb_path: Optional[str | Path] = None,
    include_intermetallics: bool = True,
    composition_step: float = 0.015,
    pressure: float = 101325,
    label_nodes: bool = True,
    tielines: int = 1,
    allow_tdb_generation: bool = False,
):
    """
    Generate a ternary isothermal phase diagram.

    Parameters
    ----------
    composition
        Ternary system label, e.g. "Cr-Mo-Nb".

    temperature
        Temperature in K.

    order
        Which permutation of the element order to use for the ternary plot.

    ax
        Optional ternary matplotlib axis.

    Returns
    -------
    fig, ax, strategy
    """

    elements = composition.split("-")

    if len(elements) != 3:
        raise ValueError(
            f"Ternary phase diagram requires 3 elements. Got {elements}."
        )

    dbf = _get_database(
        composition=composition,
        tdb_dir=tdb_dir,
        tdb_path=tdb_path,
        allow_tdb_generation=allow_tdb_generation,
    )

    element_permutations = [list(p) for p in itertools.permutations(elements)]

    if order < 0 or order >= len(element_permutations):
        raise ValueError(
            f"order must be between 0 and {len(element_permutations) - 1}."
        )

    ordered_elements = element_permutations[order]
    comps = [el.upper() for el in ordered_elements] + ["VA"]

    phases = _filter_phases(
        list(dbf.phases.keys()),
        include_intermetallics=include_intermetallics,
    )

    conditions = {
        v.T: temperature,
        v.P: pressure,
        v.X(comps[0]): (0, 1, composition_step),
        v.X(comps[2]): (0, 1, composition_step),
    }

    plot_kwargs = {}
    if ax is not None:
        plot_kwargs["ax"] = ax

    strategy = ternplot(
        dbf,
        comps,
        phases,
        conditions,
        x=v.X(comps[2]),
        y=v.X(comps[1]),
        label_nodes=label_nodes,
        tieline_color=(1, 1, 1, 1),
        tielines=tielines,
        plot_kwargs=plot_kwargs,
        return_strategy=True,
    )

    if ax is None:
        fig = plt.gcf()
        ax = plt.gca()
    else:
        fig = ax.figure

    fig.tight_layout()

    return fig, ax, strategy