from pathlib import Path
import sys
import traceback
import matplotlib.pyplot as plt
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alloy_web.config import ensure_project_imports, TDB_DIR
from alloy_web.ui import (
    element_selector,
    mole_fraction_inputs,
    show_input_summary,
    figure_to_png_bytes,
)
from alloy_web.adapters.phasefield_adapter import (
    run_phase_fraction_temperature_prediction,
)


ensure_project_imports()

st.set_page_config(
    page_title="Phase Fractions",
    layout="wide",
)

st.title("Phase Fractions vs Temperature")

st.markdown(
    """
    Compute phase fractions as a function of temperature for a fixed composition.
    """
)


with st.sidebar:
    st.header("Phase fraction inputs")

    alloy_system = element_selector(
        "Alloy system",
        default=["Cr", "Mo", "Nb", "Ta"],
        min_elements=2,
        max_elements=5,
    )

    temperature_min = st.number_input(
        "Minimum temperature / K",
        min_value=300,
        max_value=4000,
        value=300,
        step=50,
    )

    temperature_max = st.number_input(
        "Maximum temperature / K",
        min_value=300,
        max_value=5000,
        value=3000,
        step=50,
    )

    temperature_step = st.number_input(
        "Temperature step / K",
        min_value=1,
        max_value=500,
        value=100,
        step=100,
    )
    mol_ratio = mole_fraction_inputs(alloy_system)
    run_button = st.button("Run phase fraction prediction", type="primary")


left, right = st.columns([1, 1.4])


if run_button:
    try:
        with left:
            
            
            show_input_summary(
                {
                    "alloy_system": alloy_system,
                    "mol_ratio": mol_ratio,
                    "temperature_min": temperature_min,
                    "temperature_max": temperature_max,
                    "temperature_step": temperature_step,
                    "tdb_dir": str(TDB_DIR),
                }
            )
        with st.spinner("Running phase fraction prediction..."):
            result = run_phase_fraction_temperature_prediction(
                alloy_system=alloy_system,
                mol_ratio=mol_ratio,
                temperature_min=float(temperature_min),
                temperature_max=float(temperature_max),
                temperature_step=float(temperature_step),
                tdb_dir=TDB_DIR,
            )

        with right:
            st.subheader("Phase fraction plot")
            st.pyplot(result.figure, use_container_width=True)

        st.subheader("Phase fraction data")
        st.dataframe(result.data, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            st.download_button(
                "Download CSV",
                data=result.to_csv_bytes(),
                file_name=f"{result.composition}_phase_fractions.csv",
                mime="text/csv",
            )

        with col2:
            st.download_button(
                "Download plot PNG",
                data=figure_to_png_bytes(result.figure),
                file_name=f"{result.composition}_phase_fractions.png",
                mime="image/png",
            )

        plt.close(result.figure)

    except Exception as exc:
        st.error("Phase fraction prediction failed.")
        st.code(str(exc))

        with st.expander("Full traceback"):
            st.code(traceback.format_exc())