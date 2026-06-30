import streamlit as st

from alloy_web.config import ensure_project_imports


ensure_project_imports()

st.set_page_config(
    page_title="Alloy Phase Diagram Predictor",
    layout="wide",
)

st.title("Rapid and Accurate Prediction of Phase Fields")

st.markdown(
    """
    Interactive web interface for phase-diagram and solid solution stability prediction.

    Use the sidebar to choose a module:

    - **SymPlex Maps**: Quaternary/quinary property maps.
    - **Phase Fractions**: Phase fractions vs. temperature for a fixed composition.
    - **Phase Diagrams**: Binary and ternary phase diagrams.
    """
)

st.info(
	"""
	Please cite these papers if you use the data from this website:
	- [1] Prediction of Phase Fields
	- [2] SymPlex
	- [3] PyCalphad
	- [4] Materials Project
	
	"""
)