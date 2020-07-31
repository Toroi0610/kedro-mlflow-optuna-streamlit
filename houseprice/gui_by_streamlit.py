from subprocess import Popen
import streamlit as st

# sidemenu
st.sidebar.markdown(
    "# Visualize Experiments"
)

viz_type = st.sidebar.selectbox(
    "Viz type", ["Optuna dashboard",
                 "MLflow",
                 "Kedro Viz"]
)

def main():
    if viz_type == "Optuna dashboard":
        if st.sidebar.button('Run'):
            Popen("optuna dashboard --study-name distributed-example --storage sqlite:///example.db")

    if viz_type == "MLflow":
        if st.sidebar.button('Run'):
            Popen("mlflow ui")

    if viz_type == "Kedro Viz":
        if st.sidebar.button('Run'):
            Popen("kedro viz")


if __name__ == "__main__":
    main()