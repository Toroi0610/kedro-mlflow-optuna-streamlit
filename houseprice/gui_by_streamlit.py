from subprocess import call
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
        if st.button('Run'):
            call("optuna dashboard --study-name distributed-example --storage sqlite:///example.db")

    if viz_type == "MLflow":
        if st.button('Run'):
            call("mlflow ui")

    if viz_type == "Kedro Viz":
        if st.button('Run'):
            call("kedro viz")


if __name__ == "__main__":
    main()