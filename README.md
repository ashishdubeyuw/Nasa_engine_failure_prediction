# NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) FD004 Turbofan Engine Failure Prediction
### MSIS 522 | Analytics & Machine Learning

This project contains a complete, production-quality implementation covering Descriptive Analytics, Predictive Analytics, Explainability, and a final interactive Streamlit dashboard deployment for NASA's C-MAPSS FD004 dataset.

## Project Structure

- `data/` : Contains the raw C-MAPSS dataset (`train_FD004.txt`, etc.)
- `models/` : Stores pre-trained machine learning models and scalers.
- `streamlit_app.py` : The Interactive "Mission Control" web application.

## Setup Instructions

1. **Install Requirements**
   Ensure you have Python 3.9+ installed. For Streamlit Community Cloud deployments, this repo pins Python 3.12.3 via `runtime.txt` because TensorFlow requires a supported Python runtime. Run the following command from the project root:
   ```bash
   pip install -r requirements.txt
   ```

2. **Data Analysis & Model Training**
   To execute the data analysis pipeline, train the models, and build the final Jupyter Notebook:
   ```bash
   python scripts/build_project.py
   ```
   This script will:
   - Compute the Remaining Useful Life (RUL)
   - Perform feature engineering (rolling mean & rolling standard deviation)
   - Train Logistic Regression, Decision Tree, Random Forest, XGBoost and MLP Neural Network
   - Save models to `models/`
   - Generate `MSIS522_CMAPSS_Analysis.ipynb` locally

3. **Explore the Notebook Analysis**
   Open the generated notebook to review exploratory data analysis, visualizations, and the 5 real discoveries documented within the markdown.
   ```bash
   jupyter notebook notebooks/MSIS522_CMAPSS_Analysis.ipynb
   ```

4. **Launch the Mission Control Dashboard**
   The final deliverable is an interactive Streamlit application. Launch it with:
   ```bash
   streamlit run streamlit_app.py
   ```
   This will open the dashboard in your default browser at `localhost:8501`.

## Data Analysis Guidelines

- **File Used**: `train_FD004.txt` (https://data.nasa.gov/docs/legacy/CMAPSSData.zip). This file models multiple operating conditions and multiple failure modes.
- **Features Used**: The codebase automatically dynamically selects valid sensors based on variance filters and executes 20-cycle rolling windows on them. Note: FD004 includes different non-zero variance sensors compared to FD001.
- **Explainability**: SHAP (SHapley Additive exPlanations) is deeply integrated into both the Notebook and the Streamlit App to allow you to understand *why* models predict imminent failure.
- **Real-Time Simulation**: Use the 'Engine Oracle' tab in the Streamlit application to simulate variations in sensor telemetry live and see how it impacts the predicted failure probability dynamically!
