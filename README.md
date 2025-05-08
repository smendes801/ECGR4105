# Space Debris Classification & Analysis

## Overview
This project uses machine learning to classify and analyze space debris using real-world orbital data. The goal is to help reduce the risk of collisions between operational satellites and space debris by leveraging data-driven insights and visualizations.

## Project Goals
- **Classify objects** in orbit as debris or non-debris (satellites)
- **Analyze orbital patterns** and debris characteristics
- **Compare multiple ML models** for classification
- **Visualize** key features, model performance, and orbital distributions
- **Save all plots** to an `images/` directory for easy reporting

## Features
- Automated data loading from Celestrak (satellites & debris)
- Feature engineering (e.g., altitude, SSO flag)
- Model training, evaluation, and comparison (Random Forest, SVM, Logistic Regression, etc.)
- Modular codebase for easy extension and maintenance
- Professional, clear visualizations saved to disk

## File Structure
```
├── main.py              # Main workflow and entry point
├── data_loader.py       # Data acquisition and preprocessing
├── features.py          # Feature engineering and selection
├── models.py            # Model training, evaluation, and comparison
├── visualization.py     # Plotting and image saving utilities
├── utils.py             # Utility and formatting functions
├── images/              # All generated plots and images
├── requirements.txt     # Python dependencies (create if missing)
├── README.md            # This file
```

## Setup
1. **Clone the repository** and navigate to the project directory.
2. **Install dependencies** (recommended: use a virtual environment):
   ```bash
   pip install -r requirements.txt
   ```
   If `requirements.txt` is missing, install these main packages:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn requests sgp4
   ```
3. **Ensure you have internet access** (the code fetches live data from Celestrak).

## Usage
Run the main script:
```bash
python main.py
```
- The script will print progress and results to the terminal.
- All plots and visualizations will be saved in the `images/` directory.

## Output
- **Terminal:** Key statistics, model metrics, and analysis summaries
- **images/**: Contains all generated plots, including:
  - Feature correlation matrix
  - Confusion matrices
  - Feature importance
  - Orbital pattern scatter plots
  - Model performance bar charts

## Customization
- To add new features or models, edit the relevant module (e.g., `features.py`, `models.py`).
- To change data sources, update the URLs in `data_loader.py`.

## Project Motivation
Space debris poses a growing threat to satellites and space operations. By classifying and analyzing debris, this project aims to support the development of safer, more efficient space traffic management and debris mitigation strategies.

---
*For questions or contributions, please open an issue or pull request!* 