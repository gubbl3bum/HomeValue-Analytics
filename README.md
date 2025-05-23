# HomeValue-Analytics

Analytic tool for housing prices from given CSV files. Application provides comprehensive data analysis, visualization, and machine learning capabilities for real estate data (and other also).

## Features

* **Data Loading & Preview**
  * CSV file import support
  * Interactive data preview with in-place editing
  * Dynamic data type detection

* **Data Filtering**
  * Numeric range filters
  * Categorical value filters
  * Missing value handling
  * Subtable extraction (general filtering for exploration)
  * Machine learning-specific extraction (feature selection, target variable preparation)

* **Statistical Analysis**
  * **Descriptive Statistics**: Mean, median, standard deviation, etc.
  * **Correlation Analysis**: Correlation heatmaps and pairwise correlations.
  * **Categorical Analysis**: Frequency distributions, price per square meter, etc.

* **Data Visualization**
  * Scatter plots
  * Histograms
  * Box plots
  * Bar charts
  * Line charts
  * Correlation heatmaps
  * Pie charts

* **Machine Learning**
  * K-means clustering
  * Linear/Logistic regression
  * Random Forest models
  * Feature importance analysis
  * Model performance metrics

## Requirements

* Python 3.12 or newer
* Dependencies listed in `requirements.txt`

## Dataset Requirements

The CSV file should contain columns such as:

* Price (numeric)
* Area (numeric)
* Location (categorical)
* Additional features (rooms, year built, etc.)

Example dataset can be found on [Kaggle](https://www.kaggle.com/datasets/krzysztofjamroz/apartment-prices-in-poland)

## Setup Instructions

### Create/Activate Virtual Environment

Every script requires to be run from `.venv`:

```shell
python3 -m venv .venv

source ./.venv/Scripts/activate # activate venv (Linux)
.\.venv\Scripts\Activate.ps1    # activate venv (Windows)

pip install -r requirements.txt # download dependencies
```

### Development Mode

Needs to have activated `.venv`.

```shell
streamlit run src/main.py         # run in browser
python src/app.py                 # run desktop view
streamlit run src/main.py --server.enableStaticServing true  # debug mode
```

### Build Desktop Application

```shell
cd ~/HomeValue-Analytics         # go to project root
pyinstaller --onefile --clean --add-binary ".venv/Scripts/streamlit.exe;." --add-data "src;src" src/app.py # create executable
```

### Production Mode

```shell
cd dist/                         # go to distribution directory
HomeValue-Analytics.exe          # run the application
```

## Usage Guide

1. **Data Loading**
   * Launch the application
   * Click "Choose File" to upload your CSV
   * Preview and edit data if needed

2. **Data Filtering**
   * Use numeric filters for price ranges
   * Select categorical values for filtering
   * Extract specific rows/columns
   * Apply or reset filters as needed

3. **Analysis**
   * Select columns for statistical analysis
   * Generate visualizations
   * Perform machine learning analysis

4. **Export Results**
   * Download filtered data
   * Save generated charts
   * Export analysis reports

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
