# Rossmann Pharmaceuticals Sales Predictions Project

## Story Line

The finance team from Rossmann wants to forecast sales in all their stores across several cities six weeks ahead of time. Managers in individual stores rely on their years of experience as well as their personal judgment to forecast sales. 

The data team identified factors such as promotions, competition, school and state holidays, seasonality, and locality as necessary for predicting the sales across the various stores.

The job is to build and serve an end-to-end product that delivers this prediction to analysts in the finance team. 


## Table of Contents

1. [Project Structure](#project-structure)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Data Description](#data-description)
5. [Analysis Components](#analysis-components)
6. [Running Tests](#running-tests)
7. [Contributing](#contributing)
8. [License](#license)

## Project Structure

```
project/
├── .vscode/
│   └── settings.json
├── .github/
│   └── workflows/
│       └── unittests.yml
├── .gitignore
├── requirements.txt
├── README.md
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   └── data_cleaner.py
│   └── utils.py
│   ├── eda_analyzer.py
│   └── sales_analyzer.py
├── notebooks/
│   ├── __init__.py
│   ├── README.md
│   └── eda_sales_analyzer.ipynb
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py
│   ├── test_eda_analyzer.py
│   └── test_sales_analyzer.py
└── scripts/
    ├── __init__.py
    ├── README.md
    └── run_analysis.py
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/brukGit/tenx-w4.git
   cd w4
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Place your data file in the `resources/Data/` directory.

2. Run the main analysis script:
   ```
   python scripts/run_analysis.py
   ```

3. Open and run the Jupyter notebook for detailed exploratory data analysis:
   ```
   jupyter notebook notebooks/eda_sales_analyzer.ipynb
   ```

## Data Description

The dataset includes information about stores, customers, state holidays, promotions, and competitors.
Key columns include:

- Promotional information - Promo, Promo2
- Competitors : CompetitionDistance , CompetitionOpenSince
- Customers and Sales: Customers, Sales, 
- Store details: Store, StoreType, Open, etc. 
- Holidays: StateHoliday, SchoolHoliday, etc.


## Analysis Components

1. **Data Loading and Preprocessing**: Handled by `src/data_loader.py`
2. **Data Cleaning**: Handled by `src/data_cleaning.py`
3. **Basic Data Overview**: Handled by `src/utils.py`
4. **Exploratory Data Analysis**: Implemented in `src/eda_analyzer.py`
5. **Sales Analysis**: Provided by `src/sales_analyzer.py`


## Running Tests

To run the unit tests:

```
python -m unittest discover tests
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.

---

For more information or support, please contact the project maintainers.