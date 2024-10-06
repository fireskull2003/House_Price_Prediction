# House_Price_Prediction
Use a dataset that includes information about housing prices and features like square footage, number of bedrooms, etc. to train a model that  predict the price of a new house

This project aims to predict house prices using historical data. The dataset includes features such as square footage, number of bedrooms, number of bathrooms, year built, and lot size. The project utilizes Linear Regression and Random Forest Regressor models to predict house prices and compares their performance.


Project Structure
+ **task 3/kc_house_data.csv:** The dataset containing house-related data for training and testing the models.
+ **house_price_prediction.py:** Main Python script that performs data analysis, model training, and evaluation.
+ **README.md:** This file providing an overview of the project.
## Getting Started
**Prerequisites**
Ensure you have the following installed:

* Python 3.x
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
Install the required Python libraries using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```
## Dataset
The dataset used in this project contains the following key features:

* **SquareFootage:** The size of the house in square feet.
* **Bedrooms:** Number of bedrooms.
* **Bathrooms:** Number of bathrooms.
* **YearBuilt:** The year the house was built.
* **LotSize:** The size of the lot in square feet.
* **Price:** The price of the house (target variable).
## Running the Project
To run the script:

1. Clone this repository.
2. Ensure the dataset is in the task 3/ directory.
3. Run the `house_price_prediction.py` script.
```bash
python house_price_prediction.py
```
### Key Steps in the Script
1. **Data Loading:** The dataset is loaded using Pandas, and an overview of the data is printed (head() and info()).
2. Exploratory Data Analysis (EDA):
* A heatmap is generated to visualize the correlation between the features and the house price.
* A histogram of house price distribution is plotted.
3. Data Preprocessing:
* Missing values are dropped (dropna()).
* The dataset is split into features (X) and target (y).
* The data is split into training and testing sets using train_test_split().
4. **Model Training:**
* A Linear Regression model is trained using the training set.
* A Random Forest Regressor is also trained on the same data for comparison.
5. **Model Evaluation:**
* Mean Squared Error (MSE) and R² score are used to evaluate model performance.
* Scatter plots are generated to compare actual vs. predicted house prices for the Random Forest model.


## Visualizations
+ **Correlation Heatmap:** Shows the correlation between features and house price.
+ **House Price Distribution:** A histogram showing the distribution of house prices.
+ **Actual vs Predicted Prices:** A scatter plot comparing actual and predicted house prices using the Random Forest model.
