# House Price Predictor using Linear Regression

This project demonstrates how to build a basic house price prediction model using linear regression in Python with Google Colab. 
The model is built using a Kaggle dataset of Indian house prices, featuring various attributes such as the number of bedrooms, bathrooms, living area, lot area, and more.

## Overview

The goal of this project is to predict house prices based on the following features:
- **id:** Unique identifier for each record (can be dropped if not needed)
- **date:** The date when the house was listed (converted into useful features like year and month)
- **no. of bedrooms:** Total bedrooms in the house
- **no. of bathrooms:** Total bathrooms in the house
- **living area:** Size of the living area
- **lot area:** Total lot size
- **no. of floors:** Number of floors in the house
- **waterfront present:** Indicates if the property has waterfront access
- **no. of views:** Number of views or interest in the property
- **condition of house:** Qualitative rating of the house condition

The model uses these features to predict the target variable, **price**.

## Getting Started

Follow these instructions to set up and run the project in Google Colab.

### Prerequisites

- **Python 3.x**
- **Google Colab account** (if running in Colab)
- The following Python libraries:
  - `kagglehub`
  - `pandas`
  - `scikit-learn`
  - `matplotlib`
  - `seaborn`

### Installation

If you're running the project in Google Colab, install the necessary libraries using:

```python
!pip install kagglehub pandas scikit-learn matplotlib seaborn
```

### Usage
Download the Dataset
Use the kagglehub package to download the dataset from Kaggle. For example:

```python
import kagglehub
```

# Download the latest version of the dataset

``` python
path = kagglehub.dataset_download("mohamedafsal007/house-price-dataset-of-india")
print("Path to dataset files:", path)
```
### Load and Preprocess the Data

``` python
data_file = os.path.join(path, 'House Price India.csv')
df = pd.read_csv(data_file)
df.head()
```

### Drop unnecessary columns (like id)

```python
df.drop('id', axis=1, inplace=True)
```

### Convert the date column into datetime format and extract useful features such as year and month

```python
df['Date'] = pd.to_datetime(df['Date'])
df['year'] = df['Date'].dt.year
df['month'] = df['Date'].dt.month
df.drop('Date', axis=1, inplace=True)
```

### Handle any missing values or categorical variables (e.g., one-hot encoding for condition of house)

``` python
print(df.isnull().sum())
if X['condition of the house'].dtype == 'object':
    X = pd.get_dummies(X, columns=['condition of the house'], drop_first=True)
```


### Separate the features (X) from the target variable (y). Assume the target column is named price:
```python
X = df.drop(target_column, axis=1)
y = df[target_column]
```

### Split your data into training and testing sets using scikit-learn:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Create and fit the linear regression model:

```python
model = LinearRegression()
model.fit(X_train, y_train)
```

### Make predictions on the test set
```python 
y_pred = model.predict(X_test)
```

### Evaluate the model
```
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
```


### Visualize the Results
Visualize the performance of your model by plotting actual vs. predicted prices:

```python
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.show()
```

## Project Structure

├── README.md
├── HousePrice_Predict.ipynb        # Main Google Colab notebook containing the code
└── House Price India.csv           # Dataset file (downloaded via kagglehub)

## Contributing
Feel free to fork this repository and submit pull requests if you have any improvements or suggestions. For major changes, please open an issue first to discuss what you would like to change.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
