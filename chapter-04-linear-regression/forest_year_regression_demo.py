# Chapter 04 - Linear Regression
# Predicting Global Forest Area Trend Using Historical Data

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ----------------------------------------
# Step 1: Load Historical Dataset
# ----------------------------------------
data = pd.read_csv("forest_area_data.csv")

print("Loaded Historical Forest Dataset:\n")
print(data)
print("\n----------------------------------------\n")

# ----------------------------------------
# Step 2: Prepare Input and Output Variables
# ----------------------------------------
X = data[['Year']]
y = data['ForestArea']

# ----------------------------------------
# Step 3: Train Linear Regression Model
# ----------------------------------------
model = LinearRegression()
model.fit(X, y)

print("Linear Regression Model Trained Successfully.")
print("\nSlope of Best Fit Line:", model.coef_[0])
print("Intercept:", model.intercept_)
print("\n----------------------------------------\n")

# ----------------------------------------
# Step 4: Predict Future Forest Area
# ----------------------------------------
future_years = pd.DataFrame({'Year': [2030, 2040, 2050]})
predictions = model.predict(future_years)

print("Future Predictions:\n")
for year, area in zip(future_years['Year'], predictions):
    print(f"Predicted Forest Area in {year}: {area:.2f} million hectares")

print("\n----------------------------------------\n")

# ----------------------------------------
# Step 5: Create Regression Lines
# ----------------------------------------
historical_line_years = pd.DataFrame({'Year': [1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020]})
historical_line_predictions = model.predict(historical_line_years)

future_line_years = pd.DataFrame({'Year': [2020, 2030, 2040, 2050]})
future_line_predictions = model.predict(future_line_years)

# ----------------------------------------
# Step 6: Plot Graph
# ----------------------------------------
plt.figure(figsize=(10, 6))

# Historical data points
plt.scatter(X['Year'], y, color='green', s=70, label="Historical Forest Data")

# Solid best fit line through historical trend
plt.plot(historical_line_years['Year'], historical_line_predictions,
         color='darkorange', linewidth=2.5, label="Linear Regression Best Fit Line")

# Dotted future prediction extension
plt.plot(future_line_years['Year'], future_line_predictions,
         color='orangered', linestyle='--', linewidth=2)

# Future prediction points
plt.scatter(future_years['Year'], predictions,
            color='orangered', marker='D', s=90, label="Future Predictions")
            
# Labels and Title
plt.xlabel("Year")
plt.ylabel("Global Forest Area (Million Hectares)")
plt.title("Linear Regression Prediction of Global Forest Area Trend")
plt.legend()
plt.grid(True)

# Save graph image
plt.savefig("output.png", dpi=300, bbox_inches='tight')

plt.show()
