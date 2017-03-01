# wernerchao 0.11509
import numpy as np
import matplotlib.pyplot as plt
import graphlab
from graphlab.toolkits.feature_engineering import OneHotEncoder, NumericImputer
from sklearn.preprocessing import normalize


train = graphlab.SFrame("./train.csv")
test = graphlab.SFrame("./test.csv")

target = "SalePrice"
categorical_features = [
    "MSSubClass",
    "MSZoning",
    # "Street",
    # "Alley",
    # "LotShape",
    # "LandContour",
    # "Utilities",
    # "LotConfig",
    # "LandSlope",
    "Neighborhood",
    # "Condition1", # Proximity to various conditions
    # "Condition2", #
    "BldgType",
    "HouseStyle",

    # Dates
    "YearBuilt",
    "YearRemodAdd",
    "YrSold", #
    "MoSold",

    "RoofStyle",
    "RoofMatl",
    "Exterior1st",
    # "Exterior2nd",
    # "MasVnrType",

    # Condition
    "OverallCond",
    # "GarageCond",
    # "ExterCond",
    # "BsmtCond",

    # Quality
    "OverallQual",
    # "ExterQual",
    # "BsmtQual",
    # "KitchenQual",
    # "GarageQual",
    # "Fence", # Fence quality
    # "PoolQC", # Pool quality
    # "HeatingQC",
    # "FireplaceQu", # Fireplace quality

    # "Foundation",

    # Rooms
    # "BsmtFullBath", # combine
    # "BsmtHalfBath", # 1/2
    # "FullBath", # combine
    # "HalfBath", # 1/2
    "BedroomAbvGr",
    "KitchenAbvGr",

    # Other features
    "Fireplaces",
    # "GarageType",
    # "GarageYrBlt",
    # "GarageFinish",
    "GarageCars",

    "PavedDrive",

    "SaleCondition",

    # Don't really care
    # "BsmtExposure",
    # "BsmtFinType1",
    # "BsmtFinType2",

    # Utilities
    # "Heating",
    # "CentralAir",
    # "Electrical",

    # "TotRmsAbvGrd",
    # "Functional",

    # "SaleType",
    # "MiscFeature",
]

numerical_features = [
    "LotArea", # Lot size in square feet

    # "GarageArea", # Size of garage in square feet

    # just use TotalBsmtSF
    # "BsmtFinSF1", # Type 1 finished square feet
    # "BsmtFinSF2", # Type 2 finished square feet
    # "BsmtUnfSF", # Unfinished square feet of basement area
    "TotalBsmtSF", # Total square feet of basement area

    # ignore
    # "LowQualFinSF", # Low quality finished square feet (all floors)

    # just use GrLiveArea
    # "1stFlrSF", # First Floor square feet
    # "2ndFlrSF", # Second floor square feet
    "GrLivArea", # Above grade (ground) living area square feet

    # combine
    # "WoodDeckSF", # Wood deck area in square feet
    # "OpenPorchSF", # Open porch area in square feet
    # "EnclosedPorch", # Enclosed porch area in square feet
    # "3SsnPorch", # Three season porch area in square feet
    # "ScreenPorch", # Screen porch area in square feet

    # ignore
    # "MasVnrArea", # Masonry veneer area in square feet
    # "PoolArea", # Pool area in square feet
    # "LotFrontage", # Linear feet of street connected to property
    # "MiscVal" # $Value of miscellaneous feature
]

new_categorical_features = categorical_features + ["Bathrooms", "Remodeled", "RemodeledRecently"]
new_numerical_features = numerical_features + ["TotalPorchSF"]
all_features = new_categorical_features + new_numerical_features

def engineer_feature(data):
    # cache target values
    target_values = data[target] if target in data.column_names() else []

    # fix missing values
    print ">> Fix missing values"
    num_imputer = graphlab.feature_engineering.create(data, NumericImputer(features=numerical_features))
    data = num_imputer.transform(data)

    # engineer features
    data["TotalPorchSF"] = combine_porch(data)
    data["Bathrooms"] = combine_bathrooms(data)
    data["Remodeled"] = if_remodeled(data)
    data["RemodeledRecently"] = if_remodeled_recently(data)

    # normalize numerical features
    # for feature in new_numerical_features:
    #     data[feature] = normalize_feature(data, feature)

    # consolidate features/data
    print ">> Consolidate features/data"
    data = data[all_features]

    # one hot encode categorical features
    print ">> One hot encode categorical features"
    one_hot_encoder = graphlab.feature_engineering.create(data, OneHotEncoder(features=new_categorical_features))
    data = one_hot_encoder.transform(data)

    # restore cached target values if exists
    if len(target_values) > 0:
        data[target] = target_values

    return (data, one_hot_encoder, num_imputer)

def normalize_feature(data, feature):
    print ">> Normalize feature:", feature
    return normalize(np.array(data[feature]).reshape(-1,1), norm='l2', axis=0)

def square_feature(data, feature):
    print ">> Square feature:", feature
    return data[feature]**2

def combine_porch(data):
    print ">> Combining porch and deck SF"
    return data.apply(lambda x: x["WoodDeckSF"] + x["OpenPorchSF"] + x["EnclosedPorch"] + x["3SsnPorch"] + x["ScreenPorch"])

def combine_bathrooms(data):
    print ">> Combining # bathrooms"
    data = data.fillna("BsmtFullBath", 0.)
    data = data.fillna("BsmtHalfBath", 0.)
    data = data.fillna("FullBath", 0.)
    data = data.fillna("HalfBath", 0.)
    return data.apply(lambda x: "%.2f"%(x["BsmtFullBath"] + 0.5*x["BsmtHalfBath"] + x["FullBath"] + 0.5*x["HalfBath"]))

def if_remodeled(data):
    print ">> Check if the house was ever remodeled"
    return data.apply(lambda x: x["YearRemodAdd"] > x["YearBuilt"])

def if_remodeled_recently(data):
    print ">> Check if the house was recently remodeled"
    return data.apply(lambda x: abs(x["YearRemodAdd"] - x["YrSold"]) < 5)


# feature engineering
print "\n> Engineer feature for training data"
train, train_encoder, train_imputer = engineer_feature(train)

print "\n> Engineer feature for testing data"
test_id = test["Id"]
test, test_encoder, test_imputer = engineer_feature(test)

# split training/validation data
print "\n> Split train/validation 80/20"
train, validation = train.random_split(0.8, seed=1)

# train model
print "\n> Train!!"
train_features = new_numerical_features + ["encoded_features"]

l1_penalty_best = 75917.4067 #previous best
exp_values = np.logspace(-5, 8, num=100)
models = []
errors = []
coefficients = np.zeros((len(exp_values), 342))
i = 0
for value in exp_values:
    model = graphlab.linear_regression.create(train,
                                            target=target,
                                            features=train_features,
                                            validation_set=None,
                                            verbose=False,
                                            l1_penalty=value
                                            )
    models.append(model)
    errors.append(model.evaluate(validation)["rmse"])
    coefficients[i] = model.coefficients["value"]
    print "exp:", value, model.evaluate(validation)
    i += 1

# plot coefficients
# print "\n> Plot coefficients"
# print coefficients.shape
# for i in range(342):
#     coeff = coefficients[:,i]
#     plt.plot(exp_values, coeff, '.-')
# plt.yscale("log")
# plt.xscale("log")
# plt.show()


# select the model with the lowest error
print "\n> Select best model with lowest error"
min_index = np.argmin(errors)
model = models[min_index]
print "Validation error", model.evaluate(validation)
print model
print "Coefficients"
final_coefficients = model.coefficients
print "Number of discarded coefficients:", sum(final_coefficients["value"] == 0)
print "Number of coefficients kept:", sum(final_coefficients["value"] != 0)
print final_coefficients.sort("value", ascending=False)[:5]
print final_coefficients.sort("value")[:5]

# graph rmse and prediction
# print "\n> Plot prediction results"
# x = model.predict(validation)
# y = validation["SalePrice"]
# plt.scatter(x, y)
# plt.plot(y, y, "-r")
# plt.show()

# predict and save to file
print "\n> Output test result"
predictions = model.predict(test)
output = graphlab.SFrame({
    "Id": test_id,
    "SalePrice": predictions
})
output.export_csv("./submissions/main.csv")

print "\n\n>>>>> DONE <<<<<<"
# Class                          : BoostedTreesRegression

# Schema
# ------
# Number of examples             : 1197
# Number of feature columns      : 20
# Number of unpacked features    : 989

# Settings
# --------
# Number of trees                : 200
# Max tree depth                 : 150
# Training time (sec)            : 6.3423
# Training rmse                  : 347.8546
# Validation rmse                : None
# Training max_error             : 4176.375
# Validation max_error           : None




# Class                          : LinearRegression

# Schema
# ------
# Number of coefficients         : 1039
# Number of examples             : 1155
# Number of feature columns      : 20
# Number of unpacked features    : 1038

# Hyperparameters
# ---------------
# L1 penalty                     : 75917.4067
# L2 penalty                     : 0.0

# Training Summary
# ----------------
# Solver                         : fista
# Solver iterations              : 10
# Solver status                  : TERMINATED: Iteration limit reached.
# Training time (sec)            : 0.3568

# Settings
# --------
# Residual sum of squares        : 5.98496669914e+11
# Training RMSE                  : 22763.544

# Highest Positive Coefficients
# -----------------------------
# encoded_features[610]          : 87471.4177
# encoded_features[623]          : 65396.5263
# encoded_features[691]          : 56317.4832
# encoded_features[598]          : 54430.3674
# encoded_features[704]          : 48235.2499

# Lowest Negative Coefficients
# ----------------------------
# encoded_features[298]          : -41479.2696
# encoded_features[725]          : -41479.2696
# encoded_features[977]          : -11669.6794
# encoded_features[710]          : -9546.5654
# encoded_features[748]          : -8609.8142