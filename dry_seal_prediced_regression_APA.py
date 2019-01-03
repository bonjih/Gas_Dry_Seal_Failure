import pandas as pd
from datetime import timedelta
import math
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pickle_handler

# import data with time as an index
df = pd.read_csv('./csv_files/apa_with_date_time.csv', index_col='Date_and_Time')

#######################################
# utility methods for script
def which_cat(range_points, x):
    """ map a value x -> to the index (class) of its iqr """
    if x <= range_points[0]:
        return 0
    elif range_points[0] < x <= range_points[1]:
        return 1
    elif range_points[1] < x <= range_points[2]:
        return 2
    else:
        return 3


def get_classifier_method():
        """ create the classifier and save it to file """
        classifier = svm.SVR(kernel='linear')
        classifier.fit(X_train, y_train)
        pickle_handler.pickle.dump(classifier, open(pickle_path, 'wb'))
        return classifier


# features  APA say contribute to dry seal failure
############################################################

df = df[['Unit_1_HPC_Discharge_Seal_Primary_Vent_Discharge_Pressure_Value',
         'Unit_1_HPC_Vibration_Aft_X_Gap_Voltage_Value',
         'Unit_1_HPC_Vibration_Aft_X-Axis_Value',
         'Unit_1_HPC_Vibration_Forward_Y_Gap_Voltage_Value',
         'Unit_1_HPC_Vibration_Aft_Y-Axis_Value',
         'Unit_1_HPC_Vibration_Axial_Value',
         'Unit_1_HPC_Vibration_Forward_X_Gap_Voltage_Value',
         'Unit_1_HPC_Vibration_Forward_X-Axis_Value',
         'Unit_1_HPC_Vibration_Forward_Y_Gap_Voltage_Value',
         'Unit_1_HPC_Vibration_Forward_Y-Axis_Value']]

###########################################################################
# Prepare data into standardised X & y formats

label_Y = 'Unit_1_HPC_Vibration_Forward_X-Axis_Value'
forecast_col = label_Y  # to predict
forecast_out = int(math.ceil(0.001 * len(df)))  # predict some hours .. 0.001% of the year or 21 seconds?? .
df['label'] = df[forecast_col].shift(-forecast_out)
# label_dry_seal = []
X = df.drop(['label', label_Y], 1)  # return new Data frame to X
X = preprocessing.scale(X)  # normalisation
X_lately = X[-forecast_out:]  # last 10%
X = X[:-forecast_out]

df.dropna(inplace=True)
y = df['label']

# prepare, train/test split (20% train)
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)


# Get the classifier
pickle_path = 'pickles/lrpickle.pickle'
pickle_path_2 = 'pickles/lrpickle2.pickle'
clf = pickle_handler.get_clf_unpickled(get_classifier_method, pickle_path)
# Present accuracy information
mse = mean_squared_error(y_test, clf.predict(X_test))
print("MSE: %.4f" % mse)

accuracy = clf.score(X_test, y_test)  # sqr error
forecast_set = clf.predict(X_lately)
print('Accuracy: ' "%.2f%%" % (accuracy * 100))

# Create a forecast plot
df['Forecast'] = clf.predict(X)

last_date = df.iloc[-1].name
one_day = timedelta(days=1)
next_date = last_date

for i in forecast_set:
    df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]
    next_date = last_date

fig, ax = plt.subplots(1, 1)
df[label_Y].plot(color='b', ax=ax)
df['Forecast'].plot(color='r', ax=ax)
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Value')
plt.show()

# save data with predictions
df.to_csv('csv_files/data_with_predictions.csv')

# prepare data to be categorised
df = df.replace('NaN', 0.0)
# get a flat list of likeness array([...]) that contains iqr ranges
iqr_list = df['label'].quantile(q=[0.25, 0.5, 0.75]).as_matrix().flatten()
# list comprehension to map x to its iqr class
categories = [which_cat(iqr_list, x) for x in df['label']]

df['categories'] = categories
df.to_csv('csv_files/data_with_categories.csv', index=False, columns=(df.columns.difference(['label', 'Forecast'])))

