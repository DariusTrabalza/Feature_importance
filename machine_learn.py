import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns

def main():
    '''
    Pre-processes data and trains random forest tree model, then visualises importance of features.
    '''

    df = import_and_clean()
    X_train, X_test, y_train, y_test, X, y = split_test_train(df)
    X_train_scaled,X_test_scaled = scale_features(X_train,X_test)
    clf = train_model(X_train_scaled, y_train)
    y_pred = predictions(X_test_scaled, clf)
    feature_importance_df,conf_matrix = evaluations(y_test, y_pred, clf, X) 
    visualisations(feature_importance_df,conf_matrix)
    print(df.head())
    
def import_and_clean():
    '''
    Imports the file named "clean_data.csv" as a pandas DataFrame. Returning print messages for any errors that may occur at this stage.
    Also checks for null data and prints a message if identified. Removes un-needed columns and adds a new column that will 
    be the target column called "classification" based on column "net_change".

    Returns:
        df: The cleaned up dataframe.

    Raises:
        FileNotFoundError: If 'clean_data.csv' is not found in the directory.
        Exception: For other unexpected errors during file import.
    '''

    try:
        df = pd.read_csv("clean_data.csv")
    except FileNotFoundError:
        print("Error: The file was not found")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    #check for missing vals in data set
    if df.isnull().values.any():
        print("Null Values found in data set")

    #remove date column
    df = df.drop(columns = ["timestamp"])

    #add binary column for classification 1 being yes and 0 being no
    df["classification"] = np.where(df["net_change"] > 500, 1, 0)

    return df

def split_test_train(df):
    '''
    Splits the data into X and y as well as further splitting the data into train and test with a test size of 30%

    param: df : The cleaned dataframe

    Returns:
        X_train : The first 70% of X data as a matrix for training
        X_test : The last 30% of X data as a matrix for testing
        y_train : The first 70% of y data as a series for training
        y_test : The last 30% of y data as a series for testing
        X: X without unnecessary columns as a matrix
        y: The target column only as a series
    '''

    #split data in X and y
    X = df.drop(columns=["outcome20","net_change","classification","#"])
    y = df["classification"]

    #split data into training and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
    return X_train, X_test, y_train, y_test, X,y

def scale_features(X_train, X_test):
    '''
    Scales the features as the different feature have widely vary ranges.

    param : X_train : The first 70% of X data as a matrix
    param : X_test : The last 30% of X data as a matrix

    Returns:
        X_train_scaled : scaled version of X_train
        X_test_scaled :  scaled version of X_test
    '''
    scaler = StandardScaler()

    #fit the scaler to feature data
    X_train_scaled = scaler.fit_transform(X_train)
    #apply fit
    X_test_scaled = scaler.transform(X_test)

    return(X_train_scaled,X_test_scaled)


def train_model(X_train_scaled, y_train):
    '''
    Initialises Random forest Classifier
    Trains the model on data from scaled X_train using y_train as target

    param: X_train_scaled : X_train scaled because of widely different feature ranges as a matrix
    param: y_train : The last 30% of the y data as a series

    Returns: clf : Random Forest classifier that has been trained on X_train_scaled and y_train
    '''

    clf = RandomForestClassifier(n_estimators=500, random_state=42)
    clf.fit(X_train_scaled, y_train)

    return clf


def predictions(X_test_scaled, clf):
    '''
    Takes X_scaled_test and makes predictions of y using trained model

    param: X_test_scaled : X_train scaled because of widely different feature ranges as a matrix
    param: clf : Random Forest classifier that has been trained on X_train_scaled and y_train

    Returns:
        y_pred : Predictions of y generated from X_test_scaled through the trained model
    '''

    y_pred = clf.predict(X_test_scaled)

    return y_pred


def evaluations(y_test, y_pred, clf, X):
    '''
    Takes the predictions made of y and compares them to the true values of y_test and then produces evaluation metrics.
    Precision : The amount of predicted positives that were actually positive
    Recall : Of all actual positives what amount were predicted positive
    F1 : A balance between the Precision and the Recall
    
    param: y_test : The actual results of y from X_test
    param: y_pred : Predictions of y generated from X_test_scaled through the trained model
    param: clf : The trained random forest classifier
    param: X : dataframe without unnecasary columns

    Returns:
        feature_importance_df : A dataframe of showing how much each feature contributed to the probability
        conf_matrix : A confusion matrix of the results
    '''

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')

    # Generate and print the confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    

    #check feature importance
    feature_importances = clf.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    #printing feature importance
    importances = clf.feature_importances_
    feature_names = X.columns 
    feature_importances = zip(feature_names, importances)
    sorted_feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
    for feature, importance in sorted_feature_importances:
        print(f"{feature}: {importance}")

    return feature_importance_df,conf_matrix


def visualisations(feature_importance_df,conf_matrix):
    '''
    Creates a bar plot to visualise the order of importance of the features in the predictions the trained model made

    param: feature_importance_df : A dataframe of showing how much each feature contributed to the probability
    '''

    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
    plt.title('Feature Importances in RandomForest Classifier')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.show()


    sns.heatmap(conf_matrix, annot = True, cmap = "Blues", fmt ="g")
    plt.xlabel("Predicted Labels")
    plt.ylabel("Actual Labels")
    plt.title("Results Heatmap")
    plt.show()



if __name__ == "__main__":
    main()