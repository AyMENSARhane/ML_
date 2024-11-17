import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ucimlrepo import fetch_ucirepo
#model tools 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from category_encoders import OrdinalEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import recall_score, precision_score, accuracy_score

#Function to fetch the dataset
def download_dataset(dataset_name: str = 'Kidney_dataset'):
    """
    Downloads and returns a specified dataset with features and a target variable.
    
    Parameters:
        dataset_name (str): Name of the dataset. Options are:
            - 'Kidney_dataset': Chronic Kidney Disease dataset.
            - 'banknote_authentication_dataset': Banknote Authentication dataset.
    
    Returns:
        pandas.DataFrame: Dataset with features and target variable.
    
    Raises:
        Exception: If the dataset_name is not recognized.
    """
    if dataset_name == 'Kidney_dataset':
        chronic_kidney_disease = fetch_ucirepo(id=336)
        data_kidney = chronic_kidney_disease.data.features
        y = chronic_kidney_disease.data.targets
        data_kidney['classification'] = y
        return data_kidney

    elif dataset_name == 'banknote_authentication_dataset':
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
        data_banknote = pd.read_csv(url, header=None)
        data_banknote.columns = data_banknote.columns.astype('str')
        data_banknote['classification'] = data_banknote['4']
        data_banknote.drop(columns=['4'], inplace=True)
        return data_banknote

    else:
        raise Exception("not an available dataset")
    
def get_features_labels(data : pd.DataFrame):
    if 'classification' in data.columns:
        X = data.drop(columns = ['classification'])
        y = data['classification']
        return X,y
    else: 
        raise Exception("your data should have a column name 'classification' ")

class Preprocessing:
    #we don't use __init__ we just need to seperate our functions to make our code understoodable
    
    def clean_data(data: pd.DataFrame, missing_values=['*?', '\t?'], bad_values = ['\t']):
        """
        Replaces specified missing value indicators in the dataset with NaN. and removes '\t' in target

        Parameters:
            data (pd.DataFrame): The input dataset.
            missing_values (list): A list of strings representing missing values to replace with NaN.

        Returns:
            pd.DataFrame: The dataset with missing values replaced.
        """
        data = data.replace(missing_values, np.nan)
        for value in bad_values:
            data.loc[:, data.select_dtypes('object').columns] = (
                data.select_dtypes('object')
                .apply(lambda col: col.astype(str).str.replace(fr'{value}', '', regex=True).str.strip())
            )
        return data


    def convert_to_numeric(data):
        """
        Converts all columns in the dataset to numeric data types, where possible.

        Parameters:
            data (pd.DataFrame): The input dataset.

        Returns:
            pd.DataFrame: The dataset with numeric conversion applied.
        """
        for col in data.columns:
            try:
                data[col] = pd.to_numeric(data[col])
            except (ValueError, TypeError):
                pass
        return data

    def handle_missing_values(data):
        """
        Handles missing values by filling them with the mean for numeric columns 
        and the mode for categorical columns.

        Parameters:
            data (pd.DataFrame): The input dataset.

        Returns:
            pd.DataFrame: The dataset with missing values handled.
        """
        for col in data.columns:
            if data[col].dtype in [float, int]:
                data[col] = data[col].fillna(data[col].mean())
            else:
                data[col] = data[col].fillna(data[col].mode().iloc[0])
        return data

    def encode_categorical_features(data):
        """
        Encodes categorical features using one-hot encoding and optionally drops 
        the first column to avoid multicollinearity. Simplifies column names by 
        removing redundant suffixes (e.g., '_yes', '_no').

        Parameters:
            data (pd.DataFrame): The input dataset.
            drop_first (bool): Whether to drop the first column of one-hot encoded features.

        Returns:
            pd.DataFrame: The dataset with encoded categorical features.
        """
        data_encoded = OrdinalEncoder(return_df=True).fit_transform(data)
        return data_encoded

    def normalize_data(data: pd.DataFrame):
        """
        Normalizes the dataset to have a mean of 0 and a standard deviation of 1 
        using StandardScaler.

        Parameters:
            data (pd.DataFrame): The input dataset.

        Returns:
            pd.DataFrame: The normalized dataset as a DataFrame.
        """
        scaler = StandardScaler()
        num_cols = data.drop(columns=['classification']).select_dtypes('number').columns.to_list()
        scale_nums = scaler.fit_transform(data.drop(columns=['classification']).select_dtypes('number'))
        data_normalized = data.copy()
        data_normalized[num_cols] = scale_nums
        return pd.DataFrame(data_normalized, columns=data.columns)


# pipeline function
def preprocess(data: pd.DataFrame, *processors: Preprocessing):
    """
    Applies a sequence of preprocessing functions to a specified dataset.

    Parameters:
        data (pd.DataFrame): DataFrame to transform usinf predefined pipelines
        *processors (Preprocessing): A sequence of preprocessing methods 
            from the Preprocessing class to be applied in order.

    Returns:
        pandas.DataFrame: The preprocessed dataset.

    Example:
        processed_data = preprocess('Kidney_dataset', 
                                    Preprocessing.clean_data, 
                                    Preprocessing.convert_to_numeric,
                                    Preprocessing.handle_missing_values)
    """
    output = data
    for process in processors:
        output = process(output)
    return output

class MlTools:
    # tool of performing grid search with Logistic Regression
    def LR_grid_search_evaluation(
        X_train,
        X_test,
        y_train,
        y_test,
        dataset_name,
        param_grid,
        scoring):
        """
        Performs GridSearchCV with Logistic Regression and evaluates the model based on recall, precision, and accuracy.

        Parameters:
            X_train (pd.DataFrame): The training feature set.
            X_test (pd.DataFrame): The testing feature set.
            y_train (pd.Series): The training target labels.
            y_test (pd.Series): The testing target labels.
            dataset_name (str): Name of the dataset being processed.
            param_grid (dict): Dictionary containing parameters for GridSearchCV.
            scoring (str): The scoring metric to optimize for.

        Returns:
            best_model (LogisticRegression): The best model from grid search.
    """
        grid_search = GridSearchCV(
            LogisticRegression(),
            param_grid=param_grid,
            cv=5,
            scoring=scoring,
            refit='accuracy',  # Refit the best model based on accuracy
            verbose=1,
            return_train_score=True
        )

        # Perform grid search
        grid_search.fit(X_train, y_train)

        # Get the best model and its metrics
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)

        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"Results for {dataset_name}:")
        print("Best Model Parameters:", grid_search.best_params_)
        print(f"Recall: {recall:.2f}")
        print(f"Precision: {precision:.2f}")
        print(f"Accuracy: {accuracy:.2f}")

        # Plot accuracy over iterations
        results = grid_search.cv_results_
        results_frame = pd.DataFrame(results)
        results_frame.to_csv(f"grid-cv-results/LR_result-{dataset_name}.csv")
        mean_test_accuracy = results['mean_test_accuracy']

        plt.plot(range(len(mean_test_accuracy)), mean_test_accuracy, marker='o', label='Test Accuracy')
        plt.title(f"{dataset_name} - Grid Search Accuracy Over Iterations")
        plt.xlabel("Iteration")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        plt.show()
        return best_model
    
    
    def SVM_grid_search_evaluation(
        X_train,
        X_test,
        y_train,
        y_test,
        dataset_name,
        param_grid,
        scoring):
        """
        Performs GridSearchCV with Support Vector Machine (SVM) and evaluates the model based on recall, precision, and accuracy.

        Parameters:
            X_train (pd.DataFrame): The training feature set.
            X_test (pd.DataFrame): The testing feature set.
            y_train (pd.Series): The training target labels.
            y_test (pd.Series): The testing target labels.
            dataset_name (str): Name of the dataset being processed.
            param_grid (dict): Dictionary containing parameters for GridSearchCV.
            scoring (str): The scoring metric to optimize for.

        Returns:
            best_model (SVC): The best model from grid search.
        """
        
        # Initialize and perform GridSearchCV
        grid_search = GridSearchCV(
            SVC(),
            param_grid=param_grid,
            cv=5,
            scoring=scoring,
            refit='accuracy',  # Refit the best model based on accuracy
            verbose=1,
            return_train_score=True
        )

        grid_search.fit(X_train, y_train)

        # Get the best model and its predictions
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)

        # Calculate metrics
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)

        # Print results
        print(f"\n--- Results for {dataset_name} ---")
        print("Best Model Parameters:", grid_search.best_params_)
        print(f"Recall: {recall:.2f}")
        print(f"Precision: {precision:.2f}")
        print(f"Accuracy: {accuracy:.2f}")

        # Plot accuracy over iterations
        results = grid_search.cv_results_
        results_frame = pd.DataFrame(results)
        results_frame.to_csv(f"grid-cv-results/SVM_result-{dataset_name}.csv")
        mean_test_accuracy = results['mean_test_accuracy']

        plt.plot(range(len(mean_test_accuracy)), mean_test_accuracy, marker='o', label='Test Accuracy')
        plt.title(f"{dataset_name} - SVM Grid Search Accuracy Over Iterations")
        plt.xlabel("Iteration")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        plt.show()

        return best_model
    
    def DT_grid_search_evaluation(
        X_train,
        X_test,
        y_train,
        y_test,
        dataset_name,
        param_grid,
        scoring):
        
        """
        DT_grid_search_evaluation performs a grid search over hyperparameters for a Decision Tree Classifier using 
        cross-validation and evaluates the best model on the provided test data.

        Parameters:
            X_train (pd.DataFrame): Training features for the model.
            X_test (pd.DataFrame): Test features to evaluate the model performance.
            y_train (pd.Series): Training target variable.
            y_test (pd.Series): Test target variable.
            dataset_name (str): Name of the dataset used for the grid search (e.g., 'Kidney_dataset').
            param_grid (dict): A dictionary containing the hyperparameters to tune, such as 'max_depth', 'min_samples_split', etc.
            scoring (str): The scoring metric to evaluate the performance of the model (e.g., 'accuracy', 'precision', etc.).

        Returns:
            best_model (DecisionTreeClassifier): The best model obtained from the grid search, based on the specified scoring metric.

        Description:
            This function applies GridSearchCV to find the best Decision Tree model parameters, 
            evaluates its performance on test data, and prints metrics like recall, precision, and accuracy. 
            It also plots the mean test accuracy as a function of 'max_depth' to visualize the effect of different tree depths 
            on model performance. The results of the grid search are saved to a CSV file for further analysis.

        Example:
            best_model = DT_grid_search_evaluation(
                X_train, X_test, y_train, y_test, 'Kidney_dataset', param_grid, 'accuracy'
            )
        """

        grid_search = GridSearchCV(
            DecisionTreeClassifier(),
            param_grid=param_grid,
            cv=5,
            scoring=scoring,
            refit='accuracy',  # Refit the best model based on accuracy
            verbose=1,
            return_train_score=True
        )

        # Perform grid search
        grid_search.fit(X_train, y_train)

        # Get the best model and its metrics
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)

        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"Results for {dataset_name}:")
        print("Best Model Parameters:", grid_search.best_params_)
        print(f"Recall: {recall:.2f}")
        print(f"Precision: {precision:.2f}")
        print(f"Accuracy: {accuracy:.2f}")

        # Plot accuracy over iterations
        results = grid_search.cv_results_
        results_frame = pd.DataFrame(results)
        
        # Group by 'param_max_depth' and calculate mean accuracy for each depth
        grouped_results = results_frame.groupby('param_max_depth')['mean_test_accuracy'].mean().reset_index()

        # Sorting the results by max_depth for better visualization
        grouped_results = grouped_results.sort_values(by='param_max_depth')

        # Plotting the results
        plt.figure(figsize=(8, 6))
        plt.plot(grouped_results['param_max_depth'], grouped_results['mean_test_accuracy'], marker='o', linestyle='-', color='b')
        plt.title(f"Mean Test Accuracy by Max Depth - Decision Tree ({dataset_name})", fontsize=14)
        plt.xlabel("Max Depth", fontsize=12)
        plt.ylabel("Mean Test Accuracy", fontsize=12)
        plt.grid(True)
        plt.xticks(grouped_results['param_max_depth'])
        plt.show()

        # Save results to a CSV file
        results_frame.to_csv(f"grid-cv-results/DT_result-{dataset_name}.csv")
        
        return best_model
    
    #function to perform grid search with random forest classifier
    def RF_grid_search_evaluation(
            X_train,
            X_test,
            y_train,
            y_test,
            dataset_name,
            param_grid,
            scoring):
        """
        RF_grid_search_evaluation performs a grid search over hyperparameters for a Random Forest Classifier using 
        cross-validation and evaluates the best model on the provided test data.

        Parameters:
            X_train (pd.DataFrame): Training features for the model.
            X_test (pd.DataFrame): Test features to evaluate the model performance.
            y_train (pd.Series): Training target variable.
            y_test (pd.Series): Test target variable.
            dataset_name (str): Name of the dataset used for the grid search (e.g., 'Kidney_dataset').
            param_grid (dict): A dictionary containing the hyperparameters to tune, such as 'n_estimators', 'max_depth', etc.
            scoring (str): The scoring metric to evaluate the performance of the model (e.g., 'accuracy', 'precision', etc.).

        Returns:
            best_model (RandomForestClassifier): The best model obtained from the grid search, based on the specified scoring metric.

        Description:
            This function applies GridSearchCV to find the best Random Forest model parameters, 
            evaluates its performance on test data, and prints metrics like recall, precision, and accuracy. 
            It also plots the mean test accuracy as a function of 'n_estimators' to visualize the effect of different 
            numbers of trees in the forest on model performance. The results of the grid search are saved to a CSV file 
            for further analysis.
        """
        # Perform grid search
        grid_search = GridSearchCV(
            RandomForestClassifier(),
            param_grid=param_grid,
            cv=5,
            scoring=scoring,
            refit='accuracy',  # Refit the best model based on accuracy
            verbose=1,
            return_train_score=True
        )
        
        # Fit grid search
        grid_search.fit(X_train, y_train)

        # Get the best model and its metrics
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)

        # Calculate metrics
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"Results for {dataset_name}:")
        print("Best Model Parameters:", grid_search.best_params_)
        print(f"Recall: {recall:.2f}")
        print(f"Precision: {precision:.2f}")
        print(f"Accuracy: {accuracy:.2f}")

        # Get grid search results
        results = grid_search.cv_results_
        results_frame = pd.DataFrame(results)
        
        # Group by 'param_n_estimators' and calculate mean accuracy for each number of estimators
        grouped_results = results_frame.groupby('param_n_estimators')['mean_test_accuracy'].mean().reset_index()

        # Sorting the results for better visualization
        grouped_results = grouped_results.sort_values(by='param_n_estimators')

        # Plot mean test accuracy vs. number of estimators
        plt.figure(figsize=(8, 6))
        plt.plot(grouped_results['param_n_estimators'], grouped_results['mean_test_accuracy'], marker='o', linestyle='-', color='g')
        plt.title(f"Mean Test Accuracy by Number of Estimators - Random Forest ({dataset_name})", fontsize=14)
        plt.xlabel("Number of Estimators", fontsize=12)
        plt.ylabel("Mean Test Accuracy", fontsize=12)
        plt.grid(True)
        plt.xticks(grouped_results['param_n_estimators'])
        plt.show()

        # Save results to CSV
        results_frame.to_csv(f"grid-cv-results/RF_result-{dataset_name}.csv")

        return best_model
    
    # tool to perform grid search on Knn
    def KNN_grid_search_evaluation(
        X_train,
        X_test,
        y_train,
        y_test,
        dataset_name,
        param_grid,
        scoring):
        """
        KNN_grid_search_evaluation performs a grid search over hyperparameters for a K-Nearest Neighbors (KNN) classifier 
        using cross-validation and evaluates the best model on the provided test data.

        Parameters:
            X_train (pd.DataFrame): Training features for the model.
            X_test (pd.DataFrame): Test features to evaluate the model performance.
            y_train (pd.Series): Training target variable.
            y_test (pd.Series): Test target variable.
            dataset_name (str): Name of the dataset used for the grid search (e.g., 'Kidney_dataset').
            param_grid (dict): A dictionary containing the hyperparameters to tune, such as 'n_neighbors', 'weights', etc.
            scoring (str): The scoring metric to evaluate the performance of the model (e.g., 'accuracy', 'precision', etc.).

        Returns:
            best_model (KNeighborsClassifier): The best model obtained from the grid search, based on the specified scoring metric.

        Description:
            This function applies GridSearchCV to find the best KNN model parameters, 
            evaluates its performance on test data, and prints metrics like recall, precision, and accuracy. 
            It also plots the mean test accuracy as a function of the number of neighbors ('n_neighbors') 
            to visualize the effect of different numbers of neighbors on model performance. 
            The results of the grid search are saved to a CSV file for further analysis.
        """
        # Perform grid search
        grid_search = GridSearchCV(
            KNeighborsClassifier(),
            param_grid=param_grid,
            cv=5,
            scoring=scoring,
            refit='accuracy',  # Refit the best model based on accuracy
            verbose=1,
            return_train_score=True
        )
        
        # Fit grid search
        grid_search.fit(X_train, y_train)

        # Get the best model and its metrics
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)

        # Calculate metrics
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"Results for {dataset_name}:")
        print("Best Model Parameters:", grid_search.best_params_)
        print(f"Recall: {recall:.2f}")
        print(f"Precision: {precision:.2f}")
        print(f"Accuracy: {accuracy:.2f}")

        # Get grid search results
        results = grid_search.cv_results_
        results_frame = pd.DataFrame(results)
        
        # Group by 'param_n_neighbors' and calculate mean accuracy for each number of neighbors
        grouped_results = results_frame.groupby('param_n_neighbors')['mean_test_accuracy'].mean().reset_index()

        # Sorting the results for better visualization
        grouped_results = grouped_results.sort_values(by='param_n_neighbors')

        # Plot mean test accuracy vs. number of neighbors
        plt.figure(figsize=(8, 6))
        plt.plot(grouped_results['param_n_neighbors'], grouped_results['mean_test_accuracy'], marker='o', linestyle='-', color='r')
        plt.title(f"Mean Test Accuracy by Number of Neighbors - KNN ({dataset_name})", fontsize=14)
        plt.xlabel("Number of Neighbors", fontsize=12)
        plt.ylabel("Mean Test Accuracy", fontsize=12)
        plt.grid(True)
        plt.xticks(grouped_results['param_n_neighbors'])
        plt.show()

        # Save results to CSV
        results_frame.to_csv(f"grid-cv-results/KNN_result-{dataset_name}.csv")

        return best_model