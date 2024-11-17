import unittest
import pandas as pd
import numpy as np
from tools import download_dataset,Preprocessing, preprocess

class TestDownloadDataset(unittest.TestCase):
    def test_kidney_dataset(self):
        kidney_data = download_dataset('Kidney_dataset')
        self.assertIsInstance(kidney_data, pd.DataFrame, "Kidney dataset should return a DataFrame")
        self.assertIn('classification', kidney_data.columns, "'classification' column should be present in Kidney dataset")
        self.assertFalse(kidney_data.empty, "Kidney dataset should not be empty")
    
    def test_banknote_dataset(self):
        banknote_data = download_dataset('banknote_authentication_dataset')
        self.assertIsInstance(banknote_data, pd.DataFrame, "Banknote dataset should return a DataFrame")
        self.assertIn('classification', banknote_data.columns, "'classification' column should be present in Banknote dataset")
        self.assertFalse(banknote_data.empty, "Banknote dataset should not be empty")
    
    def test_invalid_dataset_name(self):
        with self.assertRaises(Exception) as context:
            download_dataset('invalid_dataset_name')
        self.assertEqual(str(context.exception), "not an available dataset", "Exception message should match expected output")

class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        """Set up sample datasets for testing."""
        self.data_with_missing = pd.DataFrame({
            'col1': ['*?', 'value1', 'value2', '\t?'],
            'col2': ['value3', '*?', 'value4', 'value5\t']
        })
        self.data_numeric = pd.DataFrame({
            'A': ['1', '2', '3', '2.9'],
            'B': [4.5, '5.6', 7.8, "WeAreTheBestGroupInMCE"]
        })
        self.data_with_nulls = pd.DataFrame({
            'A': [1, 2, np.nan, 4],
            'B': ['cat', 'dog', np.nan, 'dog']
        })
        self.data_categorical = pd.DataFrame({
            'A': ['yes', 'no', 'yes', 'no'],
            'B': ['present', 'absent', 'present', 'absent']
        })
        self.data_to_normalize = pd.DataFrame({
            'A': [10, 20, 30],
            'B': [1, 2, 3],
            'classification': ['cat', 'dog', 'cat']
        })

    def test_clean_data(self):
        """Test replacing missing values with NaN."""
        cleaned_data = Preprocessing.clean_data(self.data_with_missing)
        expected = pd.DataFrame({
            'col1': [np.nan, 'value1', 'value2', np.nan],
            'col2': ['value3', np.nan, 'value4', 'value5']
        })
        self.assertEqual(cleaned_data['col2'][0], 'value3')  
        self.assertEqual(cleaned_data['col2'][3], 'value5')

    def test_convert_to_numeric(self):
        """Test converting columns to numeric where possible."""
        converted_data = Preprocessing.convert_to_numeric(self.data_numeric)
        self.assertTrue(pd.api.types.is_numeric_dtype(converted_data['A']), "Column A should be converted to numeric")
        self.assertFalse(pd.api.types.is_numeric_dtype(converted_data['B']), "Column B should remain partially non-numeric")

    def test_handle_missing_values(self):
        """Test handling missing values by filling them appropriately."""
        handled_data = Preprocessing.handle_missing_values(self.data_with_nulls)
        self.assertFalse(handled_data.isnull().values.any(), "All missing values should be filled")
        self.assertAlmostEqual(handled_data['A'][2], 2.33, delta = 1e-2 ,msg = "Numeric missing values should be filled with the mean")
        self.assertEqual(handled_data['B'][2], 'dog', "Categorical missing values should be filled with the mode")

    def test_encode_categorical_features(self):
        """Test encoding categorical features with one-hot encoding."""
        encoded_data = Preprocessing.encode_categorical_features(self.data_categorical)
        self.assertTrue(pd.api.types.is_numeric_dtype(encoded_data['A']), "A should be categoricaly encoded")
        self.assertTrue(pd.api.types.is_numeric_dtype(encoded_data['B']), "B should be categoricaly encoded")

    def test_normalize_data(self):
        """Test normalizing numeric data to have mean 0 and std 1."""
        normalized_data = Preprocessing.normalize_data(self.data_to_normalize)
        self.assertAlmostEqual(normalized_data['A'].mean(), 0, delta=1e-1, msg = "Normalized data should have a mean of 0")
        self.assertAlmostEqual(normalized_data['A'].std(), 1, delta=1, msg = "Normalized data should have a standard deviation of 1")
        self.assertAlmostEqual(normalized_data['B'].mean(), 0, delta=1e-1, msg = "Normalized data should have a mean of 0")
        self.assertAlmostEqual(normalized_data['B'].std(), 1, delta=1, msg = "Normalized data should have a standard deviation of 1")
        self.assertTrue(pd.api.types.is_object_dtype(normalized_data['classification']), msg= "'classification' should remain Categorical")
        
class TestPreprocessFunction(unittest.TestCase):
    def setUp(self):
        """Set up test datasets."""
        self.data = pd.DataFrame({
            'A': [1, 2, '*?', 4],
            'classification': ['cat', 'dog', '\t?', 'dog'],
            'C' : ['1.2', 1, 3 , '24'],
        })

    def test_preprocess_pipeline(self):
        """Test the preprocess pipeline with multiple steps."""
        processed_data = preprocess(
            self.data,
            Preprocessing.clean_data,
            Preprocessing.convert_to_numeric,
            Preprocessing.handle_missing_values,
            Preprocessing.normalize_data,
            Preprocessing.encode_categorical_features
        )

        self.assertTrue(pd.api.types.is_numeric_dtype(processed_data['A']), "'A' should be numeric after conversion")
        self.assertTrue(pd.api.types.is_numeric_dtype(processed_data['classification']), "'classification' remains catetgory")
        self.assertTrue(processed_data.apply(lambda x: ((x == '\t?') | (x == '*?'))).sum().sum() == 0, msg = "dataframe still have missing values")
        self.assertTrue(processed_data.isnull().sum().sum() == 0, msg="data contains null values")
        self.assertAlmostEqual(processed_data['A'].mean(), 0, delta=1e-1, msg = "Normalized data should have a mean of 0")
        self.assertAlmostEqual(processed_data['A'].std(), 1, delta=1, msg = "Normalized data should have a standard deviation of 1")
        self.assertAlmostEqual(processed_data['C'].mean(), 0, delta=1e-1, msg = "Normalized data should have a mean of 0")
        self.assertAlmostEqual(processed_data['C'].std(), 1, delta=1, msg = "Normalized data should have a standard deviation of 1")

if __name__ == '__main__':
    unittest.main()