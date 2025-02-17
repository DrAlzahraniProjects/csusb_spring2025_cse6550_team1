# Importing the SQLite library for database operations
import sqlite3  

# Purpose: Initialize the DatabaseClient with a connection to the SQLite database.
# Input: db_path (str) - The path to the SQLite database file (default: "confusion_matrix.db").
# Output: A connection to the SQLite database.
# Processing: Establishes a connection to the SQLite database.

class DatabaseClient:
    """
    A client to manage confusion matrix performance metrics using SQLite.
    Provides methods to initialize, update, retrieve, and reset metrics.
    """
    def __init__(self, db_path="confusion_matrix.db"):
        """
        Initialize the DatabaseClient with a connection to the SQLite database.
        """
        self.connection = sqlite3.connect(db_path)

# Purpose: Create the performance_metrics table and initialize it with default values.
# Input: None
# Output: A table created in the database with initial values for performance metrics.
# Processing: Drops the existing table if it exists, creates a new table, and inserts default values for the performance metrics.

    def create_performance_metrics_table(self):
        with self.connection:
            self.connection.execute("DROP TABLE IF EXISTS performance_metrics;")
            self.connection.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                true_positive INTEGER,
                true_negative INTEGER,
                false_positive INTEGER,
                false_negative INTEGER,
                accuracy REAL,
                precision REAL,
                sensitivity REAL,
                specificity REAL,
                f1_score REAL,
                recall REAL
            )
        ''')
            self.connection.execute('''
                INSERT INTO performance_metrics (id, true_positive, true_negative, false_positive, false_negative, accuracy, precision, sensitivity, specificity, f1_score, recall)
                VALUES (1, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            ''')
# Purpose: Increment a specified performance metric by a given value.
# Input: metric (str) - The name of the metric to increment (e.g., 'true_positive').
# increment_value (int) - The value by which to increment the metric (default: 1).
#  Output: The performance metric is updated in the database.
#  Processing: Updates the specified metric in the database by incrementing it, ensuring it does not go below 0.

    def increment_performance_metric(self, metric, increment_value=1):
        with self.connection:
            self.connection.execute(f'''
                UPDATE performance_metrics
                SET {metric} = CASE
                    WHEN {metric} + {increment_value} < 0 THEN 0
                    ELSE {metric} + {increment_value}
                END
                WHERE id = 1
            ''')
# Purpose: Safely divides two numbers and returns a default value if division by zero occurs.
# Input: numerator (int/float) - The number to be divided.
#        denominator (int/float) - The number by which to divide.
#        default (optional) - A value to return if division by zero occurs (default: None).
# Output: The result of the division (rounded to 3 decimal places) or the default value if division by zero occurs.
#  Processing: Checks if the denominator is zero before performing the division; returns the default value if it is zero.


    def safe_division(self, numerator, denominator, default=None):
        if denominator == 0:
            return default
        return round(numerator / denominator, 3)
# Purpose: Update the performance metrics (accuracy, precision, sensitivity, specificity, f1_score, recall).
# Input: None
# Output: Updated performance metrics in the database.
# Processing: Calculates the metrics based on the current values of true_positive, true_negative, false_positive, and false_negative,
#             and then updates the performance metrics in the database.


    def update_performance_metrics(self):
        metrics = self.get_performance_metrics('true_positive, true_negative, false_positive, false_negative')
        accuracy = self.safe_division(
            metrics['true_positive'] + metrics['true_negative'],
            metrics['true_positive'] + metrics['true_negative'] + metrics['false_positive'] + metrics['false_negative']
        )
        precision = self.safe_division(metrics['true_positive'], metrics['true_positive'] + metrics['false_positive'])
        sensitivity = self.safe_division(metrics['true_positive'], metrics['true_positive'] + metrics['false_negative'])
        specificity = self.safe_division(metrics['true_negative'], metrics['true_negative'] + metrics['false_positive'])
        recall = self.safe_division(metrics['true_positive'], metrics['true_positive'] + metrics['false_negative'])
        
        if precision is None or sensitivity is None:
            f1_score = None
        else:
            f1_score = self.safe_division(2 * precision * sensitivity, precision + sensitivity)

        with self.connection:
            self.connection.execute('''
                UPDATE performance_metrics
                SET accuracy = ?, precision = ?, sensitivity = ?, specificity = ?, f1_score = ?, recall = ?
                WHERE id = 1
            ''', (accuracy, precision, sensitivity, specificity, f1_score, recall))
# Purpose: Retrieve the performance metrics from the database.
# Input: columns (str) - The columns to retrieve from the performance_metrics table (default: '*').
# Output: The performance metrics retrieved from the database.
# Processing: Retrieves the specified columns from the performance_metrics table and returns the result.

    def get_performance_metrics(self, columns='*'):
        cursor = self.connection.cursor()
        cursor.row_factory = sqlite3.Row
        cursor.execute(f'''
            SELECT {columns} FROM performance_metrics
            WHERE id = 1
        ''')
        result = cursor.fetchone()
        return result
# Purpose: Reset all performance metrics to initial zero state.
# Input: None
# Output: Metrics reset in the database.
# Processing: Updates the table with default values.
    def reset_performance_metrics(self):
        """
        Purpose: Reset all performance metrics to initial zero state.
        Input: None
        Output: Metrics reset in the database.
        Processing: Updates the table with default values.
        """
        with self.connection:
            self.connection.execute('''
                UPDATE performance_metrics
                SET true_positive = 0, true_negative = 0, false_positive = 0, false_negative = 0, accuracy = 0.0, precision = 0.0, sensitivity = 0.0, specificity = 0.0, f1_score = 0.0, recall = 0.0
                WHERE id = 1
            ''')
