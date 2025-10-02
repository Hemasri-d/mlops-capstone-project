"""
Data exploration and processing module for retail analytics pipeline.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    """Main class for processing retail sales data."""
    
    def __init__(self, data_path: str):
        """
        Initialize DataProcessor with data path.
        
        Args:
            data_path: Path to the CSV file containing sales data
        """
        self.data_path = data_path
        self.df = None
        self.processed_df = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Load and perform initial data exploration.
        
        Returns:
            pandas.DataFrame: Loaded sales data
        """
        try:
            logger.info(f"Loading data from {self.data_path}")
            self.df = pd.read_csv(self.data_path)
            
            logger.info(f"Dataset shape: {self.df.shape}")
            logger.info(f"Columns: {self.df.columns.tolist()}")
            logger.info(f"Data types:\n{self.df.dtypes}")
            logger.info(f"Missing values:\n{self.df.isnull().sum()}")
            
            return self.df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def clean_data(self) -> pd.DataFrame:
        """
        Clean and preprocess the data.
        
        Returns:
            pandas.DataFrame: Cleaned data
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        logger.info("Starting data cleaning process...")
        
        # Create a copy for processing
        df_clean = self.df.copy()
        
        # Convert invoice_date to datetime
        df_clean['invoice_date'] = pd.to_datetime(df_clean['invoice_date'], format='%d/%m/%Y')
        
        # Calculate total amount per transaction
        df_clean['total_amount'] = df_clean['quantity'] * df_clean['price']
        
        # Add derived features
        df_clean['year'] = df_clean['invoice_date'].dt.year
        df_clean['month'] = df_clean['invoice_date'].dt.month
        df_clean['day_of_week'] = df_clean['invoice_date'].dt.day_name()
        df_clean['quarter'] = df_clean['invoice_date'].dt.quarter
        
        # Calculate age groups
        df_clean['age_group'] = pd.cut(
            df_clean['age'], 
            bins=[0, 25, 35, 45, 55, 65, 100], 
            labels=['18-25', '26-35', '36-45', '46-55', '56-65', '65+']
        )
        
        # Calculate transaction value categories
        df_clean['transaction_value'] = pd.cut(
            df_clean['total_amount'],
            bins=[0, 100, 500, 1000, 5000, float('inf')],
            labels=['Low', 'Medium', 'High', 'Very High', 'Premium']
        )
        
        # Remove any rows with missing critical data
        df_clean = df_clean.dropna(subset=['customer_id', 'invoice_date', 'total_amount'])
        
        logger.info(f"Data cleaning completed. Shape: {df_clean.shape}")
        self.processed_df = df_clean
        
        return df_clean
    
    def get_data_summary(self) -> Dict:
        """
        Get comprehensive data summary.
        
        Returns:
            Dict: Summary statistics and insights
        """
        if self.processed_df is None:
            raise ValueError("Data not processed. Call clean_data() first.")
        
        df = self.processed_df
        
        summary = {
            'total_transactions': len(df),
            'unique_customers': df['customer_id'].nunique(),
            'date_range': {
                'start': df['invoice_date'].min(),
                'end': df['invoice_date'].max()
            },
            'total_revenue': df['total_amount'].sum(),
            'average_transaction_value': df['total_amount'].mean(),
            'categories': df['category'].value_counts().to_dict(),
            'shopping_malls': df['shopping_mall'].value_counts().to_dict(),
            'payment_methods': df['payment_method'].value_counts().to_dict(),
            'gender_distribution': df['gender'].value_counts().to_dict(),
            'age_groups': df['age_group'].value_counts().to_dict()
        }
        
        return summary
    
    def calculate_customer_metrics(self) -> pd.DataFrame:
        """
        Calculate customer-level metrics for segmentation.
        
        Returns:
            pandas.DataFrame: Customer metrics
        """
        if self.processed_df is None:
            raise ValueError("Data not processed. Call clean_data() first.")
        
        logger.info("Calculating customer metrics...")
        
        df = self.processed_df
        
        # Customer-level aggregations
        customer_metrics = df.groupby('customer_id').agg({
            'total_amount': ['sum', 'mean', 'count'],
            'invoice_date': ['min', 'max'],
            'category': lambda x: x.nunique(),
            'shopping_mall': lambda x: x.nunique(),
            'payment_method': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown',
            'gender': 'first',
            'age': 'first'
        }).round(2)
        
        # Flatten column names
        customer_metrics.columns = [
            'total_spent', 'avg_transaction_value', 'transaction_count',
            'first_purchase_date', 'last_purchase_date', 'categories_purchased',
            'malls_visited', 'preferred_payment_method', 'gender', 'age'
        ]
        
        # Calculate additional metrics
        customer_metrics['days_since_last_purchase'] = (
            datetime.now() - customer_metrics['last_purchase_date']
        ).dt.days
        
        customer_metrics['customer_lifespan_days'] = (
            customer_metrics['last_purchase_date'] - customer_metrics['first_purchase_date']
        ).dt.days
        
        # Calculate recency score (days since last purchase)
        customer_metrics['recency_score'] = pd.cut(
            customer_metrics['days_since_last_purchase'],
            bins=[0, 30, 90, 180, 365, float('inf')],
            labels=[5, 4, 3, 2, 1]
        ).astype(int)
        
        # Calculate frequency score (transaction count)
        customer_metrics['frequency_score'] = pd.cut(
            customer_metrics['transaction_count'],
            bins=[0, 1, 3, 6, 12, float('inf')],
            labels=[1, 2, 3, 4, 5]
        ).astype(int)
        
        # Calculate monetary score (total spent)
        customer_metrics['monetary_score'] = pd.cut(
            customer_metrics['total_spent'],
            bins=[0, 500, 1000, 2500, 5000, float('inf')],
            labels=[1, 2, 3, 4, 5]
        ).astype(int)
        
        # Calculate RFM score
        customer_metrics['rfm_score'] = (
            customer_metrics['recency_score'].astype(str) +
            customer_metrics['frequency_score'].astype(str) +
            customer_metrics['monetary_score'].astype(str)
        ).astype(int)
        
        logger.info(f"Customer metrics calculated for {len(customer_metrics)} customers")
        
        return customer_metrics
    
    def segment_customers(self, customer_metrics: pd.DataFrame) -> pd.DataFrame:
        """
        Segment customers into high-value and low-value categories.
        
        Args:
            customer_metrics: DataFrame with customer metrics
            
        Returns:
            pandas.DataFrame: Customer metrics with segmentation
        """
        logger.info("Segmenting customers...")
        
        # Define segmentation criteria
        high_value_threshold = customer_metrics['total_spent'].quantile(0.8)
        frequent_threshold = customer_metrics['transaction_count'].quantile(0.7)
        recent_threshold = customer_metrics['days_since_last_purchase'].quantile(0.3)
        
        def assign_segment(row):
            if (row['total_spent'] >= high_value_threshold and 
                row['transaction_count'] >= frequent_threshold and
                row['days_since_last_purchase'] <= recent_threshold):
                return 'Champions'
            elif (row['total_spent'] >= high_value_threshold and 
                  row['transaction_count'] >= frequent_threshold):
                return 'Loyal Customers'
            elif (row['total_spent'] >= high_value_threshold):
                return 'High Value'
            elif (row['transaction_count'] >= frequent_threshold):
                return 'Frequent Buyers'
            elif (row['days_since_last_purchase'] <= recent_threshold):
                return 'Recent Customers'
            else:
                return 'At Risk'
        
        customer_metrics['customer_segment'] = customer_metrics.apply(assign_segment, axis=1)
        
        # Calculate segment statistics
        segment_stats = customer_metrics.groupby('customer_segment').agg({
            'total_spent': ['count', 'sum', 'mean'],
            'transaction_count': 'mean',
            'days_since_last_purchase': 'mean'
        }).round(2)
        
        logger.info("Customer segmentation completed")
        logger.info(f"Segment distribution:\n{customer_metrics['customer_segment'].value_counts()}")
        
        return customer_metrics
    
    def analyze_seasonal_trends(self) -> Dict:
        """
        Analyze seasonal sales trends.
        
        Returns:
            Dict: Seasonal analysis results
        """
        if self.processed_df is None:
            raise ValueError("Data not processed. Call clean_data() first.")
        
        logger.info("Analyzing seasonal trends...")
        
        df = self.processed_df
        
        # Monthly trends
        monthly_sales = df.groupby(['year', 'month']).agg({
            'total_amount': ['sum', 'count'],
            'customer_id': 'nunique'
        }).round(2)
        
        # Quarterly trends
        quarterly_sales = df.groupby(['year', 'quarter']).agg({
            'total_amount': ['sum', 'count'],
            'customer_id': 'nunique'
        }).round(2)
        
        # Day of week trends
        dow_sales = df.groupby('day_of_week').agg({
            'total_amount': ['sum', 'mean', 'count']
        }).round(2)
        
        # Category seasonal analysis
        category_monthly = df.groupby(['category', 'month'])['total_amount'].sum().unstack(fill_value=0)
        
        seasonal_analysis = {
            'monthly_trends': monthly_sales.to_dict(),
            'quarterly_trends': quarterly_sales.to_dict(),
            'day_of_week_trends': dow_sales.to_dict(),
            'category_seasonality': category_monthly.to_dict()
        }
        
        logger.info("Seasonal analysis completed")
        
        return seasonal_analysis
    
    def analyze_payment_methods(self) -> Dict:
        """
        Analyze payment method patterns and their impact.
        
        Returns:
            Dict: Payment method analysis
        """
        if self.processed_df is None:
            raise ValueError("Data not processed. Call clean_data() first.")
        
        logger.info("Analyzing payment methods...")
        
        df = self.processed_df
        
        # Payment method analysis
        payment_analysis = df.groupby('payment_method').agg({
            'total_amount': ['sum', 'mean', 'count'],
            'customer_id': 'nunique',
            'quantity': 'mean'
        }).round(2)
        
        # Payment method by category
        payment_category = df.groupby(['payment_method', 'category'])['total_amount'].sum().unstack(fill_value=0)
        
        # Payment method by mall
        payment_mall = df.groupby(['payment_method', 'shopping_mall'])['total_amount'].sum().unstack(fill_value=0)
        
        # Payment method by age group
        payment_age = df.groupby(['payment_method', 'age_group'])['total_amount'].sum().unstack(fill_value=0)
        
        analysis_results = {
            'payment_summary': payment_analysis.to_dict(),
            'payment_by_category': payment_category.to_dict(),
            'payment_by_mall': payment_mall.to_dict(),
            'payment_by_age': payment_age.to_dict()
        }
        
        logger.info("Payment method analysis completed")
        
        return analysis_results


def main():
    """Main function to demonstrate data processing."""
    # Initialize processor
    processor = DataProcessor('..\..\customer_shopping_data.csv')
    
    # Load and clean data
    df = processor.load_data()
    clean_df = processor.clean_data()
    
    # Get data summary
    summary = processor.get_data_summary()
    print("Data Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    # Calculate customer metrics
    customer_metrics = processor.calculate_customer_metrics()
    
    # Segment customers
    segmented_customers = processor.segment_customers(customer_metrics)
    
    # Analyze trends
    seasonal_trends = processor.analyze_seasonal_trends()
    payment_analysis = processor.analyze_payment_methods()
    
    print("\nCustomer Segmentation:")
    print(segmented_customers['customer_segment'].value_counts())
    
    return processor, clean_df, segmented_customers


if __name__ == "__main__":
    processor, data, customers = main()
