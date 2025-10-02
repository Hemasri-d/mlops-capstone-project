"""
Data exploration and processing module for retail analytics pipeline.
"""
import pandas as pd
import numpy as np
from ..models.segmentation import train_kmeans_segmentation as _train_kmeans, prepare_segmentation_features as _prep_features
from ..models.forecast import naive_monthly_forecast as _naive_forecast
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

    def load_data_from_directory(self, directory_path: str) -> pd.DataFrame:
        """
        Load and concatenate multiple CSVs from a directory into a single DataFrame.

        Args:
            directory_path: Path to directory containing daily sales CSV files

        Returns:
            pandas.DataFrame: Loaded concatenated sales data
        """
        try:
            logger.info(f"Loading all CSV files from directory: {directory_path}")
            csv_files = pd.Series(list(pd.io.common.os.listdir(directory_path)))
            csv_files = csv_files[csv_files.str.lower().str.endswith('.csv')]
            if len(csv_files) == 0:
                raise FileNotFoundError("No CSV files found in directory")

            frames = []
            for f in csv_files:
                full_path = pd.io.common.os.path.join(directory_path, str(f))
                logger.info(f"Reading {full_path}")
                frames.append(pd.read_csv(full_path))

            self.df = pd.concat(frames, ignore_index=True)
            logger.info(f"Concatenated shape: {self.df.shape}")
            return self.df
        except Exception as e:
            logger.error(f"Error loading directory data: {e}")
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
        
        # Calculate total amount per transaction (gross before discounts)
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

    def calculate_profitability(
        self,
        discount_rate_by_category: Optional[Dict[str, float]] = None,
        cost_rate_by_category: Optional[Dict[str, float]] = None,
        per_row_discount_rate_column: Optional[str] = None,
        per_row_unit_cost_column: Optional[str] = None,
    ) -> Dict:
        """
        Calculate profitability after discounts and estimated costs.

        Args:
            discount_rate_by_category: Optional mapping of category to discount rate (0-1)
            cost_rate_by_category: Optional mapping of category to cost rate as a fraction of price (0-1)
            per_row_discount_rate_column: Optional column name containing per-row discount rate (0-1)
            per_row_unit_cost_column: Optional column name containing per-row unit cost (absolute currency)

        Returns:
            Dict: profitability KPIs and breakdowns
        """
        if self.processed_df is None:
            raise ValueError("Data not processed. Call clean_data() first.")

        df = self.processed_df.copy()

        # Determine discount rate per row
        if per_row_discount_rate_column and per_row_discount_rate_column in df.columns:
            df['discount_rate'] = df[per_row_discount_rate_column].clip(lower=0, upper=1).fillna(0.0)
        else:
            discount_rate_by_category = discount_rate_by_category or {}
            df['discount_rate'] = df['category'].map(discount_rate_by_category).fillna(0.0)

        # Determine unit cost per row
        if per_row_unit_cost_column and per_row_unit_cost_column in df.columns:
            df['unit_cost'] = df[per_row_unit_cost_column].fillna(0.0)
        else:
            cost_rate_by_category = cost_rate_by_category or {}
            # Default cost rate if not provided
            default_cost_rate = 0.6
            df['cost_rate'] = df['category'].map(cost_rate_by_category).fillna(default_cost_rate)
            df['unit_cost'] = (df['price'] * df['cost_rate']).round(2)

        # Revenue after discounts
        df['net_price'] = (df['price'] * (1 - df['discount_rate'])).round(2)
        df['net_revenue'] = (df['net_price'] * df['quantity']).round(2)

        # Cost of goods sold
        df['cogs'] = (df['unit_cost'] * df['quantity']).round(2)

        # Profit and margin
        df['profit'] = (df['net_revenue'] - df['cogs']).round(2)
        df['margin_pct'] = np.where(df['net_revenue'] > 0, (df['profit'] / df['net_revenue']).round(4), 0.0)

        # Build a JSON-serializable by_month with string keys like "YYYY-MM"
        monthly = (
            df.groupby(['year', 'month'])[['net_revenue', 'cogs', 'profit']]
            .sum()
            .round(2)
            .reset_index()
        )
        by_month_serializable = {}
        for _, row in monthly.iterrows():
            key = f"{int(row['year'])}-{int(row['month']):02d}"
            by_month_serializable[key] = {
                'net_revenue': float(row['net_revenue']),
                'cogs': float(row['cogs']),
                'profit': float(row['profit']),
            }

        # Helper to serialize groupby with multiple numeric columns into {index: {col: float}}
        def serialize_groupby_sum(index_col: str) -> Dict[str, Dict[str, float]]:
            agg = (
                df.groupby(index_col)[['net_revenue', 'cogs', 'profit']]
                .sum()
                .round(2)
            )
            out: Dict[str, Dict[str, float]] = {}
            for idx, row in agg.iterrows():
                out[str(idx)] = {
                    'net_revenue': float(row['net_revenue']),
                    'cogs': float(row['cogs']),
                    'profit': float(row['profit']),
                }
            return out

        profitability_summary = {
            'total_gross_revenue': float(df['total_amount'].sum().round(2)),
            'total_net_revenue': float(df['net_revenue'].sum().round(2)),
            'total_cogs': float(df['cogs'].sum().round(2)),
            'total_profit': float(df['profit'].sum().round(2)),
            'avg_margin_pct': float(df['margin_pct'].replace([np.inf, -np.inf], np.nan).fillna(0).mean().round(4)),
            'by_category': serialize_groupby_sum('category'),
            'by_mall': serialize_groupby_sum('shopping_mall'),
            'by_payment_method': serialize_groupby_sum('payment_method'),
            'by_month': by_month_serializable,
        }

        return profitability_summary
    
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

    def rfm_loyalty_analysis(self) -> Dict:
        """
        Perform RFM-based loyalty analysis and return distributions.

        Returns:
            Dict: distributions and top segments
        """
        metrics = self.calculate_customer_metrics()
        segmented = self.segment_customers(metrics)

        rfm_bins = {
            'recency_score': segmented['recency_score'].value_counts().sort_index().to_dict(),
            'frequency_score': segmented['frequency_score'].value_counts().sort_index().to_dict(),
            'monetary_score': segmented['monetary_score'].value_counts().sort_index().to_dict(),
        }
        segments = segmented['customer_segment'].value_counts().to_dict()

        # Top customers sample
        top_customers = (
            segmented.sort_values(['rfm_score', 'total_spent', 'transaction_count'], ascending=[False, False, False])
            .head(20)
            .reset_index()
            .to_dict(orient='records')
        )

        return {
            'rfm_distribution': rfm_bins,
            'segment_counts': segments,
            'top_customers_sample': top_customers,
        }

    def top_customers(self, top_percent: float = 0.10, min_transactions: int = 1) -> Dict:
        """
        Compute top X% customers by total spend, optionally filtering by min transactions.

        Args:
            top_percent: fraction between 0 and 1
            min_transactions: minimum number of transactions

        Returns:
            Dict: summary and list of top customers
        """
        metrics = self.calculate_customer_metrics()
        if min_transactions > 1:
            metrics = metrics[metrics['transaction_count'] >= min_transactions]

        metrics = metrics.sort_values('total_spent', ascending=False)
        n = max(1, int(len(metrics) * top_percent))
        top = metrics.head(n).reset_index()

        return {
            'selected': n,
            'total_customers': int(self.processed_df['customer_id'].nunique()),
            'top_percent': float(top_percent),
            'customers': top[['customer_id', 'total_spent', 'transaction_count', 'rfm_score']].to_dict(orient='records'),
        }

    def repeat_vs_onetime(self) -> Dict:
        """
        Compare contribution between one-time buyers and repeat customers.
        """
        metrics = self.calculate_customer_metrics()
        metrics = metrics.reset_index()
        metrics['is_repeat'] = metrics['transaction_count'] > 1

        df = self.processed_df
        # Map customer-level repeat flag to transactions
        cust_flag = metrics.set_index('customer_id')['is_repeat']
        tx = df.join(cust_flag, on='customer_id')

        summary = tx.groupby('is_repeat').agg(
            total_revenue=('total_amount', 'sum'),
            transactions=('invoice_no', 'nunique'),
            customers=('customer_id', 'nunique'),
        ).round(2).reset_index()

        out: Dict[str, Dict[str, float]] = {}
        for _, row in summary.iterrows():
            key = 'repeat' if bool(row['is_repeat']) else 'one_time'
            out[key] = {
                'total_revenue': float(row['total_revenue']),
                'transactions': int(row['transactions']),
                'customers': int(row['customers']),
            }
        return out

    def categories_kpis(self) -> Dict:
        """
        Category-level KPIs including revenue share, profit, and margin.
        """
        # Reuse profitability logic with default rates
        prof = self.calculate_profitability()
        cat = prof['by_category']
        # cat is {category: {net_revenue, cogs, profit}}
        total_net = sum(v['net_revenue'] for v in cat.values()) or 1.0
        result: Dict[str, Dict[str, float]] = {}
        for k, v in cat.items():
            net = float(v['net_revenue'])
            profit = float(v['profit'])
            margin = (profit / net) if net > 0 else 0.0
            result[str(k)] = {
                'net_revenue': net,
                'profit': profit,
                'margin_pct': float(round(margin, 4)),
                'revenue_share_pct': float(round(net / total_net, 4)),
            }
        return result

    def simulate_campaign(self, segment: str = 'High Value', discount: float = 0.10) -> Dict:
        """
        Simulate a discount campaign targeting a customer segment; estimate ROI.

        Args:
            segment: target segment label from segment_customers
            discount: discount rate (0-1)
        """
        metrics = self.calculate_customer_metrics()
        segmented = self.segment_customers(metrics)

        # Identify targeted customers
        target = segmented[segmented['customer_segment'] == segment]
        targeted_ids = set(target.index)

        df = self.processed_df.copy()
        df['is_target'] = df['customer_id'].isin(targeted_ids)

        # Baseline profitability (no extra discount)
        base = self.calculate_profitability()
        base_profit = float(base['total_profit'])

        # Apply additional discount to targeted customers only
        df['extra_discount'] = np.where(df['is_target'], discount, 0.0)
        df['net_price_sim'] = (df['price'] * (1 - df.get('discount_rate', 0) - df['extra_discount'])).clip(lower=0)
        df['net_revenue_sim'] = (df['net_price_sim'] * df['quantity']).round(2)

        # Approximate COGS as before using default rule
        default_cost_rate = 0.6
        df['unit_cost_sim'] = (df['price'] * default_cost_rate).round(2)
        df['cogs_sim'] = (df['unit_cost_sim'] * df['quantity']).round(2)
        df['profit_sim'] = (df['net_revenue_sim'] - df['cogs_sim']).round(2)

        sim_profit = float(df['profit_sim'].sum().round(2))
        uplift = sim_profit - base_profit

        # Cost of discount given
        discount_cost = float(((df['price'] * df['extra_discount']) * df['quantity']).sum().round(2))
        roi = (uplift / discount_cost) if discount_cost > 0 else 0.0

        return {
            'target_segment': segment,
            'targeted_customers': int(len(targeted_ids)),
            'base_profit': base_profit,
            'simulated_profit': sim_profit,
            'profit_uplift': float(round(uplift, 2)),
            'discount_cost': discount_cost,
            'roi': float(round(roi, 4)),
        }

    def train_kmeans_segmentation(self, n_clusters: int = 5, random_state: int = 42) -> Dict:
        """
        Train a simple KMeans model on customer metrics and return labels and summary.

        Args:
            n_clusters: number of clusters
            random_state: rng seed

        Returns:
            Dict: cluster counts and sample assignments
        """
        metrics = self.calculate_customer_metrics()
        features = metrics[[
            'total_spent',
            'avg_transaction_value',
            'transaction_count',
            'days_since_last_purchase',
            'categories_purchased',
            'malls_visited',
        ]].fillna(0.0)

        labels, inertia = _train_kmeans(metrics, n_clusters=n_clusters, random_state=random_state)
        metrics = metrics.copy()
        metrics['kmeans_cluster'] = labels
        counts = metrics['kmeans_cluster'].value_counts().sort_index().to_dict()
        sample = metrics.reset_index().head(20)[['customer_id', 'kmeans_cluster']].to_dict(orient='records')
        return {
            'n_clusters': n_clusters,
            'cluster_counts': {str(int(k)): int(v) for k, v in counts.items()},
            'inertia': float(inertia),
            'sample_assignments': sample,
        }

    def forecast_monthly_sales_naive(self, periods: int = 3) -> Dict:
        """
        Naive monthly sales forecast using last observed value.

        Args:
            periods: number of future months to forecast

        Returns:
            Dict: history and forecast series
        """
        if self.processed_df is None:
            raise ValueError("Data not processed. Call clean_data() first.")

        df = self.processed_df
        hist = (
            df.groupby(['year', 'month'])['total_amount']
            .sum()
            .reset_index()
            .sort_values(['year', 'month'])
        )
        return _naive_forecast(hist, periods=periods)
    
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
        
        # Monthly trends (JSON-safe keys and values)
        monthly_df = (
            df.groupby(['year', 'month']).agg(
                total_revenue=('total_amount', 'sum'),
                transaction_count=('total_amount', 'count'),
                unique_customers=('customer_id', 'nunique'),
            ).round(2).reset_index()
        )
        monthly_trends: Dict[str, Dict[str, float]] = {}
        for _, row in monthly_df.iterrows():
            key = f"{int(row['year'])}-{int(row['month']):02d}"
            monthly_trends[key] = {
                'total_revenue': float(row['total_revenue']),
                'transaction_count': int(row['transaction_count']),
                'unique_customers': int(row['unique_customers']),
            }

        # Quarterly trends
        quarterly_df = (
            df.groupby(['year', 'quarter']).agg(
                total_revenue=('total_amount', 'sum'),
                transaction_count=('total_amount', 'count'),
                unique_customers=('customer_id', 'nunique'),
            ).round(2).reset_index()
        )
        quarterly_trends: Dict[str, Dict[str, float]] = {}
        for _, row in quarterly_df.iterrows():
            key = f"{int(row['year'])}-Q{int(row['quarter'])}"
            quarterly_trends[key] = {
                'total_revenue': float(row['total_revenue']),
                'transaction_count': int(row['transaction_count']),
                'unique_customers': int(row['unique_customers']),
            }

        # Day of week trends
        dow_df = (
            df.groupby('day_of_week').agg(
                total_revenue=('total_amount', 'sum'),
                avg_revenue=('total_amount', 'mean'),
                transaction_count=('total_amount', 'count'),
            ).round(2)
        )
        dow_trends: Dict[str, Dict[str, float]] = {}
        for dow, row in dow_df.iterrows():
            dow_trends[str(dow)] = {
                'total_revenue': float(row['total_revenue']),
                'avg_revenue': float(row['avg_revenue']),
                'transaction_count': int(row['transaction_count']),
            }

        # Category seasonal analysis (category -> month -> revenue)
        cat_month = (
            df.groupby(['category', 'month'])['total_amount']
            .sum()
            .unstack(fill_value=0)
            .round(2)
        )
        category_seasonality: Dict[str, Dict[str, float]] = {}
        for category, row in cat_month.iterrows():
            month_map: Dict[str, float] = {}
            for m, val in row.items():
                month_map[f"{int(m):02d}"] = float(val)
            category_seasonality[str(category)] = month_map

        seasonal_analysis = {
            'monthly_trends': monthly_trends,
            'quarterly_trends': quarterly_trends,
            'day_of_week_trends': dow_trends,
            'category_seasonality': category_seasonality,
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
        
        # Payment method analysis with named aggregations (avoids MultiIndex)
        payment_summary_df = (
            df.groupby('payment_method').agg(
                total_revenue=('total_amount', 'sum'),
                avg_transaction_value=('total_amount', 'mean'),
                transaction_count=('total_amount', 'count'),
                unique_customers=('customer_id', 'nunique'),
                avg_quantity=('quantity', 'mean'),
            ).round(2)
        )

        # Convert to serializable dict
        payment_summary: Dict[str, Dict[str, float]] = {}
        for method, row in payment_summary_df.iterrows():
            payment_summary[str(method)] = {
                'total_revenue': float(row['total_revenue']),
                'avg_transaction_value': float(row['avg_transaction_value']),
                'transaction_count': int(row['transaction_count']),
                'unique_customers': int(row['unique_customers']),
                'avg_quantity': float(row['avg_quantity']),
            }

        # Payment method by category/mall/age group with Python floats
        def table_to_serializable(table_index_cols: List[str]) -> Dict[str, Dict[str, float]]:
            pivot = (
                df.groupby(table_index_cols)['total_amount']
                .sum()
                .unstack(fill_value=0)
                .round(2)
            )
            out: Dict[str, Dict[str, float]] = {}
            for idx, row in pivot.iterrows():
                row_dict: Dict[str, float] = {}
                for col, val in row.items():
                    row_dict[str(col)] = float(val)
                out[str(idx)] = row_dict
            return out

        payment_by_category = table_to_serializable(['payment_method', 'category'])
        payment_by_mall = table_to_serializable(['payment_method', 'shopping_mall'])
        payment_by_age = table_to_serializable(['payment_method', 'age_group'])

        analysis_results = {
            'payment_summary': payment_summary,
            'payment_by_category': payment_by_category,
            'payment_by_mall': payment_by_mall,
            'payment_by_age': payment_by_age,
        }
        
        logger.info("Payment method analysis completed")
        
        return analysis_results


def main():
    """Main function to demonstrate data processing."""
    # Initialize processor
    processor = DataProcessor('..\customer_shopping_data.csv')
    
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
