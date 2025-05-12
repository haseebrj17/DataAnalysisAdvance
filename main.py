import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta
import os
import warnings
import json
from langchain.llms import OpenAI
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.memory import ConversationBufferMemory

# Set display options and suppress warnings
pd.set_option('display.max_columns', None)
warnings.filterwarnings('ignore')

# Project constants
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

class RetailAnalytics:
    """
    A comprehensive class for retail sales analytics combining time series analysis
    and business intelligence techniques.
    """

    def __init__(self, sales_data_path=None, api_key=None):
        """
        Initialize the RetailAnalytics class.

        Parameters:
        -----------
        sales_data_path : str
            Path to the sales data CSV file
        api_key : str
            OpenAI API key for LLM integration
        """
        self.sales_df = None
        self.features_df = None
        self.merged_df = None
        self.models = {}
        self.api_key = api_key
        self.memory = ConversationBufferMemory(return_messages=True)
        self.insights_cache = {}  # Initialize as empty dict
        self.visualizations = {}

        if sales_data_path:
            self.load_data(sales_data_path)

    def load_data(self, sales_path, features_path=None):
        """
        Load sales data and optional features data

        Parameters:
        -----------
        sales_path : str
            Path to the sales data CSV file
        features_path : str, optional
            Path to the features data CSV file
        """
        print(f"Loading data from {sales_path}...")

        # Load sales data
        self.sales_df = pd.read_csv(sales_path)

        # Check if date column exists, if not try to infer it
        if 'Date' in self.sales_df.columns:
            try:
                self.sales_df['Date'] = pd.to_datetime(self.sales_df['Date'], errors='coerce')
            except Exception as e:
                print(f"Warning: Date conversion issue: {e}")
        else:
            date_candidates = [col for col in self.sales_df.columns
                               if 'date' in col.lower() or 'time' in col.lower()]
            if date_candidates:
                self.sales_df['Date'] = pd.to_datetime(self.sales_df[date_candidates[0]])

        # Load features data if provided
        if features_path:
            self.features_df = pd.read_csv(features_path)
            if 'Date' in self.features_df.columns:
                self.features_df['Date'] = pd.to_datetime(self.features_df['Date'])

            # Merge sales and features data
            self.merged_df = pd.merge(self.sales_df, self.features_df,
                                      on=['Date'], how='left')
        else:
            self.merged_df = self.sales_df.copy()

        print(f"Data loaded successfully. Shape: {self.merged_df.shape}")

        # Initialize RAG system with the data
        self._initialize_rag_system()

    def _initialize_rag_system(self):
        """
        Initialize the Retrieval-Augmented Generation system for querying the data using ChromaDB.
        Creates document embeddings and sets up the retrieval system.
        """
        if self.api_key is None:
            print("Warning: No API key provided. RAG system not initialized.")
            return

        try:
            # Prepare the data summaries for RAG
            self.generate_data_summaries()

            # Import ChromaDB modules
            from langchain_chroma import Chroma
            from langchain.document_loaders import DataFrameLoader
            from langchain_openai import OpenAIEmbeddings
            from langchain_openai import ChatOpenAI
            from langchain.chains.retrieval_qa.base import RetrievalQA
            import uuid

            # Convert dataframe to documents for embedding
            data_loader = DataFrameLoader(self.merged_df)
            documents = data_loader.load()

            # Create OpenAI embeddings
            self.embeddings = OpenAIEmbeddings(api_key=self.api_key)

            # Create a unique persistent directory for ChromaDB
            persist_directory = f"chroma_db_{uuid.uuid4().hex[:8]}"

            try:
                # Initialize ChromaDB vector store
                self.vectorstore = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embeddings,
                    persist_directory=persist_directory
                )
            except Exception as e:
                print(f"Error initializing ChromaDB: {e}")
                if "text" in str(e):
                    print("This may be related to document formatting - check document content types")
                self.vectorstore = None

            # Create retriever
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )

            # Create QA chain
            self.llm = ChatOpenAI(temperature=0, api_key=self.api_key)
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.retriever,
                return_source_documents=True
            )

            print("RAG system initialized successfully with ChromaDB")
        except Exception as e:
            print(f"Error initializing RAG system: {e}")
            # Fallback to simple approach if ChromaDB fails
            try:
                # Initialize a simple OpenAI model for insights
                from langchain.llms import OpenAI
                self.llm = OpenAI(openai_api_key=self.api_key)
                print("Simple LLM system initialized as fallback (without vector search)")
            except Exception as nested_e:
                print(f"Error initializing fallback LLM system: {nested_e}")

    def generate_data_summaries(self):
        """
        Generate key summaries about the data that will be used for RAG
        """
        # Overall statistics summary
        self.data_summary = {
            "dataset_info": {
                "rows": len(self.merged_df),
                "columns": list(self.merged_df.columns),
                "date_range": {
                    "start": self.merged_df['Date'].min().strftime('%Y-%m-%d'),
                    "end": self.merged_df['Date'].max().strftime('%Y-%m-%d')
                }
            }
        }

        # Generate numerical summaries
        numeric_cols = self.merged_df.select_dtypes(include=['float64', 'int64']).columns
        self.data_summary["numeric_stats"] = self.merged_df[numeric_cols].describe().to_dict()

        # Generate categorical summaries
        categorical_cols = self.merged_df.select_dtypes(include=['object']).columns
        categorical_stats = {}
        for col in categorical_cols:
            if col != 'Date':
                categorical_stats[col] = self.merged_df[col].value_counts().to_dict()

        self.data_summary["categorical_stats"] = categorical_stats

        # Add time-based aggregations if 'Sales' column exists
        if 'Sales' in self.merged_df.columns:
            # Monthly sales
            monthly_sales = self.merged_df.groupby(self.merged_df['Date'].dt.to_period('M'))['Sales'].agg(['sum', 'mean', 'count'])
            self.data_summary["time_based_stats"] = {
                "monthly_sales": monthly_sales.to_dict()
            }

            # Product performance if Product column exists
            if 'Product' in self.merged_df.columns:
                product_sales = self.merged_df.groupby('Product')['Sales'].agg(['sum', 'mean', 'count']).sort_values('sum', ascending=False)
                self.data_summary["product_stats"] = product_sales.to_dict()

            # Region performance if Region column exists
            if 'Region' in self.merged_df.columns:
                region_sales = self.merged_df.groupby('Region')['Sales'].agg(['sum', 'mean', 'count']).sort_values('sum', ascending=False)
                self.data_summary["region_stats"] = region_sales.to_dict()

        print("Data summaries generated successfully")

    def clean_data(self):
        """
        Clean and preprocess the data
        """
        print("Cleaning data...")

        df = self.merged_df.copy()

        # Convert date column if it exists
        if 'Date' in df.columns:
            try:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                # Handle any NaT values
                df = df.dropna(subset=['Date'])
            except Exception as e:
                print(f"Warning: Date cleaning issue: {e}")

        # Handle missing values - using advanced imputation
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

        for col in numeric_cols:
            # If less than 5% missing, use median
            if df[col].isna().sum() / len(df) < 0.05:
                df[col] = df[col].fillna(df[col].median())
            # If more than 5% missing, use more advanced methods
            elif 'Product' in df.columns and 'Region' in df.columns:
                # Fill with group median first
                df[col] = df.groupby(['Product', 'Region'])[col].transform(
                    lambda x: x.fillna(x.median()))

                # Fill remaining with overall median
                df[col] = df[col].fillna(df[col].median())
            else:
                # Simple median imputation
                df[col] = df[col].fillna(df[col].median())

        # Add derived time features
        if 'Date' in df.columns:
            df['Year'] = df['Date'].dt.year
            df['Month'] = df['Date'].dt.month
            df['Week'] = df['Date'].dt.isocalendar().week
            df['DayOfWeek'] = df['Date'].dt.dayofweek
            df['Quarter'] = df['Date'].dt.quarter
            df['YearMonth'] = df['Date'].dt.strftime('%Y-%m')
            df['MonthDay'] = df['Date'].dt.strftime('%m-%d')

            # Add seasonality features
            df['SinWeek'] = np.sin(2 * np.pi * df['Week'] / 52)
            df['CosWeek'] = np.cos(2 * np.pi * df['Week'] / 52)

        # Store the cleaned data
        self.merged_df = df

        print("Data cleaning completed successfully.")
        return self.merged_df

    def explore_data(self, save_path=None):
        """
        Perform exploratory data analysis

        Parameters:
        -----------
        save_path : str, optional
            Path to save the plots
        """
        print("Performing exploratory data analysis...")

        df = self.merged_df

        # Summary statistics
        summary = df.describe().T
        print("\nSummary Statistics:")
        print(summary)

        # Create plots directory if saving plots
        if save_path:
            os.makedirs(save_path, exist_ok=True)

        # Create a set of plots to understand the data
        self._plot_sales_trends(save_path)
        self._plot_correlation_heatmap(save_path)
        self._plot_feature_relationships(save_path)

        if 'Region' in df.columns:
            self._plot_region_comparison(save_path)

        # Generate customer insights if demographic data exists
        if 'Customer_Age' in df.columns or 'Customer_Gender' in df.columns:
            self._analyze_customer_demographics(save_path)

        print("Exploratory data analysis completed.")

    def _plot_sales_trends(self, save_path=None):
        """Plot sales trends over time"""
        df = self.merged_df

        # Ensure we have a date column
        if 'Date' not in df.columns:
            print("Cannot plot time trends: No Date column found")
            return

        # Identify sales column
        if 'Sales' in df.columns:
            sales_col = 'Sales'
        else:
            # Find a column that might represent sales
            potential_sales_cols = [col for col in df.columns if 'sale' in col.lower()]
            if potential_sales_cols:
                sales_col = potential_sales_cols[0]
            else:
                print("Cannot plot sales trends: No sales column identified")
                return

        # Create time series plot - plotly version
        sales_by_date = df.groupby('Date')[sales_col].sum().reset_index()

        fig = px.line(sales_by_date, x='Date', y=sales_col,
                      title='Total Sales Over Time')

        # Add trend line
        x = np.array(range(len(sales_by_date)))
        y = sales_by_date[sales_col].values
        trend_coeffs = np.polyfit(x, y, 1)
        trend_line = np.polyval(trend_coeffs, x)

        fig.add_scatter(x=sales_by_date['Date'], y=trend_line,
                        mode='lines', name='Trend', line=dict(color='red', dash='dash'))

        fig.update_layout(
            xaxis_title='Date',
            yaxis_title=sales_col,
            legend_title='Legend',
            hovermode='x unified'
        )

        # Store the visualization
        self.visualizations['sales_trends'] = fig

        if save_path:
            fig.write_html(os.path.join(save_path, 'sales_trends.html'))

        # Create additional time-based visualizations
        if 'Product' in df.columns:
            # Sales by product over time
            product_sales = df.groupby(['Date', 'Product'])[sales_col].sum().reset_index()

            fig = px.line(product_sales, x='Date', y=sales_col, color='Product',
                          title='Sales by Product Over Time')

            fig.update_layout(
                xaxis_title='Date',
                yaxis_title=sales_col,
                legend_title='Product',
                hovermode='x unified'
            )

            # Store the visualization
            self.visualizations['product_sales_trend'] = fig

            if save_path:
                fig.write_html(os.path.join(save_path, 'product_sales_trend.html'))

    def _plot_correlation_heatmap(self, save_path=None):
        """Plot correlation heatmap of numerical features"""
        df = self.merged_df

        # Select only numeric columns
        numeric_df = df.select_dtypes(include=['float64', 'int64'])

        # Drop any columns that are all NaN
        numeric_df = numeric_df.dropna(axis=1, how='all')

        # Calculate correlation matrix
        corr_matrix = numeric_df.corr()

        # Create correlation heatmap with plotly
        fig = px.imshow(corr_matrix,
                        text_auto='.2f',
                        aspect='auto',
                        title='Feature Correlation Matrix')

        fig.update_layout(
            width=800,
            height=800
        )

        # Store the visualization
        self.visualizations['correlation_heatmap'] = fig

        if save_path:
            fig.write_html(os.path.join(save_path, 'correlation_heatmap.html'))

    def _plot_feature_relationships(self, save_path=None):
        """Plot relationships between sales and other features"""
        df = self.merged_df

        # Identify sales column
        if 'Sales' in df.columns:
            sales_col = 'Sales'
        else:
            potential_sales_cols = [col for col in df.columns if 'sale' in col.lower()]
            if potential_sales_cols:
                sales_col = potential_sales_cols[0]
            else:
                print("Cannot plot feature relationships: No sales column identified")
                return

        # Identify potential feature columns
        feature_cols = []
        for potential_feature in ['Customer_Age', 'Customer_Satisfaction']:
            if potential_feature in df.columns:
                feature_cols.append(potential_feature)

        if not feature_cols:
            print("No standard feature columns found to plot relationships")
            return

        # Create interactive scatter plots for each feature
        for feature in feature_cols:
            fig = px.scatter(df, x=feature, y=sales_col,
                             color='Customer_Gender' if 'Customer_Gender' in df.columns else None,
                             trendline='ols',
                             title=f'Relationship Between {sales_col} and {feature}')

            fig.update_layout(
                xaxis_title=feature,
                yaxis_title=sales_col,
                legend_title='Gender' if 'Customer_Gender' in df.columns else None
            )

            # Store the visualization
            self.visualizations[f'sales_vs_{feature}'] = fig

            if save_path:
                fig.write_html(os.path.join(save_path, f'sales_vs_{feature}.html'))

    def _plot_region_comparison(self, save_path=None):
        """Plot comparison of sales across regions"""
        df = self.merged_df

        # Identify sales column
        if 'Sales' in df.columns:
            sales_col = 'Sales'
        else:
            potential_sales_cols = [col for col in df.columns if 'sale' in col.lower()]
            if potential_sales_cols:
                sales_col = potential_sales_cols[0]
            else:
                print("Cannot plot region comparison: No sales column identified")
                return

        # Calculate total sales by region
        if 'Region' in df.columns:
            region_sales = df.groupby('Region')[sales_col].sum().reset_index()

            # Create bar chart of regions by total sales
            fig = px.bar(region_sales, x='Region', y=sales_col,
                         title='Total Sales by Region',
                         color='Region')

            fig.update_layout(
                xaxis_title='Region',
                yaxis_title=f'Total {sales_col}'
            )

            # Store the visualization
            self.visualizations['sales_by_region'] = fig

            if save_path:
                fig.write_html(os.path.join(save_path, 'sales_by_region.html'))

            # If we have product data, create heatmap of product performance by region
            if 'Product' in df.columns:
                # Get region-product total sales
                region_product_sales = df.groupby(['Region', 'Product'])[sales_col].sum().reset_index()

                # Pivot to create a matrix of region vs product
                pivot_sales = region_product_sales.pivot(index='Region', columns='Product', values=sales_col)

                # Create heatmap
                fig = px.imshow(pivot_sales,
                                text_auto='.0f',
                                aspect='auto',
                                title='Sales by Region and Product')

                fig.update_layout(
                    xaxis_title='Product',
                    yaxis_title='Region'
                )

                # Store the visualization
                self.visualizations['region_product_heatmap'] = fig

                if save_path:
                    fig.write_html(os.path.join(save_path, 'region_product_heatmap.html'))

    def generate_insights(self, query=None):
        """
        Generate business intelligence insights using RAG approach

        Parameters:
        -----------
        query : str, optional
            Specific query to generate insights for

        Returns:
        --------
        dict
            Dictionary with insights
        """
        if self.api_key is None:
            print("Cannot generate insights: No API key provided for LLM")
            return {"error": "No API key provided for LLM"}

        # If no specific query, generate general insights
        if query is None:
            query = "What are the key business insights from the sales data?"

        # Initialize insights_cache as dict if it doesn't exist or is not a dict
        if not hasattr(self, 'insights_cache') or not isinstance(self.insights_cache, dict):
            self.insights_cache = {}

        # Check if we have cached insights for this query
        if query in self.insights_cache:
            return self.insights_cache[query]

        try:
            # Prepare context by analyzing the data
            # This will include summaries of key metrics and patterns
            df = self.merged_df

            # Identify sales column
            sales_col = 'Sales' if 'Sales' in df.columns else df.columns[1]

            # Calculate key metrics for context
            total_sales = df[sales_col].sum()
            avg_sales = df[sales_col].mean()

            # Get top products if available
            product_insights = ""
            if 'Product' in df.columns:
                product_sales = df.groupby('Product')[sales_col].sum().sort_values(ascending=False)
                top_product = product_sales.index[0]
                bottom_product = product_sales.index[-1]
                product_insights = f"Top selling product: {top_product}, Lowest selling product: {bottom_product}."

            # Get regional insights if available
            region_insights = ""
            if 'Region' in df.columns:
                region_sales = df.groupby('Region')[sales_col].sum().sort_values(ascending=False)
                top_region = region_sales.index[0]
                bottom_region = region_sales.index[-1]
                region_insights = f"Best performing region: {top_region}, Lowest performing region: {bottom_region}."

            # Get customer demographic insights if available
            customer_insights = ""
            if 'Customer_Age' in df.columns and 'Customer_Gender' in df.columns:
                age_gender_sales = df.groupby(['Customer_Gender', pd.cut(df['Customer_Age'],
                                                                         bins=[0, 25, 35, 45, 55, 100],
                                                                         labels=['18-25', '26-35', '36-45', '46-55', '56+'])])[sales_col].mean()
                customer_insights = f"Customer demographics show varying purchasing patterns across age groups and genders."

            # Get time-based insights if available
            time_insights = ""
            if 'Date' in df.columns:
                monthly_sales = df.groupby(df['Date'].dt.to_period('M'))[sales_col].sum()
                peak_month = monthly_sales.idxmax()
                lowest_month = monthly_sales.idxmin()
                time_insights = f"Peak sales period: {peak_month}, Lowest sales period: {lowest_month}."

            # Combine context
            analysis_context = f"""
            Dataset contains {len(df)} records with total sales of {total_sales}. 
            Average sale value is {avg_sales:.2f}. 
            {product_insights}
            {region_insights}
            {customer_insights}
            {time_insights}
            """

            # Create prompt for the LLM
            prompt_template = f"""
            You are an expert business intelligence analyst looking at retail sales data.

            Here is a summary of the key metrics from the data:
            {analysis_context}

            Based on this information, please provide insightful business analysis to answer the following question:
            {query}

            Your answer should:
            1. Be concise and actionable
            2. Focus on key insights rather than restating the data
            3. Provide strategic recommendations when relevant
            4. Consider multiple facets (products, regions, customer segments, time trends)
            5. Be backed by the data provided

            IMPORTANT: Only make claims that are supported by the data summary provided.
            """

            # Generate insights using RAG
            insights = ""
            if hasattr(self, 'qa_chain'):
                try:
                    result = self.qa_chain({"query": prompt_template})
                    insights = result['result']
                except Exception as e:
                    print(f"Error using QA chain: {e}")
                    # Fall back to OpenAI directly
                    if hasattr(self, 'llm'):
                        response = self.llm(prompt_template)
                        insights = response.content if hasattr(response, 'content') else str(response)
                    else:
                        # Create a new LLM instance
                        llm = OpenAI(openai_api_key=self.api_key)
                        insights = llm(prompt_template)
            else:
                # Fallback to a simpler approach using OpenAI directly
                try:
                    llm = OpenAI(openai_api_key=self.api_key)
                    insights = llm(prompt_template)
                except Exception as e:
                    print(f"Error using OpenAI API: {e}")
                    insights = "Unable to generate AI insights. Please check your API key."

            # Cache the insights
            insight_result = {
                "query": query,
                "insights": insights,
                "context": analysis_context,
                "timestamp": datetime.now().isoformat()
            }
            self.insights_cache[query] = insight_result

            return insight_result

        except Exception as e:
            print(f"Error generating insights: {e}")
            return {"error": str(e)}

    def generate_example_data(self, n_stores=5, n_departments=5, start_date='2022-01-01', periods=365):
        """Generate example retail data for demo purposes"""

        # Create date range
        dates = pd.date_range(start=start_date, periods=periods)

        # Create sample store and department IDs
        stores = range(1, n_stores + 1)
        departments = range(1, n_departments + 1)

        # Create product list
        products = ['Electronics', 'Clothing', 'Groceries', 'Home Goods', 'Sports']
        regions = ['North', 'South', 'East', 'West', 'Central']
        genders = ['Male', 'Female']

        # Generate random data
        data = []
        for date in dates:
            for store in stores:
                for dept in departments:
                    # Add some seasonality and trends
                    base_sales = 1000 + (store * 100) + (dept * 50)
                    seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * date.dayofyear / 365)
                    trend_factor = 1 + (0.001 * (date - pd.to_datetime(start_date)).days)

                    # Random factors
                    random_factor = np.random.normal(1, 0.2)

                    # Calculate sales
                    sales = base_sales * seasonal_factor * trend_factor * random_factor

                    # Customer info
                    customer_age = np.random.randint(18, 65)
                    customer_gender = np.random.choice(genders)
                    customer_satisfaction = round(np.random.uniform(1, 5), 1)

                    # Create record
                    record = {
                        'Date': date,
                        'Store': store,
                        'Dept': dept,
                        'Product': np.random.choice(products),
                        'Region': np.random.choice(regions),
                        'Sales': round(sales),
                        'Customer_Age': customer_age,
                        'Customer_Gender': customer_gender,
                        'Customer_Satisfaction': customer_satisfaction
                    }
                    data.append(record)

        # Create DataFrame
        self.sales_df = pd.DataFrame(data)
        self.merged_df = self.sales_df.copy()

        print(f"Generated example data with {len(self.sales_df)} records")
        return self.sales_df

    def get_visualization(self, viz_name):
        """
        Get a specific visualization by name

        Parameters:
        -----------
        viz_name : str
            Name of the visualization to retrieve

        Returns:
        --------
        plotly.graph_objects.Figure
            The requested visualization
        """
        if viz_name in self.visualizations:
            return self.visualizations[viz_name]
        else:
            print(f"Visualization '{viz_name}' not found")
            return None

    def export_report(self, format='html', include_insights=True, include_visualizations=True):
        """
        Export a comprehensive report

        Parameters:
        -----------
        format : str
            Report format ('html', 'json', 'csv')
        include_insights : bool
            Whether to include AI-generated insights
        include_visualizations : bool
            Whether to include visualizations

        Returns:
        --------
        str or dict
            Report content
        """
        # Generate report data
        report = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "data_shape": self.merged_df.shape,
                "date_range": {
                    "start": str(self.merged_df['Date'].min()) if 'Date' in self.merged_df else None,
                    "end": str(self.merged_df['Date'].max()) if 'Date' in self.merged_df else None
                }
            },
            "summary_statistics": {}
        }

        # Add summary statistics
        numeric_cols = self.merged_df.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            report["summary_statistics"][col] = {
                "mean": float(self.merged_df[col].mean()),
                "median": float(self.merged_df[col].median()),
                "std": float(self.merged_df[col].std()),
                "min": float(self.merged_df[col].min()),
                "max": float(self.merged_df[col].max())
            }

        # Add insights
        if include_insights and self.insights_cache:
            # Check the type of insights_cache and handle accordingly
            if isinstance(self.insights_cache, dict):
                # Keep it as a dictionary with original keys
                report["insights"] = self.insights_cache

                # Alternatively, if you need to extract values but maintain dict structure:
                insights_dict = {}
                for i, (key, value) in enumerate(self.insights_cache.items()):
                    insights_dict[f"insight_{i}"] = value
                report["insights"] = insights_dict
            if isinstance(self.insights_cache, list):
                # Convert list to dictionary with keys
                insights_dict = {}
                for i, insight in enumerate(self.insights_cache):
                    insights_dict[f"insight_{i}"] = insight
                report["insights"] = insights_dict
            else:
                # Unexpected format: convert to string representation
                report["insights"] = {"System insights": {"query": "System insights",
                                                          "insights": f"Insights data in unexpected format: {type(self.insights_cache)}",
                                                          "timestamp": datetime.now().isoformat()}}

        # Add model evaluation if available
        if self.models:
            report["models"] = {}
            for model_name, model_results in self.models.items():
                if "test_metrics" in model_results:
                    report["models"][model_name] = {
                        "metrics": model_results["test_metrics"]
                    }
                elif "metrics" in model_results:
                    report["models"][model_name] = {
                        "metrics": model_results["metrics"]
                    }

        # Format the report
        if format == 'json':
            # Convert visualizations to image data if needed
            if include_visualizations and self.visualizations:
                report["visualizations"] = {}
                for viz_name, viz_fig in self.visualizations.items():
                    try:
                        report["visualizations"][viz_name] = {
                            "data": viz_fig.to_json(),
                            "type": "plotly"
                        }
                    except Exception as e:
                        print(f"Error exporting visualization {viz_name}: {e}")

            return json.dumps(report, indent=2)

        elif format == 'html':
            # Create HTML report
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>InsightForge BI Report</title>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .header {{ background-color: #4CAF50; color: white; padding: 20px; }}
                    .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; }}
                    .insight-card {{ background-color: #f9f9f9; padding: 15px; margin-bottom: 15px; border-left: 5px solid #4CAF50; }}
                    .viz-container {{ height: 500px; margin: 20px 0; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }}
                    tr:hover {{ background-color: #f5f5f5; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>InsightForge Business Intelligence Report</h1>
                    <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>

                <div class="section">
                    <h2>Dataset Overview</h2>
                    <p>Records: {self.merged_df.shape[0]}</p>
                    <p>Features: {self.merged_df.shape[1]}</p>
                    <p>Date Range: {self.merged_df['Date'].min().strftime('%Y-%m-%d') if 'Date' in self.merged_df else 'N/A'} to 
                     {self.merged_df['Date'].max().strftime('%Y-%m-%d') if 'Date' in self.merged_df else 'N/A'}</p>
                </div>
            """

            # Add insights section
            if include_insights and self.insights_cache:
                html_content += """
                <div class="section">
                    <h2>Key Business Insights</h2>
                """

                for query, insight in self.insights_cache.items():
                    html_content += f"""
                    <div class="insight-card">
                        <h3>Query: {query}</h3>
                        <p>{insight['insights']}</p>
                    </div>
                    """

                html_content += "</div>"

            # Add visualizations
            if include_visualizations and self.visualizations:
                html_content += """
                <div class="section">
                    <h2>Data Visualizations</h2>
                """

                for viz_name, viz_fig in self.visualizations.items():
                    viz_id = f"viz_{viz_name.replace(' ', '_')}"
                    html_content += f"""
                    <h3>{viz_name.replace('_', ' ').title()}</h3>
                    <div id="{viz_id}" class="viz-container"></div>
                    <script>
                        var data = {viz_fig.to_json()};
                        Plotly.newPlot('{viz_id}', data.data, data.layout);
                    </script>
                    """

                html_content += "</div>"

            # Add statistics section
            html_content += """
            <div class="section">
                <h2>Summary Statistics</h2>
                <table>
                    <tr>
                        <th>Feature</th>
                        <th>Mean</th>
                        <th>Median</th>
                        <th>Std Dev</th>
                        <th>Min</th>
                        <th>Max</th>
                    </tr>
            """

            for col in numeric_cols:
                html_content += f"""
                <tr>
                    <td>{col}</td>
                    <td>{self.merged_df[col].mean():.2f}</td>
                    <td>{self.merged_df[col].median():.2f}</td>
                    <td>{self.merged_df[col].std():.2f}</td>
                    <td>{self.merged_df[col].min():.2f}</td>
                    <td>{self.merged_df[col].max():.2f}</td>
                </tr>
                """

            html_content += """
                </table>
            </div>
            """

            # Add model evaluation if available
            if self.models:
                html_content += """
                <div class="section">
                    <h2>Model Evaluation</h2>
                    <table>
                        <tr>
                            <th>Model</th>
                            <th>MAE</th>
                            <th>RMSE</th>
                            <th>MAPE</th>
                        </tr>
                """

                for model_name, model_results in self.models.items():
                    metrics = {}
                    if "test_metrics" in model_results:
                        metrics = model_results["test_metrics"]
                    elif "metrics" in model_results:
                        metrics = model_results["metrics"]

                    if metrics:
                        html_content += f"""
                        <tr>
                            <td>{model_name}</td>
                            <td>{metrics.get('mae', 'N/A'):.2f}</td>
                            <td>{metrics.get('rmse', 'N/A'):.2f}</td>
                            <td>{metrics.get('mape', 'N/A'):.4f}</td>
                        </tr>
                        """

                html_content += """
                    </table>
                </div>
                """

            # Close HTML
            html_content += """
            </body>
            </html>
            """

            return html_content

        elif format == 'csv':
            # For CSV, just export the statistics
            stats_df = pd.DataFrame()

            for col in numeric_cols:
                col_stats = pd.Series({
                    'mean': self.merged_df[col].mean(),
                    'median': self.merged_df[col].median(),
                    'std': self.merged_df[col].std(),
                    'min': self.merged_df[col].min(),
                    'max': self.merged_df[col].max()
                }, name=col)

                stats_df = pd.concat([stats_df, col_stats], axis=1)

            return stats_df.to_csv()

        else:
            print(f"Unsupported format: {format}")
            return None

    def _analyze_customer_demographics(self, save_path=None):
        """Analyze customer demographics and their impact on sales"""
        df = self.merged_df

        # Identify sales column
        if 'Sales' in df.columns:
            sales_col = 'Sales'
        else:
            potential_sales_cols = [col for col in df.columns if 'sale' in col.lower()]
            if potential_sales_cols:
                sales_col = potential_sales_cols[0]
            else:
                print("Cannot analyze customer demographics: No sales column identified")
                return

        # Gender analysis if available
        if 'Customer_Gender' in df.columns:
            gender_sales = df.groupby('Customer_Gender')[sales_col].agg(['sum', 'mean', 'count']).reset_index()

            # Create bar chart for gender comparison
            fig = px.bar(gender_sales, x='Customer_Gender', y='sum',
                         title='Total Sales by Customer Gender',
                         color='Customer_Gender')

            fig.update_layout(
                xaxis_title='Gender',
                yaxis_title=f'Total {sales_col}'
            )

            # Store the visualization
            self.visualizations['sales_by_gender'] = fig

            if save_path:
                fig.write_html(os.path.join(save_path, 'sales_by_gender.html'))

        # Age analysis if available
        if 'Customer_Age' in df.columns:
            # Create age groups
            df['Age_Group'] = pd.cut(df['Customer_Age'],
                                     bins=[0, 25, 35, 45, 55, 100],
                                     labels=['18-25', '26-35', '36-45', '46-55', '56+'])

            age_sales = df.groupby('Age_Group')[sales_col].agg(['sum', 'mean', 'count']).reset_index()

            # Create bar chart for age group comparison
            fig = px.bar(age_sales, x='Age_Group', y='sum',
                         title='Total Sales by Customer Age Group',
                         color='Age_Group')

            fig.update_layout(
                xaxis_title='Age Group',
                yaxis_title=f'Total {sales_col}',
                xaxis={'categoryorder':'array', 'categoryarray':['18-25', '26-35', '36-45', '46-55', '56+']}
            )

            # Store the visualization
            self.visualizations['sales_by_age_group'] = fig

            if save_path:
                fig.write_html(os.path.join(save_path, 'sales_by_age_group.html'))

            # If we have customer satisfaction data, analyze it by demographics
            if 'Customer_Satisfaction' in df.columns:
                # Create a heatmap of satisfaction by age group and gender
                if 'Customer_Gender' in df.columns:
                    satisfaction_pivot = df.groupby(['Age_Group', 'Customer_Gender'])['Customer_Satisfaction'].mean().reset_index()
                    satisfaction_pivot = satisfaction_pivot.pivot(index='Age_Group', columns='Customer_Gender', values='Customer_Satisfaction')

                    fig = px.imshow(satisfaction_pivot,
                                    text_auto='.2f',
                                    aspect='auto',
                                    title='Average Customer Satisfaction by Age Group and Gender')

                    fig.update_layout(
                        xaxis_title='Gender',
                        yaxis_title='Age Group'
                    )

                    # Store the visualization
                    self.visualizations['satisfaction_heatmap'] = fig

                    if save_path:
                        fig.write_html(os.path.join(save_path, 'satisfaction_heatmap.html'))

    def time_series_decomposition(self, store_id=None, dept_id=None, save_path=None):
        """
        Decompose time series data into trend, seasonality, and residual components

        Parameters:
        -----------
        store_id : int, optional
            Store ID to analyze
        dept_id : int, optional
            Department ID to analyze
        save_path : str, optional
            Path to save the plots
        """
        df = self.merged_df

        # Identify sales column
        if 'Sales' in df.columns:
            sales_col = 'Sales'
        else:
            potential_sales_cols = [col for col in df.columns if 'sale' in col.lower()]
            if potential_sales_cols:
                sales_col = potential_sales_cols[0]
            else:
                print("Cannot perform time series decomposition: No sales column identified")
                return

        # Filter data if store and department are specified
        filtered_df = df.copy()
        if store_id is not None and 'Store' in df.columns:
            filtered_df = filtered_df[filtered_df['Store'] == store_id]

        if dept_id is not None and 'Dept' in df.columns:
            filtered_df = filtered_df[filtered_df['Dept'] == dept_id]

        # Ensure we have a date column
        if 'Date' not in filtered_df.columns:
            print("Cannot perform time series decomposition: No Date column found")
            return

        # Aggregate sales by date
        sales_by_date = filtered_df.groupby('Date')[sales_col].sum().reset_index()
        sales_by_date = sales_by_date.set_index('Date')

        # Check if we have enough data points
        if len(sales_by_date) < 10:
            print("Not enough data points for decomposition")
            return

        # Perform time series decomposition
        try:
            # Determine the period (frequency) of the data
            # For this example, assume monthly data (period=12) or weekly data (period=52)
            if len(sales_by_date) >= 52:
                period = 52  # weekly data
            else:
                period = 12  # monthly data

            decomposition = seasonal_decompose(sales_by_date[sales_col], model='additive', period=period)

            trend = decomposition.trend
            seasonal = decomposition.seasonal
            residual = decomposition.resid

            # Create interactive decomposition plot with Plotly
            fig = make_subplots(rows=4, cols=1,
                                subplot_titles=('Original', 'Trend', 'Seasonality', 'Residuals'))

            fig.add_trace(go.Scatter(x=sales_by_date.index, y=sales_by_date[sales_col],
                                     mode='lines', name='Original'), row=1, col=1)

            fig.add_trace(go.Scatter(x=sales_by_date.index, y=trend,
                                     mode='lines', name='Trend'), row=2, col=1)

            fig.add_trace(go.Scatter(x=sales_by_date.index, y=seasonal,
                                     mode='lines', name='Seasonality'), row=3, col=1)

            fig.add_trace(go.Scatter(x=sales_by_date.index, y=residual,
                                     mode='lines', name='Residuals'), row=4, col=1)

            fig.update_layout(height=800, title_text="Time Series Decomposition")

            # Store the visualization
            self.visualizations['time_series_decomposition'] = fig

            if save_path:
                fig.write_html(os.path.join(save_path, 'time_series_decomposition.html'))

            return decomposition

        except Exception as e:
            print(f"Error in time series decomposition: {e}")
            return None

    def train_linear_regression(self, target=None, features=None, test_size=0.2):
        """
        Train a linear regression model to predict sales

        Parameters:
        -----------
        target : str, optional
            Target column name (default will try to identify sales column)
        features : list, optional
            List of feature column names (default will use available features)
        test_size : float
            Proportion of data to use for testing

        Returns:
        --------
        dict
            Dictionary with model, predictions, and evaluation metrics
        """
        df = self.merged_df

        # Identify target column if not specified
        if target is None:
            if 'Sales' in df.columns:
                target = 'Sales'
            else:
                potential_sales_cols = [col for col in df.columns if 'sale' in col.lower()]
                if potential_sales_cols:
                    target = potential_sales_cols[0]
                else:
                    print("Cannot train model: No sales target column identified")
                    return None

        # Identify feature columns if not specified
        if features is None:
            # Exclude non-feature columns
            exclude_cols = [target, 'Date', 'Region', 'Product', 'Store', 'Dept']

            # Include only numeric columns that aren't in the exclude list
            features = [col for col in df.select_dtypes(include=['float64', 'int64']).columns
                        if col not in exclude_cols]

            # Add derived features if they exist
            if 'Year' in df.columns:
                features.append('Year')
            if 'Month' in df.columns:
                features.append('Month')
            if 'Quarter' in df.columns:
                features.append('Quarter')
            if 'DayOfWeek' in df.columns:
                features.append('DayOfWeek')

            # Add categorical columns as one-hot encoded features
            if 'Region' in df.columns:
                region_dummies = pd.get_dummies(df['Region'], prefix='Region')
                df = pd.concat([df, region_dummies], axis=1)
                features.extend(region_dummies.columns)

            if 'Product' in df.columns:
                product_dummies = pd.get_dummies(df['Product'], prefix='Product')
                df = pd.concat([df, product_dummies], axis=1)
                features.extend(product_dummies.columns)

            if 'Customer_Gender' in df.columns:
                gender_dummies = pd.get_dummies(df['Customer_Gender'], prefix='Gender')
                df = pd.concat([df, gender_dummies], axis=1)
                features.extend(gender_dummies.columns)

        print(f"Training linear regression model...")
        print(f"Target: {target}")
        print(f"Features: {features}")

        # Prepare data
        X = df[features].copy()
        y = df[target]

        # Handle missing values in features
        for col in X.columns:
            if X[col].isna().sum() > 0:
                X[col] = X[col].fillna(X[col].median())

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=RANDOM_SEED)

        # Train linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Make predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Evaluate model
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_rmse = np.sqrt(train_mse)
        train_mape = mean_absolute_percentage_error(y_train, y_train_pred)

        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_rmse = np.sqrt(test_mse)
        test_mape = mean_absolute_percentage_error(y_test, y_test_pred)

        # Print evaluation metrics
        print("\nModel Evaluation:")
        print(f"Training MAE: {train_mae:.2f}")
        print(f"Training RMSE: {train_rmse:.2f}")
        print(f"Training MAPE: {train_mape:.4f}")
        print(f"Test MAE: {test_mae:.2f}")
        print(f"Test RMSE: {test_rmse:.2f}")
        print(f"Test MAPE: {test_mape:.4f}")

        # Feature importance
        feature_importance = pd.DataFrame({
            'Feature': features,
            'Coefficient': model.coef_
        })
        feature_importance = feature_importance.sort_values('Coefficient', ascending=False)

        print("\nFeature Importance:")
        print(feature_importance)

        # Create visualizations for model evaluation
        # Actual vs. Predicted scatter plot
        fig_scatter = px.scatter(x=y_test, y=y_test_pred,
                                 labels={'x': 'Actual', 'y': 'Predicted'},
                                 title='Actual vs. Predicted Values')

        # Add perfect prediction line
        min_val = min(y_test.min(), y_test_pred.min())
        max_val = max(y_test.max(), y_test_pred.max())
        fig_scatter.add_scatter(x=[min_val, max_val], y=[min_val, max_val],
                                mode='lines', name='Perfect Prediction',
                                line=dict(color='red', dash='dash'))

        # Feature importance bar chart
        fig_importance = px.bar(feature_importance.head(10),
                                x='Coefficient', y='Feature',
                                orientation='h',
                                title='Top 10 Feature Importance')

        # Store visualizations
        self.visualizations['actual_vs_predicted'] = fig_scatter
        self.visualizations['feature_importance'] = fig_importance

        # Store model and results
        model_results = {
            'model': model,
            'features': features,
            'feature_importance': feature_importance,
            'train_pred': y_train_pred,
            'test_pred': y_test_pred,
            'train_metrics': {
                'mae': train_mae,
                'mse': train_mse,
                'rmse': train_rmse,
                'mape': train_mape
            },
            'test_metrics': {
                'mae': test_mae,
                'mse': test_mse,
                'rmse': test_rmse,
                'mape': test_mape
            },
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }

        # Save this model
        self.models['linear_regression'] = model_results

        return model_results

    def train_arima_model(self, order=(1,1,1), seasonal_order=None):
        """
        Train an ARIMA model for time series forecasting

        Parameters:
        -----------
        order : tuple
            (p, d, q) order for ARIMA model
        seasonal_order : tuple, optional
            (P, D, Q, S) seasonal order for ARIMA model

        Returns:
        --------
        dict
            Dictionary with model and results
        """
        df = self.merged_df

        # Identify sales column
        if 'Sales' in df.columns:
            sales_col = 'Sales'
        else:
            potential_sales_cols = [col for col in df.columns if 'sale' in col.lower()]
            if potential_sales_cols:
                sales_col = potential_sales_cols[0]
            else:
                print("Cannot train ARIMA model: No sales column identified")
                return None

        # Ensure we have a date column
        if 'Date' not in df.columns:
            print("Cannot train ARIMA model: No Date column found")
            return None

        # Aggregate sales by date
        sales_by_date = df.groupby('Date')[sales_col].sum().reset_index()
        sales_by_date = sales_by_date.set_index('Date')

        # Split data into train and test
        train_size = int(len(sales_by_date) * 0.8)
        train_data = sales_by_date.iloc[:train_size]
        test_data = sales_by_date.iloc[train_size:]

        print(f"Training ARIMA model with order={order}")
        print(f"Training data: {train_data.shape}, Test data: {test_data.shape}")

        try:
            # Train ARIMA model
            if seasonal_order:
                # Use SARIMA model if seasonal_order is provided
                model = sm.tsa.statespace.SARIMAX(
                    train_data[sales_col],
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                model_name = "SARIMA"
            else:
                # Use ARIMA model
                model = ARIMA(train_data[sales_col], order=order)
                model_name = "ARIMA"

            model_fit = model.fit()

            # Make predictions for test period
            predictions = model_fit.forecast(steps=len(test_data))

            # Evaluate model
            mae = mean_absolute_error(test_data[sales_col], predictions)
            mse = mean_squared_error(test_data[sales_col], predictions)
            rmse = np.sqrt(mse)
            mape = mean_absolute_percentage_error(test_data[sales_col], predictions)

            print(f"\n{model_name} Model Evaluation:")
            print(f"MAE: {mae:.2f}")
            print(f"RMSE: {rmse:.2f}")
            print(f"MAPE: {mape:.4f}")

            # Create visualization
            fig = go.Figure()

            # Add training data
            fig.add_trace(go.Scatter(
                x=train_data.index,
                y=train_data[sales_col],
                mode='lines',
                name='Training Data'
            ))

            # Add test data
            fig.add_trace(go.Scatter(
                x=test_data.index,
                y=test_data[sales_col],
                mode='lines',
                name='Actual Test Data'
            ))

            # Add predictions
            fig.add_trace(go.Scatter(
                x=test_data.index,
                y=predictions,
                mode='lines',
                name='Predictions',
                line=dict(color='red')
            ))

            fig.update_layout(
                title=f'{model_name} Model Forecast',
                xaxis_title='Date',
                yaxis_title=sales_col,
                legend_title='Legend',
                hovermode='x unified'
            )

            # Store the visualization
            self.visualizations[f'{model_name.lower()}_forecast'] = fig

            # Store model results
            model_key = model_name.lower()

            model_results = {
                'model': model_fit,
                'train_data': train_data,
                'test_data': test_data,
                'predictions': predictions,
                'metrics': {
                    'mae': mae,
                    'mse': mse,
                    'rmse': rmse,
                    'mape': mape
                }
            }

            self.models[model_key] = model_results

            return model_results

        except Exception as e:
            print(f"Error in ARIMA model training: {e}")
            return None

    def forecast_future_sales(self, model_key='linear_regression', periods=4):
        """
        Forecast future sales

        Parameters:
        -----------
        model_key : str
            Key of the model in self.models
        periods : int
            Number of periods to forecast

        Returns:
        --------
        pd.DataFrame
            DataFrame with forecasted sales
        """
        if model_key not in self.models:
            print(f"Model '{model_key}' not found")
            return None

        df = self.merged_df
        model_results = self.models[model_key]

        # Check if it's a time series model (ARIMA/SARIMA)
        if 'arima' in model_key or 'sarima' in model_key:
            # Simple forecast for time series model
            forecast = model_results['model'].forecast(steps=periods)

            # Create forecast dataframe
            last_date = model_results['test_data'].index[-1]

            # Determine frequency (daily, weekly, monthly)
            if len(model_results['train_data']) > 300:  # Likely daily data
                freq = 'D'
            elif len(model_results['train_data']) > 70:  # Likely weekly data
                freq = 'W'
            else:  # Likely monthly data
                freq = 'M'

            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods, freq=freq)

            forecast_df = pd.DataFrame({
                'Date': future_dates,
                'Forecast': forecast
            })

            # Create visualization
            fig = go.Figure()

            # Add historical data - FIX HERE
            train_data_series = pd.Series(
                data=model_results['train_data'].iloc[:, 0].values,
                index=model_results['train_data'].index
            )
            test_data_series = pd.Series(
                data=model_results['test_data'].iloc[:, 0].values,
                index=model_results['test_data'].index
            )
            historical_series = pd.concat([train_data_series, test_data_series])

            # Add historical data
            fig.add_trace(go.Scatter(
                x=historical_series.index,
                y=historical_series.values,
                mode='lines',
                name='Historical Data'
            ))

            # Add forecast
            fig.add_trace(go.Scatter(
                x=forecast_df['Date'],
                y=forecast_df['Forecast'],
                mode='lines+markers',
                name='Forecast',
                line=dict(color='red')
            ))

            fig.update_layout(
                title='Sales Forecast',
                xaxis_title='Date',
                yaxis_title='Sales',
                legend_title='Legend',
                hovermode='x unified'
            )

            # Store the visualization
            self.visualizations['forecast'] = fig

            return forecast_df

        else:
            # Linear regression forecast - more complex as we need to create future feature values
            if 'Date' not in df.columns:
                print("Cannot forecast: No Date column found")
                return None

            # Get the latest data point
            latest_date = df['Date'].max()

            # Determine frequency (daily, weekly, monthly)
            date_diffs = df['Date'].sort_values().diff().dropna()
            median_diff = date_diffs.median().days

            if median_diff < 3:  # Daily data
                future_dates = [latest_date + timedelta(days=i+1) for i in range(periods)]
            elif median_diff < 10:  # Weekly data
                future_dates = [latest_date + timedelta(weeks=i+1) for i in range(periods)]
            else:  # Monthly data
                future_dates = [latest_date + pd.DateOffset(months=i+1) for i in range(periods)]

            # Create future data points
            future_data = []

            # Get the features used in the model
            model_features = model_results['features']

            for future_date in future_dates:
                # Start with most recent data point as a template
                base_data = df.iloc[-1:].copy()

                # Update date
                base_data['Date'] = future_date

                # Update date-derived features
                if 'Year' in model_features:
                    base_data['Year'] = future_date.year
                if 'Month' in model_features:
                    base_data['Month'] = future_date.month
                if 'Week' in model_features:
                    base_data['Week'] = future_date.isocalendar()[1]
                if 'DayOfWeek' in model_features:
                    base_data['DayOfWeek'] = future_date.dayofweek
                if 'Quarter' in model_features:
                    base_data['Quarter'] = future_date.quarter
                if 'YearMonth' in model_features and 'YearMonth' in base_data.columns:
                    base_data['YearMonth'] = future_date.strftime('%Y-%m')
                if 'MonthDay' in model_features and 'MonthDay' in base_data.columns:
                    base_data['MonthDay'] = future_date.strftime('%m-%d')

                # Add seasonal features
                if 'SinWeek' in model_features:
                    week = future_date.isocalendar()[1]
                    base_data['SinWeek'] = np.sin(2 * np.pi * week / 52)
                if 'CosWeek' in model_features:
                    week = future_date.isocalendar()[1]
                    base_data['CosWeek'] = np.cos(2 * np.pi * week / 52)

                future_data.append(base_data)

            future_df = pd.concat(future_data, ignore_index=True)

            # Extract features for prediction - ENSURE ALL FEATURES EXIST
            # If any features are missing, create them with default values
            for feature in model_features:
                if feature not in future_df.columns:
                    # For categorical features encoded as dummies
                    if feature.startswith(('Region_', 'Product_', 'Gender_')):
                        # If it's in the original data, keep the value, otherwise default to 0
                        if feature in df.columns:
                            future_df[feature] = df.iloc[-1][feature]
                        else:
                            future_df[feature] = 0
                    else:
                        # For other features, use median from training data
                        if feature in model_results['X_train'].columns:
                            future_df[feature] = model_results['X_train'][feature].median()
                        else:
                            future_df[feature] = 0
                            print(f"Warning: Feature {feature} not found in training data, using 0")

            # Now ensure we have all required features in correct order
            X_future = future_df[model_features].copy()

            # Handle any missing values
            for col in X_future.columns:
                if X_future[col].isna().sum() > 0:
                    X_future[col] = X_future[col].fillna(X_future[col].median() if len(X_future) > 0 else 0)

            # Make predictions
            predictions = model_results['model'].predict(X_future)

            # Add predictions to future dataframe
            future_df['Forecast'] = predictions

            # Create a results dataframe
            result_df = future_df[['Date', 'Forecast']].copy()

            if 'Region' in future_df.columns:
                result_df['Region'] = future_df['Region']

            if 'Product' in future_df.columns:
                result_df['Product'] = future_df['Product']

            # Create visualization
            # Get historical sales data
            sales_col = 'Sales' if 'Sales' in df.columns else df.columns[1]  # Fallback to second column
            historical = df[['Date', sales_col]].copy()

            fig = go.Figure()

            # Add historical data
            fig.add_trace(go.Scatter(
                x=historical['Date'],
                y=historical[sales_col],
                mode='lines',
                name='Historical Sales'
            ))

            # Add forecast
            fig.add_trace(go.Scatter(
                x=result_df['Date'],
                y=result_df['Forecast'],
                mode='lines+markers',
                name='Forecast',
                line=dict(color='red')
            ))

            fig.update_layout(
                title='Sales Forecast',
                xaxis_title='Date',
                yaxis_title='Sales',
                legend_title='Legend',
                hovermode='x unified'
            )

            # Store the visualization
            self.visualizations['forecast'] = fig

            return result_df

    def analyze_seasonal_patterns(self, seasonal_col='Month', save_path=None):
        """
        Analyze seasonal patterns in sales

        Parameters:
        -----------
        seasonal_col : str
            Column to use for seasonal analysis (Month, Quarter, Week, etc.)
        save_path : str, optional
            Path to save the plots

        Returns:
        --------
        pd.DataFrame
            DataFrame with seasonal statistics
        """
        df = self.merged_df.copy()

        # Check if we have the seasonal column
        if seasonal_col not in df.columns:
            # Try to derive from Date if we have it
            if 'Date' in df.columns:
                if seasonal_col == 'Month':
                    df['Month'] = df['Date'].dt.month
                elif seasonal_col == 'Quarter':
                    df['Quarter'] = df['Date'].dt.quarter
                elif seasonal_col == 'Week':
                    df['Week'] = df['Date'].dt.isocalendar().week
                else:
                    print(f"Cannot derive {seasonal_col} from Date")
                    return None
            else:
                print(f"Cannot analyze seasonal patterns: No {seasonal_col} column found")
                return None

        # Identify sales column
        if 'Sales' in df.columns:
            sales_col = 'Sales'
        else:
            potential_sales_cols = [col for col in df.columns if 'sale' in col.lower()]
            if potential_sales_cols:
                sales_col = potential_sales_cols[0]
            else:
                print("Cannot analyze seasonal patterns: No sales column identified")
                return None

        print(f"Analyzing seasonal patterns by {seasonal_col}...")

        # Group data by seasonal column
        seasonal_stats = df.groupby(seasonal_col)[sales_col].agg(['mean', 'median', 'std', 'count']).reset_index()

        # Create labels for months if analyzing by month
        if seasonal_col == 'Month':
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            seasonal_stats['MonthName'] = seasonal_stats['Month'].apply(lambda x: month_names[x-1])

            # Create seasonal plot
            fig = px.bar(seasonal_stats, x='MonthName', y='mean',
                         error_y='std',
                         title=f'Average Sales by Month',
                         labels={'mean': f'Average {sales_col}', 'MonthName': 'Month'})

            # Ensure correct month order
            fig.update_xaxes(categoryorder='array',
                             categoryarray=month_names)

        else:
            # Create seasonal plot for quarters or weeks
            fig = px.bar(seasonal_stats, x=seasonal_col, y='mean',
                         error_y='std',
                         title=f'Average Sales by {seasonal_col}',
                         labels={'mean': f'Average {sales_col}'})

        fig.update_layout(
            xaxis_title=seasonal_col,
            yaxis_title=f'Average {sales_col}',
            hovermode='closest'
        )

        # Store the visualization
        self.visualizations[f'seasonal_{seasonal_col}'] = fig

        if save_path:
            fig.write_html(os.path.join(save_path, f'seasonal_{seasonal_col}.html'))

        return seasonal_stats