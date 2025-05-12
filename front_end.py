import streamlit.components.v1 as components
import streamlit as st
import pandas as pd
import os
import plotly.express as px
from datetime import datetime
import base64
import tempfile

# Import our RetailAnalytics class
from main import RetailAnalytics

# Set page configuration
st.set_page_config(
    page_title="InsightForge BI Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define CSS styles
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4CAF50;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #2196F3;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .card {
        padding: 1.5rem;
        border-radius: 5px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        background-color: #f9f9f9;
        border-left: 5px solid #4CAF50;
    }
    .insight-text {
        font-size: 1.1rem;
    }
    .small-text {
        font-size: 0.8rem;
        color: #666;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #2196F3;
    }
    .metric-label {
        font-size: 1rem;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analytics' not in st.session_state:
    st.session_state.analytics = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'insights' not in st.session_state:
    st.session_state.insights = {}
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "Overview"

def load_data(file, api_key=None):
    """Load data from uploaded file and initialize RetailAnalytics"""
    try:
        # Create a temporary file to store the uploaded content
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
            # Write the content to the temporary file
            tmp_file.write(file.getvalue())
            tmp_path = tmp_file.name

        # Initialize RetailAnalytics with the temporary file
        analytics = RetailAnalytics(sales_data_path=tmp_path, api_key=api_key)
        analytics.clean_data()

        # Store in session state
        st.session_state.analytics = analytics
        st.session_state.data_loaded = True

        # Remove the temporary file
        os.unlink(tmp_path)

        return True
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return False

def generate_demo_data():
    """Generate demo data for testing"""
    analytics = RetailAnalytics()
    analytics.generate_example_data(n_stores=5, n_departments=5,
                                    start_date='2022-01-01', periods=365)
    analytics.clean_data()

    st.session_state.analytics = analytics
    st.session_state.data_loaded = True
    return True

def prepare_dataframe_for_display(df):
    """Prepare a DataFrame for display by converting problematic columns."""
    display_df = df.copy()
    # Convert timestamp columns to strings
    for col in display_df.columns:
        if pd.api.types.is_datetime64_any_dtype(display_df[col]):
            display_df[col] = display_df[col].dt.strftime('%Y-%m-%d')
    return display_df

def display_data_overview():
    """Display data overview tab"""
    analytics = st.session_state.analytics
    df = analytics.merged_df

    st.markdown('<div class="section-header">Dataset Overview</div>', unsafe_allow_html=True)

    # Display basic dataset information
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Total Records</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{len(df):,}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        sales_col = 'Sales' if 'Sales' in df.columns else df.columns[1]
        st.markdown(f'<div class="metric-label">Total {sales_col}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{df[sales_col].sum():,.0f}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        if 'Date' in df.columns:
            date_range = f"{df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}"
        else:
            date_range = "Not available"
        st.markdown('<div class="metric-label">Date Range</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value" style="font-size: 1.2rem;">{date_range}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Display data preview
    st.markdown('<div class="section-header">Data Preview</div>', unsafe_allow_html=True)
    display_df = df.copy()
    if 'Date' in display_df.columns:
        # Convert timestamp to string for display
        display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
    st.dataframe(display_df.head(10))

    # Show data statistics
    st.markdown('<div class="section-header">Summary Statistics</div>', unsafe_allow_html=True)
    # Exclude Date column from statistics
    stats_df = df.select_dtypes(exclude=['datetime64']).describe()
    st.dataframe(stats_df)

    # Display key visualizations
    st.markdown('<div class="section-header">Key Metrics</div>', unsafe_allow_html=True)

    # Row 1 of visualizations
    col1, col2 = st.columns(2)

    with col1:
        # Display sales over time if available
        if 'sales_trends' in analytics.visualizations:
            st.plotly_chart(analytics.visualizations['sales_trends'], use_container_width=True)
        elif 'Date' in df.columns and ('Sales' in df.columns or 'Weekly_Sales' in df.columns):
            # Create visualization on the fly if not already exists
            sales_col = 'Sales' if 'Sales' in df.columns else 'Weekly_Sales'
            sales_by_date = df.groupby('Date')[sales_col].sum().reset_index()
            fig = px.line(sales_by_date, x='Date', y=sales_col, title='Sales Over Time')
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Display regional distribution if available
        if 'Region' in df.columns and 'Sales' in df.columns:
            region_sales = df.groupby('Region')['Sales'].sum().reset_index()
            fig = px.pie(region_sales, values='Sales', names='Region', title='Sales by Region')
            st.plotly_chart(fig, use_container_width=True)
        elif 'sales_by_region' in analytics.visualizations:
            st.plotly_chart(analytics.visualizations['sales_by_region'], use_container_width=True)

    # Row 2 of visualizations
    col1, col2 = st.columns(2)

    with col1:
        # Display product distribution if available
        if 'Product' in df.columns and 'Sales' in df.columns:
            product_sales = df.groupby('Product')['Sales'].sum().reset_index()
            fig = px.bar(product_sales, x='Product', y='Sales', title='Sales by Product')
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Display customer distribution if available
        if 'Customer_Age' in df.columns and 'Sales' in df.columns:
            df['Age_Group'] = pd.cut(df['Customer_Age'],
                                     bins=[0, 25, 35, 45, 55, 100],
                                     labels=['18-25', '26-35', '36-45', '46-55', '56+'])
            age_sales = df.groupby('Age_Group')['Sales'].sum().reset_index()
            fig = px.bar(age_sales, x='Age_Group', y='Sales', title='Sales by Age Group')
            st.plotly_chart(fig, use_container_width=True)
        elif 'sales_by_age_group' in analytics.visualizations:
            st.plotly_chart(analytics.visualizations['sales_by_age_group'], use_container_width=True)

def display_advanced_analytics():
    """Display advanced analytics tab"""
    analytics = st.session_state.analytics
    df = analytics.merged_df

    st.markdown('<div class="section-header">Advanced Analytics</div>', unsafe_allow_html=True)

    # Sidebar for analytical options
    analysis_type = st.sidebar.selectbox(
        "Select Analysis Type",
        ["Correlation Analysis", "Time Series Decomposition", "Seasonal Patterns", "Predictive Modeling"]
    )

    if analysis_type == "Correlation Analysis":
        st.markdown("### Correlation Analysis")
        st.markdown("Explore relationships between different variables in the dataset.")

        # Display correlation heatmap
        if 'correlation_heatmap' in analytics.visualizations:
            st.plotly_chart(analytics.visualizations['correlation_heatmap'], use_container_width=True)
        else:
            # Generate on the fly
            numeric_df = df.select_dtypes(include=['float64', 'int64'])
            corr_matrix = numeric_df.corr()
            fig = px.imshow(corr_matrix, text_auto='.2f', aspect='auto', title='Feature Correlation Matrix')
            st.plotly_chart(fig, use_container_width=True)

        # Feature relationship analysis
        st.markdown("### Feature Relationships")

        # Select features to analyze
        col1, col2 = st.columns(2)

        with col1:
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            x_feature = st.selectbox("Select X Feature", numeric_cols)

        with col2:
            y_feature = st.selectbox("Select Y Feature", [col for col in numeric_cols if col != x_feature])

        # Create scatter plot for selected features
        color_feature = None
        if 'Customer_Gender' in df.columns:
            color_feature = st.checkbox("Color by Gender", value=True)

        if color_feature:
            fig = px.scatter(df, x=x_feature, y=y_feature, color='Customer_Gender', trendline='ols',
                             title=f'Relationship Between {x_feature} and {y_feature}')
        else:
            fig = px.scatter(df, x=x_feature, y=y_feature, trendline='ols',
                             title=f'Relationship Between {x_feature} and {y_feature}')

        st.plotly_chart(fig, use_container_width=True)

    elif analysis_type == "Time Series Decomposition":
        st.markdown("### Time Series Decomposition")
        st.markdown("Decompose time series data into trend, seasonality, and residual components.")

        # Check if we can perform decomposition
        if 'Date' not in df.columns:
            st.warning("Cannot perform time series decomposition: No Date column found.")
        else:
            # Perform decomposition or use existing results
            if 'time_series_decomposition' in analytics.visualizations:
                st.plotly_chart(analytics.visualizations['time_series_decomposition'], use_container_width=True)
            else:
                try:
                    # Determine sales column
                    sales_col = 'Sales' if 'Sales' in df.columns else 'Weekly_Sales'

                    # Perform time series decomposition
                    decomposition = analytics.time_series_decomposition()

                    # Display the result if available
                    if 'time_series_decomposition' in analytics.visualizations:
                        st.plotly_chart(analytics.visualizations['time_series_decomposition'], use_container_width=True)
                    else:
                        st.info("Time series decomposition completed, but no visualization is available.")
                except Exception as e:
                    st.error(f"Error performing time series decomposition: {e}")

    elif analysis_type == "Seasonal Patterns":
        st.markdown("### Seasonal Patterns Analysis")
        st.markdown("Analyze how sales patterns change across different time periods.")

        # Select seasonal column
        seasonal_col = st.selectbox(
            "Select Time Period",
            ["Month", "Quarter", "DayOfWeek"],
            index=0
        )

        # Perform seasonal analysis
        try:
            seasonal_stats = analytics.analyze_seasonal_patterns(seasonal_col=seasonal_col)

            if f'seasonal_{seasonal_col}' in analytics.visualizations:
                st.plotly_chart(analytics.visualizations[f'seasonal_{seasonal_col}'], use_container_width=True)
            else:
                st.info(f"Seasonal analysis for {seasonal_col} completed, but no visualization is available.")

                # Try to create visualization on the fly
                if seasonal_stats is not None:
                    # Identify sales column
                    sales_col = 'Sales' if 'Sales' in df.columns else 'Weekly_Sales'

                    if seasonal_col == 'Month':
                        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                        seasonal_stats['MonthName'] = seasonal_stats['Month'].apply(lambda x: month_names[x-1])
                        fig = px.bar(seasonal_stats, x='MonthName', y='mean',
                                     title=f'Average Sales by Month',
                                     labels={'mean': f'Average {sales_col}'})
                        fig.update_xaxes(categoryorder='array', categoryarray=month_names)
                    else:
                        fig = px.bar(seasonal_stats, x=seasonal_col, y='mean',
                                     title=f'Average Sales by {seasonal_col}',
                                     labels={'mean': f'Average {sales_col}'})

                    st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error analyzing seasonal patterns: {e}")

    elif analysis_type == "Predictive Modeling":
        st.markdown("### Predictive Modeling")
        st.markdown("Train models to predict future sales based on historical patterns.")

        # Select model type
        model_type = st.selectbox(
            "Select Model Type",
            ["Linear Regression", "ARIMA Time Series"],
            index=0
        )

        # Create tabs for training and forecasting
        train_tab, forecast_tab = st.tabs(["Train Model", "Forecast"])

        with train_tab:
            if model_type == "Linear Regression":
                st.markdown("#### Linear Regression Model")

                # Check if model is already trained
                if 'linear_regression' in analytics.models:
                    st.success("Linear regression model already trained.")

                    # Display model metrics
                    metrics = analytics.models['linear_regression']['test_metrics']
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("MAE", f"{metrics['mae']:.2f}")
                    with col2:
                        st.metric("RMSE", f"{metrics['rmse']:.2f}")
                    with col3:
                        st.metric("MAPE", f"{metrics['mape']:.4f}")

                    # Show feature importance
                    if 'feature_importance' in analytics.visualizations:
                        st.plotly_chart(analytics.visualizations['feature_importance'], use_container_width=True)
                else:
                    # Button to train model
                    if st.button("Train Linear Regression Model"):
                        with st.spinner("Training model..."):
                            try:
                                model_results = analytics.train_linear_regression()
                                if model_results:
                                    st.success("Model trained successfully!")

                                    # Display model metrics
                                    metrics = model_results['test_metrics']
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("MAE", f"{metrics['mae']:.2f}")
                                    with col2:
                                        st.metric("RMSE", f"{metrics['rmse']:.2f}")
                                    with col3:
                                        st.metric("MAPE", f"{metrics['mape']:.4f}")

                                    # Show feature importance
                                    if 'feature_importance' in analytics.visualizations:
                                        st.plotly_chart(analytics.visualizations['feature_importance'], use_container_width=True)
                            except Exception as e:
                                st.error(f"Error training model: {e}")

            elif model_type == "ARIMA Time Series":
                st.markdown("#### ARIMA Time Series Model")

                # Check if model is already trained
                if 'arima' in analytics.models:
                    st.success("ARIMA model already trained.")

                    # Display model metrics
                    metrics = analytics.models['arima']['metrics']
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("MAE", f"{metrics['mae']:.2f}")
                    with col2:
                        st.metric("RMSE", f"{metrics['rmse']:.2f}")
                    with col3:
                        st.metric("MAPE", f"{metrics['mape']:.4f}")

                    # Show forecast visualization
                    if 'arima_forecast' in analytics.visualizations:
                        st.plotly_chart(analytics.visualizations['arima_forecast'], use_container_width=True)
                else:
                    # ARIMA parameters
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        p = st.number_input("p (AR order)", min_value=0, max_value=5, value=1)
                    with col2:
                        d = st.number_input("d (Differencing)", min_value=0, max_value=2, value=1)
                    with col3:
                        q = st.number_input("q (MA order)", min_value=0, max_value=5, value=1)

                    # Button to train model
                    if st.button("Train ARIMA Model"):
                        with st.spinner("Training model..."):
                            try:
                                model_results = analytics.train_arima_model(order=(p,d,q))
                                if model_results:
                                    st.success("ARIMA model trained successfully!")

                                    # Display model metrics
                                    metrics = model_results['metrics']
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("MAE", f"{metrics['mae']:.2f}")
                                    with col2:
                                        st.metric("RMSE", f"{metrics['rmse']:.2f}")
                                    with col3:
                                        st.metric("MAPE", f"{metrics['mape']:.4f}")

                                    # Show forecast visualization
                                    if 'arima_forecast' in analytics.visualizations:
                                        st.plotly_chart(analytics.visualizations['arima_forecast'], use_container_width=True)
                            except Exception as e:
                                st.error(f"Error training ARIMA model: {e}")

        with forecast_tab:
            st.markdown("#### Sales Forecast")

            # Check if models are available
            available_models = []
            if 'linear_regression' in analytics.models:
                available_models.append("Linear Regression")
            if 'arima' in analytics.models:
                available_models.append("ARIMA")

            if not available_models:
                st.warning("No trained models available. Please train a model first.")
            else:
                # Select model for forecasting
                forecast_model = st.selectbox("Select Model for Forecast", available_models)

                # Forecast parameters
                periods = st.slider("Forecast Periods", min_value=1, max_value=12, value=6)

                # Button to generate forecast
                if st.button("Generate Forecast"):
                    with st.spinner("Generating forecast..."):
                        try:
                            model_key = forecast_model.lower().replace(" ", "_")
                            forecast_results = analytics.forecast_future_sales(model_key=model_key, periods=periods)

                            if forecast_results is not None:
                                st.success("Forecast generated successfully!")

                                # Display forecast data
                                st.dataframe(forecast_results)

                                # Show forecast visualization
                                if 'forecast' in analytics.visualizations:
                                    st.plotly_chart(analytics.visualizations['forecast'], use_container_width=True)
                        except Exception as e:
                            st.error(f"Error generating forecast: {e}")

def display_ai_insights():
    """Display AI Insights tab"""
    analytics = st.session_state.analytics

    st.markdown('<div class="section-header">AI-Powered Business Insights</div>', unsafe_allow_html=True)
    st.markdown("""
    This section uses natural language processing to analyze your data and generate actionable business insights.
    Ask specific business questions or get general insights about your data.
    """)

    # API key input for LLM integration
    api_key = st.sidebar.text_input("Enter your OpenAI API Key (optional)", type="password")
    if api_key:
        analytics.api_key = api_key

    # Check if API key is available
    if not analytics.api_key:
        st.warning("OpenAI API key not provided. Some AI features may be limited.")

    # Input for custom query
    st.markdown("### Ask your data a business question")
    query = st.text_input("Enter your question:",
                          placeholder="E.g., What are the key trends in our sales data?")

    # Predefined queries
    st.markdown("Or select from common business questions:")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("What are the key sales trends?"):
            query = "What are the key sales trends over time and how can we capitalize on them?"

    with col2:
        if st.button("Which products/regions perform best?"):
            query = "Which products and regions are performing best and worst, and what actionable insights can we derive?"

    # Generate and display insights
    insights = None
    if query:
        if query in st.session_state.insights:
            insights = st.session_state.insights[query]
        else:
            with st.spinner(f"Analyzing data to answer: '{query}'"):
                try:
                    insights = analytics.generate_insights(query)
                    if 'error' not in insights:
                        st.session_state.insights[query] = insights
                    else:
                        st.error(f"Error generating insights: {insights['error']}")
                        insights = None
                except Exception as e:
                    st.error(f"Error: {e}")
                    insights = None

        if insights and 'error' not in insights:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f"<h3>{query}</h3>", unsafe_allow_html=True)
            st.markdown(f'<div class="insight-text">{insights["insights"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="small-text">Generated at: {datetime.fromisoformat(insights["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # Display previous insights
    if len(st.session_state.insights) > 0:
        st.markdown("### Previous Insights")

        for q, insight in st.session_state.insights.items():
            if q != query:  # Don't show duplicate of current query
                with st.expander(q):
                    st.markdown(f'<div class="insight-text">{insight["insights"]}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="small-text">Generated at: {datetime.fromisoformat(insight["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")}</div>', unsafe_allow_html=True)

def display_reporting():
    """Display Reporting tab"""
    analytics = st.session_state.analytics

    st.markdown('<div class="section-header">Reporting & Exports</div>', unsafe_allow_html=True)
    st.markdown("Generate comprehensive reports and export visualizations from your data analysis.")

    # Report generation options
    st.markdown("### Generate Business Report")

    col1, col2, col3 = st.columns(3)

    with col1:
        report_format = st.selectbox("Report Format", ["HTML", "JSON", "CSV"])

    with col2:
        include_insights = st.checkbox("Include AI Insights", value=True)

    with col3:
        include_visualizations = st.checkbox("Include Visualizations", value=True)

    # Button to generate report
    if st.button("Generate Report"):
        with st.spinner("Generating report..."):
            try:
                report = analytics.export_report(
                    format=report_format.lower(),
                    include_insights=include_insights,
                    include_visualizations=include_visualizations
                )

                if report:
                    st.success("Report generated successfully!")

                    # Download button
                    format_extension = {
                        "HTML": "html",
                        "JSON": "json",
                        "CSV": "csv"
                    }

                    filename = f"insightforge_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{format_extension[report_format]}"

                    # Convert report to downloadable format
                    b64 = base64.b64encode(report.encode()).decode()
                    href = f'<a href="data:file/{format_extension[report_format]};base64,{b64}" download="{filename}">Download {report_format} Report</a>'
                    st.markdown(href, unsafe_allow_html=True)

                    # Preview for HTML reports
                    if report_format == "HTML":
                        with st.expander("Preview Report"):
                            components.html(report, height=600, scrolling=True)
            except Exception as e:
                st.error(f"Error generating report: {e}")

    # Export individual visualizations
    st.markdown("### Export Individual Visualizations")

    if analytics.visualizations:
        viz_names = list(analytics.visualizations.keys())
        selected_viz = st.selectbox("Select Visualization", viz_names)

        if selected_viz:
            fig = analytics.get_visualization(selected_viz)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

                # Export options
                export_format = st.selectbox("Export Format", ["HTML", "PNG", "SVG", "JSON"])

                if st.button("Export Visualization"):
                    if export_format == "HTML":
                        # Export as HTML
                        html_fig = fig.to_html()
                        b64 = base64.b64encode(html_fig.encode()).decode()
                        href = f'<a href="data:text/html;base64,{b64}" download="{selected_viz}.html">Download HTML</a>'
                        st.markdown(href, unsafe_allow_html=True)
                    elif export_format == "JSON":
                        # Export as JSON
                        json_fig = fig.to_json()
                        b64 = base64.b64encode(json_fig.encode()).decode()
                        href = f'<a href="data:application/json;base64,{b64}" download="{selected_viz}.json">Download JSON</a>'
                        st.markdown(href, unsafe_allow_html=True)
                    else:
                        st.info(f"To export as {export_format}, please use the export button in the visualization toolbar.")
    else:
        st.info("No visualizations available to export.")

def main():
    """Main function to run the Streamlit app"""
    # Display header
    st.markdown('<div class="main-header">InsightForge Business Intelligence Assistant</div>', unsafe_allow_html=True)
    st.markdown("""
    Transform your business data into actionable insights with AI-powered analytics.
    Upload your data, analyze trends, get AI-powered recommendations, and generate reports.
    """)

    # Sidebar for navigation
    st.sidebar.title("Navigation")

    # Data loading section always appears in sidebar
    st.sidebar.markdown("### Data Source")

    # File uploader
    uploaded_file = st.sidebar.file_uploader("Upload CSV Data", type="csv")

    # API key input
    api_key = st.sidebar.text_input("OpenAI API Key (for AI insights)", type="password")

    # Load data buttons
    col1, col2 = st.sidebar.columns(2)

    with col1:
        if uploaded_file is not None:
            if st.sidebar.button("Load Data"):
                with st.spinner("Loading data..."):
                    success = load_data(uploaded_file, api_key)
                    if success:
                        st.sidebar.success("Data loaded successfully!")

    with col2:
        if st.sidebar.button("Use Demo Data"):
            with st.spinner("Generating demo data..."):
                success = generate_demo_data()
                if success:
                    st.sidebar.success("Demo data generated!")

    # If data is loaded, show tabs and content
    if st.session_state.data_loaded:
        # Tabs appear in sidebar after data is loaded
        selected_tab = st.sidebar.radio("Navigate",
                                        ["Overview", "Advanced Analytics", "AI Insights", "Reporting"],
                                        index=0)

        st.session_state.active_tab = selected_tab

        # Display selected tab content
        if selected_tab == "Overview":
            display_data_overview()
        elif selected_tab == "Advanced Analytics":
            display_advanced_analytics()
        elif selected_tab == "AI Insights":
            display_ai_insights()
        elif selected_tab == "Reporting":
            display_reporting()
    else:
        # If no data is loaded, show welcome screen
        st.markdown("""
        ## Welcome to InsightForge

        This tool helps you analyze your business data and uncover actionable insights:

        1. **Upload your data** using the sidebar or generate demo data
        2. **Explore dashboards** to understand your business performance
        3. **Get AI-powered insights** tailored to your specific business questions
        4. **Create reports** to share with your team

        ### Getting Started
        Upload a CSV file or click "Use Demo Data" to begin exploring.
        """)

        # Sample data format description
        with st.expander("Expected Data Format"):
            st.markdown("""
            Your CSV file should include the following columns for best results:

            - **Date**: Transaction dates in YYYY-MM-DD format
            - **Sales**: Numerical sales values
            - **Product**: Product categories or names
            - **Region**: Geographical regions
            - **Customer demographics**: Age, gender, etc. (optional)

            Example:
            ```
            Date,Product,Region,Sales,Customer_Age,Customer_Gender,Customer_Satisfaction
            2022-01-01,Widget A,North,786,26,Male,4.2
            2022-01-02,Widget B,South,850,29,Female,3.8
            ```
            """)

if __name__ == "__main__":
    main()