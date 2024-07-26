# Business-Analyser

This code provides a comprehensive technical solution for comparative analysis, forecasting, and prescriptive insights in a Streamlit application. It leverages various Python libraries, including `pandas`, `plotly`, `Prophet`, and the Hugging Face's `llama_index` for natural language query processing. The primary focus is on analyzing sales data from both the company and its competitor, forecasting future sales trends, and providing actionable recommendations based on the analysis.

### Key Components and Functions

1. **Data Loading and Caching (`load_data`)**:
   - This function reads CSV files containing sales data for the company and its competitor. It uses Streamlit's `@st.cache_data` decorator to cache the loaded data, enhancing the app's performance by avoiding redundant data loads. The `ORDERDATE` column is converted to a datetime format for proper analysis.

2. **Comparative Sales Analysis (`compare_sales`)**:
   - This function compares the sales performance across different product lines for both the company and its competitor. It groups sales data by `PRODUCTLINE` and sums the sales values, returning a DataFrame with the comparison.

3. **AI Insights (`get_ai_insights`)**:
   - This function uses the Hugging Face's `PandasQueryEngine` to process natural language queries on the sales data. It generates insights by querying the data and returning responses.

4. **Sales Suggestions (`suggest_sales_increase`)**:
   - This function provides actionable suggestions to increase sales based on the comparative analysis. It identifies underperforming product lines compared to competitors and suggests focusing on high-growth product lines.

### Streamlit Application

The Streamlit app has several key sections:

1. **Title and Data Display**:
   - The app begins with a title and displays a sample of the loaded data to the user.

2. **Descriptive Analysis**:
   - Various visualizations are generated using Plotly:
     - **Sales Distribution**: A histogram of sales over order dates.
     - **Product Line Sales Comparison**: A bar chart showing sales by product line.
     - **Deal Size Distribution**: A pie chart displaying the distribution of deal sizes.
     - **Sales by Country**: A choropleth map visualizing sales data by country.

3. **Predictive Analysis**:
   - Sales forecasting is performed using Facebook's Prophet library. The data is aggregated by `ORDERDATE`, and a model is trained to predict future sales trends. The forecast includes a 30-day projection and a yearly trend analysis.

4. **Prescriptive Analysis**:
   - This section includes AI-generated insights and a comparison of sales performance across different product lines. It also provides specific suggestions for increasing sales based on the comparative analysis.

### Technologies and Libraries

- **Pandas**: For data manipulation and analysis.
- **Plotly**: For creating interactive visualizations.
- **Prophet**: For time series forecasting.
- **Streamlit**: For building the interactive web application.
- **Hugging Face `llama_index`**: For natural language processing and generating AI insights.

### Conclusion

This code provides a powerful tool for analyzing and interpreting sales data, forecasting future trends, and generating actionable insights. The integration of AI-driven queries and suggestions makes it a valuable resource for business decision-making and strategy planning.
