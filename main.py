import openai
import os
import plotly.express as px
from prophet import Prophet
import streamlit as st
import pandas as pd
from llama_index.experimental.query_engine import PandasQueryEngine

openai.api_key = st.secrets["OPENAI_API_KEY"]

# Load datasets
@st.cache_data
def load_data():
    company_file_path = 'Dataset/my_compony_data.csv'
    competitor_file_path = 'Dataset/competitor_compony_data.csv'

    company_data = pd.read_csv(company_file_path)
    competitor_data = pd.read_csv(competitor_file_path)

    # Convert ORDERDATE to datetime format
    company_data['ORDERDATE'] = pd.to_datetime(company_data['ORDERDATE'])
    competitor_data['ORDERDATE'] = pd.to_datetime(competitor_data['ORDERDATE'])
    
    return company_data, competitor_data

# Example comparative analysis function
def compare_sales(company_df, competitor_df):
    company_sales = company_df.groupby('PRODUCTLINE')['SALES'].sum()
    competitor_sales = competitor_df.groupby('PRODUCTLINE')['SALES'].sum()
    
    comparison = pd.DataFrame({
        'Company Sales': company_sales,
        'Competitor Sales': competitor_sales
    }).fillna(0)
    
    return comparison

# Define the function to get AI insights
def get_ai_insights(queries, engine):
    responses = {}
    for query in queries:
        response = engine.query(query)
        responses[query] = response
    return responses

# Define function for sales suggestions
def suggest_sales_increase(company_df, competitor_df):
    suggestions = []
    
    sales_comparison = compare_sales(company_df, competitor_df)
    underperforming = sales_comparison[sales_comparison['Company Sales'] < sales_comparison['Competitor Sales']]
    
    if not underperforming.empty:
        suggestions.append("Consider enhancing marketing strategies for product lines with lower sales compared to competitors.")
    
    high_growth = company_df.groupby('PRODUCTLINE')['SALES'].sum().sort_values(ascending=False).head(3)
    suggestions.append(f"Focus on high-growth product lines such as {', '.join(high_growth.index)}.")
    
    return suggestions

# Streamlit app
def main():
    st.title('Sales Analyser')
    
    company_data, competitor_data = load_data()
    st.subheader("here is the sample data that we use")
    st.write(company_data.head())

    st.subheader('Sales Comparison')
    sales_comparison = compare_sales(company_data, competitor_data)
    st.write(sales_comparison)
    
    st.header("> Descriptive Analysis")
    
    st.write("Sales Distribution")
    fig = px.histogram(company_data, x='ORDERDATE', y='SALES', title='Sales Distribution by Order Date', labels={'SALES': 'Sales Amount', 'ORDERDATE': 'Order Date'})
    st.plotly_chart(fig)
    
    st.write("Product Line Sales Comparison")
    
    fig = px.bar(company_data, x='PRODUCTLINE', y='SALES', 
                title='Sales by Product Line', 
                labels={'PRODUCTLINE': 'Product Line', 'SALES': 'Sales'},
                color='PRODUCTLINE',  
                text='SALES'  
                )

    fig.update_layout(
        xaxis_title='Product Line',
        yaxis_title='Sales',
        yaxis_tickprefix='$',  
        title_x=0.5,  
        margin=dict(t=50, l=25, r=25, b=25) 
    )

    fig.update_xaxes(categoryorder='total descending')
    st.plotly_chart(fig)
    
    
    st.write('Deal Size Distribution')
    fig = px.pie(company_data, names='DEALSIZE', title='Deal Size Distribution')
    st.plotly_chart(fig)

    st.write("Sales by Country")
    
    fig = px.choropleth(
        company_data,
        locations='COUNTRY',
        locationmode='country names',
        color='SALES',
        title='Sales by Country',
        labels={'SALES': 'Sales'},
        color_continuous_scale=px.colors.sequential.Plasma, 
        hover_name='COUNTRY',
        hover_data={'SALES': True}  
    )

    fig.update_layout(
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type='equirectangular' 
        ),
        title_font=dict(size=24),
        title_x=0.5,
        margin={"r":0,"t":50,"l":0,"b":0}
    )
    st.plotly_chart(fig)
    
    st.header("> Predictive Analysis")
    
    # Aggregate sales data by date
    sales_data = company_data.groupby('ORDERDATE')['SALES'].sum().reset_index()
    sales_data.columns = ['ds', 'y']
    
    model = Prophet(
        yearly_seasonality=True,  
        weekly_seasonality=True,  
        daily_seasonality=False   
    )

    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)


    # Adding holidays (example with US holidays)
    holidays = pd.DataFrame({
        'holiday': 'sales_event',
        'ds': pd.to_datetime(['2003-02-01', '2003-12-25']),
        'lower_window': 0,
        'upper_window': 1,
    })
    model.add_country_holidays(country_name='US')
    
    model.fit(sales_data)
    
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    
    st.subheader("Now here we have Forecasting for next 30 day's based on data")
    fig = px.line(forecast, x='ds', y=['yhat', 'yhat_lower', 'yhat_upper'], title='Sales Forecast', labels={'ds': 'Date', 'value': 'Sales'})
    fig.update_layout(yaxis_title='Sales', xaxis_title='Date')
    st.plotly_chart(fig)
    
    st.write("Yearly Forecasting")
    fig_yearly = px.line(forecast, x='ds', y='yearly', title='Yearly Component', labels={'ds': 'Date', 'yearly': 'Yearly'})
    st.plotly_chart(fig_yearly)

    st.header("> Prescriptive Analysis")
    query_engine = PandasQueryEngine(df=company_data, verbose=False)
    user_queries = [
        "Compare sales performance between different product lines.",
        "Which product lines have the highest sales growth?"
        ]
    
    insights = get_ai_insights(user_queries, query_engine)
    for query, response in insights.items():
        st.write(f"**Question:** {query}")
        st.write(f"**Response:** {response}")

    st.subheader('Suggestions to Increase Sales')
    sales_suggestions = suggest_sales_increase(company_data, competitor_data)
    for suggestion in sales_suggestions:
        st.write(f"- {suggestion}")

if __name__ == "__main__":
    main()
