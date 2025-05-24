import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from prophet import Prophet
from scipy.interpolate import make_interp_spline
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime
import requests
import io

# Check for openpyxl dependency
try:
    import openpyxl
except ImportError:
    st.error("Thư viện 'openpyxl' không được cài đặt. Vui lòng cài đặt bằng lệnh: `pip install openpyxl`.")
    st.stop()

# Set Streamlit page configuration with a softer layout
st.set_page_config(page_title="Business Dashboard", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for softer theme and sidebar styling
st.markdown("""
    <style>
    .main {background-color: #2e2e2e; color: #d3d3d3;}
    .sidebar .sidebar-content {background-color: #2e2e2e;}
    .stButton>button {
        background-color: #4a7f8e;
        color: #d3d3d3;
        border-radius: 5px;
        width: 100%;
        padding: 10px;
        margin-bottom: 10px;
    }
    .stButton>button:hover {
        background-color: #3a6a78;
        color: #d3d3d3;
    }
    h1, h2, h3, p, div {color: #d3d3d3;}
    </style>
""", unsafe_allow_html=True)

# Load data from Google Drive
@st.cache_data
def load_data():
    try:
        file_id = '1hkZZ2ks60wbMXfEeiJsxrCihpve5tpNA'
        url = f'https://docs.google.com/spreadsheets/d/{file_id}/export?format=xlsx'
        response = requests.get(url)
        response.raise_for_status()
        df = pd.read_excel(io.BytesIO(response.content), sheet_name='Sheet1', engine='openpyxl')
        return df
    except Exception as e:
        st.error(f"Không thể tải dữ liệu: {str(e)}. Vui lòng đảm bảo link Google Drive được chia sẻ công khai với quyền 'Anyone with the link'.")
        return pd.DataFrame()

df_final = load_data()
if df_final.empty:
    st.error("Không thể tải dữ liệu. Vui lòng kiểm tra link Google Drive hoặc cấu trúc dữ liệu.")
    st.stop()
df_cop = df_final.copy()

# Sidebar for tab navigation
st.sidebar.title("Điều hướng")
tab_selection = st.sidebar.radio("Chọn tab:", ["Tổng quan Doanh nghiệp", "Dự báo Prophet"], label_visibility="collapsed")

# Tab 1: Business Overview
if tab_selection == "Tổng quan Doanh nghiệp":
    st.title("Tổng quan Doanh nghiệp")
    
    # Calculate KPIs
    try:
        revenue = df_final['Tổng số tiền người mua thanh toán'].sum()
        total_cost = (df_final['Phí cố định'].sum() + 
                      df_final['Phí Dịch Vụ'].sum() + 
                      df_final['Phí thanh toán'].sum())
        shipping_cost = df_final['Phí vận chuyển mà người mua trả'].sum()
        profit = revenue - total_cost - shipping_cost
        total_sales = df_final['Số lượng'].sum()
    except KeyError as e:
        st.error(f"Lỗi: Cột {str(e)} không tồn tại trong dữ liệu.")
        st.stop()

    metrics = {
        'Revenue': revenue,
        'Cost': total_cost,
        'Profit': profit,
        'Sales': total_sales
    }

    # Create KPI plot
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), facecolor='#2e2e2e')
    kpi_labels = ['Revenue', 'Cost', 'Profit', 'Sales']
    kpi_titles = ['Doanh thu (VND)', 'Chi phí (VND)', 'Lợi nhuận (VND)', 'Số lượng bán']
    kpi_colors = ['#4a7f8e', '#6a8a95', '#8a959c', '#a5a5a5']

    for ax, label, title, color in zip(axes.flatten(), kpi_labels, kpi_titles, kpi_colors):
        value = metrics[label]
        display_value = f"{value:,.0f}" if label != 'Sales' else f"{int(value)}"
        raw_value = value
        ax.set_facecolor(color)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('#d3d3d3')
            spine.set_linewidth(0.5)
        ax.text(0.5, 0.75, f"{title}", ha='center', va='center', color='#d3d3d3', fontsize=14, fontweight='bold')
        ax.text(0.5, 0.45, display_value, ha='center', va='center', color='#d3d3d3', fontsize=20)
        ax.text(0.5, 0.15, f"▲ {raw_value:,.0f}", ha='center', va='center', color='#a5d6a7', fontsize=12)

    fig.text(0.5, 0.95, 'Tổng Quan KPI', ha='center', va='center', color='#d3d3d3', fontsize=18, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    st.pyplot(fig)

    # Yearly revenue and profit
    try:
        df_final['Năm'] = pd.to_datetime(df_final['Ngày đặt hàng']).dt.year
        df_final = df_final.fillna(0)
        df_yearly = df_final.groupby('Năm').agg({
            'Tổng số tiền người mua thanh toán': 'sum',
            'Phí cố định': 'sum',
            'Phí Dịch Vụ': 'sum',
            'Phí thanh toán': 'sum',
            'Phí vận chuyển mà người mua trả': 'sum'
        }).reset_index()
        df_yearly['Lợi nhuận'] = (df_yearly['Tổng số tiền người mua thanh toán'] -
                                  df_yearly['Phí cố định'] -
                                  df_yearly['Phí Dịch Vụ'] -
                                  df_yearly['Phí thanh toán'] -
                                  df_yearly['Phí vận chuyển mà người mua trả'])
    except KeyError as e:
        st.error(f"Lỗi: Cột {str(e)} không tồn tại trong dữ liệu.")
        st.stop()

    fig_yearly = px.bar(df_yearly, x='Năm', y=['Tổng số tiền người mua thanh toán', 'Lợi nhuận'],
                        title='Doanh thu và Lợi nhuận theo Năm',
                        labels={'value': 'Giá trị (VND)', 'variable': 'Chỉ số'},
                        barmode='group',
                        color_discrete_map={'Tổng số tiền người mua thanh toán': '#4a7f8e', 'Lợi nhuận': '#6a8a95'},
                        text_auto='.2s')
    fig_yearly.update_layout(
        xaxis_title='Năm',
        yaxis_title='Giá trị (VND)',
        legend_title='Chỉ số',
        plot_bgcolor='#2e2e2e',
        paper_bgcolor='#2e2e2e',
        font_color='#d3d3d3',
        height=600,
        width=800,
        bargap=0.2
    )
    fig_yearly.update_traces(textposition='auto', textfont=dict(size=12))
    st.plotly_chart(fig_yearly, use_container_width=True)

# Tab 2: Prophet Forecasting
else:
    st.title("Dự báo Doanh thu với Prophet")

    # Prepare data for Prophet
    try:
        df = df_cop[['Ngày đặt hàng', 'Tổng số tiền người mua thanh toán']].rename(columns={'Ngày đặt hàng': 'ds', 'Tổng số tiền người mua thanh toán': 'y'})
        df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
        df = df.dropna(subset=['ds'])
        daily_data = df.groupby('ds')['y'].sum().reset_index()
        daily_data = daily_data[daily_data['y'] >= 0]
        daily_data = daily_data[daily_data['y'] < 1e8]
        daily_data['y_smooth'] = daily_data['y'].rolling(window=7, center=True, min_periods=1).mean()
        daily_data['y_million'] = daily_data['y'] / 1_000_000
        daily_data['y_smooth_million'] = daily_data['y_smooth'] / 1_000_000
        daily_data['ds_numeric'] = daily_data['ds'].apply(lambda x: x.timestamp())
        daily_data = daily_data.sort_values('ds_numeric')
    except KeyError as e:
        st.error(f"Lỗi: Cột {str(e)} không tồn tại trong dữ liệu.")
        st.stop()

    # Smooth actual data
    x = daily_data['ds_numeric']
    y = daily_data['y_smooth_million']
    x_smooth = np.linspace(x.min(), x.max(), 300)
    spl = make_interp_spline(x, y, k=3)
    y_smooth = spl(x_smooth)
    ds_smooth = pd.to_datetime(x_smooth, unit='s', origin='unix')

    # Train Prophet model
    model = Prophet(
        changepoint_prior_scale=0.005,
        yearly_seasonality=20,
        weekly_seasonality=True,
        daily_seasonality=True,
        interval_width=0.8
    )
    model.fit(daily_data[['ds', 'y']])

    # Dynamic forecast period selection
    forecast_period = st.slider("Chọn số ngày dự báo (tối đa 365 ngày):", min_value=1, max_value=365, value=30)

    # Future forecast
    future = model.make_future_dataframe(periods=forecast_period, freq='D')
    future_forecast = model.predict(future)
    future_forecast['yhat_smooth'] = future_forecast['yhat'].rolling(window=7, center=True, min_periods=1).mean()
    future_forecast['yhat_lower_smooth'] = future_forecast['yhat_lower'].rolling(window=7, center=True, min_periods=1).mean()
    future_forecast['yhat_upper_smooth'] = future_forecast['yhat_upper'].rolling(window=7, center=True, min_periods=1).mean()
    future_forecast['yhat_million'] = future_forecast['yhat'] / 1_000_000
    future_forecast['yhat_lower_million'] = future_forecast['yhat_lower'] / 1_000_000
    future_forecast['yhat_upper_million'] = future_forecast['yhat_upper'] / 1_000_000
    future_forecast['yhat_smooth_million'] = future_forecast['yhat_smooth'] / 1_000_000
    future_forecast['yhat_lower_smooth_million'] = future_forecast['yhat_lower_smooth'] / 1_000_000
    future_forecast['yhat_upper_smooth_million'] = future_forecast['yhat_upper_smooth'] / 1_000_000
    future_forecast['yhat_smooth_million'] = future_forecast['yhat_smooth_million'].clip(lower=0)
    future_forecast['yhat_lower_smooth_million'] = future_forecast['yhat_lower_smooth_million'].clip(lower=0)
    future_forecast['yhat_upper_smooth_million'] = future_forecast['yhat_upper_smooth_million'].clip(lower=0)

    # Dynamic Plotly chart
    fig = px.line(future_forecast, x='ds', y='yhat_smooth_million', title='Dự báo Doanh thu với Prophet',
                  labels={'ds': 'Ngày', 'yhat_smooth_million': 'Doanh thu (Triệu VND)'},
                  color_discrete_sequence=['#4a7f8e'])
    fig.add_scatter(x=daily_data['ds'], y=daily_data['y_smooth_million'], mode='lines', name='Thực tế', line=dict(color='#6a8a95'))
    fig.add_scatter(x=future_forecast['ds'], y=future_forecast['yhat_lower_smooth_million'], mode='lines',
                    line=dict(color='rgba(74,127,142,0.2)'), name='Khoảng tin cậy (Dưới)', showlegend=False)
    fig.add_scatter(x=future_forecast['ds'], y=future_forecast['yhat_upper_smooth_million'], mode='lines',
                    fill='tonexty', line=dict(color='rgba(74,127,142,0.2)'), name='Khoảng tin cậy (Trên)')
    fig.update_layout(
        plot_bgcolor='#2e2e2e',
        paper_bgcolor='#2e2e2e',
        font_color='#d3d3d3',
        height=600,
        width=800,
        xaxis_title='Ngày',
        yaxis_title='Doanh thu (Triệu VND)',
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True)

    # Model evaluation
    past_future = daily_data[['ds']].copy()
    past_forecast = model.predict(past_future)
    eval_df = pd.merge(daily_data[['ds', 'y']], past_forecast[['ds', 'yhat']], on='ds')
    if not eval_df.empty:
        mae = mean_absolute_error(eval_df['y'], eval_df['yhat']) / 1_000_000
        rmse = np.sqrt(mean_squared_error(eval_df['y'], eval_df['yhat'])) / 1_000_000
        mape = np.mean(np.abs((eval_df['y'] - eval_df['yhat']) / eval_df['y'].replace(0, np.nan))) * 100
        r2 = r2_score(eval_df['y'], eval_df['yhat'])

        st.subheader("Đánh giá Mô hình Prophet")
        st.write(f"📊 RMSE: {rmse:.2f} triệu VND")
        st.write(f"📊 MAPE: {mape:.2f}%")
        st.write(f"📊 R² Score: {r2:.2f}")
    else:
        st.error("Không có dữ liệu để đánh giá mô hình.")

    # Calendar widget for single date prediction
    st.subheader("Dự đoán Doanh thu cho Ngày Tương Lai")
    selected_date = st.date_input("Chọn ngày để dự đoán:", min_value=daily_data['ds'].max(), max_value=daily_data['ds'].max() + pd.Timedelta(days=365))
    selected_date = pd.to_datetime(selected_date)
    
    # Predict for the selected date
    future_single = pd.DataFrame({'ds': [selected_date]})
    forecast_single = model.predict(future_single)
    predicted_value = forecast_single['yhat'].iloc[0] / 1_000_000
    predicted_lower = forecast_single['yhat_lower'].iloc[0] / 1_000_000
    predicted_upper = forecast_single['yhat_upper'].iloc[0] / 1_000_000

    st.write(f"Dự đoán doanh thu cho ngày {selected_date.strftime('%Y-%m-%d')}:")
    st.write(f"📈 Giá trị dự đoán: {predicted_value:.2f} triệu VND")
    st.write(f"📉 Khoảng tin cậy: [{predicted_lower:.2f}, {predicted_upper:.2f}] triệu VND")
