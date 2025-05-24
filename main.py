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
    st.error("Thư viện 'openpyxl' không được cài đặt. Vui lòng cài đặt bằng lệnh: `pip install openpyxl`")
    st.stop()

# Set Streamlit page configuration with a blue-themed layout
st.set_page_config(page_title="Business Dashboard", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for blue theme and sidebar styling
st.markdown("""
    <style>
    .main {background-color: #1a1a1a; color: white;}
    .sidebar .sidebar-content {background-color: #1a1a1a;}
    .stButton>button {
        background-color: #2a9d8f;
        color: white;
        border-radius: 5px;
        width: 100%;
        padding: 10px;
        margin-bottom: 10px;
    }
    .stButton>button:hover {
        background-color: #219d8f;
        color: white;
    }
    h1, h2, h3, p, div {color: white;}
    </style>
""", unsafe_allow_html=True)

# Load data from Google Drive
@st.cache_data
def load_data():
    try:
        # Google Drive file ID from the provided link
        file_id = '1hkZZ2ks60wbMXfEeiJsxrCihpve5tpNA'
        url = f'https://docs.google.com/spreadsheets/d/{file_id}/export?format=xlsx'
        response = requests.get(url)
        response.raise_for_status()
        df = pd.read_excel(io.BytesIO(response.content), sheet_name='Sheet1', engine='openpyxl')
        # Debug: Display column names for verification
        st.write("Tên cột trong dữ liệu:", df.columns.tolist())
        return df
    except Exception as e:
        st.error(f"Không thể tải dữ liệu: {str(e)}. Vui lòng đảm bảo link Google Drive được chia sẻ công khai với quyền 'Anyone with the link'. Hoặc tải file Excel về máy và sử dụng local file.")
        return pd.DataFrame()

df_final = load_data()
if df_final.empty:
    st.error("Không thể tải dữ liệu. Vui lòng kiểm tra link Google Drive hoặc cấu trúc dữ liệu. Để sử dụng local file, cập nhật hàm `load_data()` với đường dẫn file Excel.")
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
        st.error(f"Lỗi: Cột {str(e)} không tồn tại trong dữ liệu. Dưới đây là danh sách cột hiện có:")
        st.write(df_final.columns.tolist())
        st.error("Vui lòng cập nhật tên cột trong hàm `load_data()` để khớp với dữ liệu.")
        st.stop()

    metrics = {
        'Revenue': revenue,
        'Cost': total_cost,
        'Profit': profit,
        'Sales': total_sales
    }

    # Create KPI plot
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), facecolor='#1a1a1a')
    plt.style.use('dark_background')
    kpi_labels = ['Revenue', 'Cost', 'Profit', 'Sales']
    kpi_titles = ['Doanh thu (VND)', 'Chi phí (VND)', 'Lợi nhuận (VND)', 'Số lượng bán']
    kpi_colors = ['#2a9d8f', '#e76f51', '#f4a261', '#e9c46a']

    for ax, label, title, color in zip(axes.flatten(), kpi_labels, kpi_titles, kpi_colors):
        value = metrics[label]
        display_value = f"{value:,.0f}" if label != 'Sales' else f"{int(value)}"
        raw_value = value
        ax.set_facecolor(color)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('white')
            spine.set_linewidth(0.5)
        ax.text(0.5, 0.75, f"{title}", ha='center', va='center', color='white', fontsize=14, fontweight='bold')
        ax.text(0.5, 0.45, display_value, ha='center', va='center', color='white', fontsize=20)
        ax.text(0.5, 0.15, f"▲ {raw_value:,.0f}", ha='center', va='center', color='#00ff00', fontsize=12)

    fig.text(0.5, 0.95, 'Tổng Quan KPI', ha='center', va='center', color='white', fontsize=18, fontweight='bold')
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
        st.error(f"Lỗi: Cột {str(e)} không tồn tại trong dữ liệu. Dưới đây là danh sách cột hiện có:")
        st.write(df_final.columns.tolist())
        st.error("Vui lòng cập nhật tên cột trong hàm `load_data()` để khớp với dữ liệu.")
        st.stop()

    fig_yearly = px.bar(df_yearly, x='Năm', y=['Tổng số tiền người mua thanh toán', 'Lợi nhuận'],
                        title='Doanh thu và Lợi nhuận theo Năm',
                        labels={'value': 'Giá trị (VND)', 'variable': 'Chỉ số'},
                        barmode='group',
                        color_discrete_map={'Tổng số tiền người mua thanh toán': '#2a9d8f', 'Lợi nhuận': '#e76f51'},
                        text_auto='.2s')
    fig_yearly.update_layout(
        xaxis_title='Năm',
        yaxis_title='Giá trị (VND)',
        legend_title='Chỉ số',
        plot_bgcolor='#1a1a1a',
        paper_bgcolor='#1a1a1a',
        font_color='white',
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
        st.error(f"Lỗi: Cột {str(e)} không tồn tại trong dữ liệu. Dưới đây là danh sách cột hiện có:")
        st.write(df_cop.columns.tolist())
        st.error("Vui lòng cập nhật tên cột trong hàm `load_data()` để khớp với dữ liệu.")
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

    # Plot 1: Historical forecast
    past_future = daily_data[['ds']].copy()
    past_forecast = model.predict(past_future)
    past_forecast['yhat_smooth'] = past_forecast['yhat'].rolling(window=7, center=True, min_periods=1).mean()
    past_forecast['yhat_lower_smooth'] = past_forecast['yhat_lower'].rolling(window=7, center=True, min_periods=1).mean()
    past_forecast['yhat_upper_smooth'] = past_forecast['yhat_upper'].rolling(window=7, center=True, min_periods=1).mean()
    past_forecast['yhat_million'] = past_forecast['yhat'] / 1_000_000
    past_forecast['yhat_lower_million'] = past_forecast['yhat_lower'] / 1_000_000
    past_forecast['yhat_upper_million'] = past_forecast['yhat_upper'] / 1_000_000
    past_forecast['yhat_smooth_million'] = past_forecast['yhat_smooth'] / 1_000_000
    past_forecast['yhat_lower_smooth_million'] = past_forecast['yhat_lower_smooth'] / 1_000_000
    past_forecast['yhat_upper_smooth_million'] = past_forecast['yhat_upper_smooth'] / 1_000_000
    past_forecast['yhat_smooth_million'] = past_forecast['yhat_smooth_million'].clip(lower=0)
    past_forecast['yhat_lower_smooth_million'] = past_forecast['yhat_lower_smooth_million'].clip(lower=0)
    past_forecast['yhat_upper_smooth_million'] = past_forecast['yhat_upper_smooth_million'].clip(lower=0)

    fig1, ax1 = plt.subplots(figsize=(10, 6), facecolor='#1a1a1a')
    ax1.plot(ds_smooth, y_smooth, 'orange', label='Thực tế', linewidth=2)
    ax1.plot(past_forecast['ds'], past_forecast['yhat_smooth_million'], '#2a9d8f', label='Dự đoán', linewidth=2)
    ax1.fill_between(past_forecast['ds'], past_forecast['yhat_lower_smooth_million'], past_forecast['yhat_upper_smooth_million'],
                     color='skyblue', alpha=0.1, label='Khoảng tin cậy')
    ax1.set_title('Doanh thu hàng ngày - Prophet (Tập gốc)')
    ax1.set_xlabel('Ngày')
    ax1.set_ylabel('Doanh thu (Triệu VND)')
    ax1.set_ylim(0, max(daily_data['y_million'].max(), past_forecast['yhat_upper_smooth_million'].max()) * 1.1)
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig1)

    # Plot 2: Future forecast (12 months)
    future = model.make_future_dataframe(periods=365, freq='D')
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

    fig2, ax2 = plt.subplots(figsize=(10, 6), facecolor='#1a1a1a')
    ax2.plot(ds_smooth, y_smooth, 'orange', label='Thực tế', linewidth=2)
    ax2.plot(future_forecast['ds'][future_forecast['ds'] <= daily_data['ds'].max()],
             future_forecast['yhat_smooth_million'][future_forecast['ds'] <= daily_data['ds'].max()],
             '#2a9d8f', label='Dự đoán', linewidth=2)
    ax2.plot(future_forecast['ds'][future_forecast['ds'] > daily_data['ds'].max()],
             future_forecast['yhat_smooth_million'][future_forecast['ds'] > daily_data['ds'].max()],
             '#e76f51', label='Dự đoán tương lai', linewidth=2)
    ax2.fill_between(future_forecast['ds'][future_forecast['ds'] <= daily_data['ds'].max()],
                     future_forecast['yhat_lower_smooth_million'][future_forecast['ds'] <= daily_data['ds'].max()],
                     future_forecast['yhat_upper_smooth_million'][future_forecast['ds'] <= daily_data['ds'].max()],
                     color='skyblue', alpha=0.1, label='Khoảng tin cậy')
    ax2.fill_between(future_forecast['ds'][future_forecast['ds'] > daily_data['ds'].max()],
                     future_forecast['yhat_lower_smooth_million'][future_forecast['ds'] > daily_data['ds'].max()],
                     future_forecast['yhat_upper_smooth_million'][future_forecast['ds'] > daily_data['ds'].max()],
                     color='pink', alpha=0.1, label='Khoảng tin cậy')
    ax2.set_title('Dự đoán doanh thu 12 tháng tiếp theo - Prophet')
    ax2.set_xlabel('Ngày')
    ax2.set_ylabel('Doanh thu (Triệu VND)')
    ax2.set_ylim(0, max(daily_data['y_million'].max(), future_forecast['yhat_upper_smooth_million'].max()) * 1.1)
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig2)

    # Model evaluation
    eval_df = pd.merge(daily_data[['ds', 'y']], past_forecast[['ds', 'yhat']], on='ds')
    if not eval_df.empty:
        mae = mean_absolute_error(eval_df['y'], eval_df['yhat']) / 1_000_000
        rmse = np.sqrt(mean_squared_error(eval_df['y'], eval_df['yhat'])) / 1_000_000
        mape = np.mean(np.abs((eval_df['y'] - eval_df['yhat']) / eval_df['y'].replace(0, np.nan))) * 100
        r2 = r2_score(eval_df['y'], eval_df['yhat'])
        mean_y_million = eval_df['y'].mean() / 1_000_000
        mae_relative = (mae / mean_y_million) * 100 if mean_y_million > 0 else np.nan

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
