#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SKU Rationalization Dashboard
Анализ и классификация товаров розничной сети (~5000 SKU)
Запуск: streamlit run app.py
Docker: docker build -t sku-rationalization . && docker run -p 8501:8501 sku-rationalization
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import io

# ======================
# CONFIG & STYLING
# ======================
st.set_page_config(
    page_title="SKU Rationalization Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Кастомные стили
st.markdown("""
<style>
    .metric-card {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 0.5rem; color: white;}
    .recommendation-keep {background: #d4edda; padding: 0.25rem 0.5rem; border-radius: 0.25rem; font-weight: bold;}
    .recommendation-review {background: #fff3cd; padding: 0.25rem 0.5rem; border-radius: 0.25rem; font-weight: bold;}
    .recommendation-exit {background: #f8d7da; padding: 0.25rem 0.5rem; border-radius: 0.25rem; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

# ======================
# CACHE & DATA LOADING
# ======================
@st.cache_data
def load_sample_data(n_skus=5000):
    """Генерация демо-данных для демонстрации"""
    np.random.seed(42)
    data = {
        'sku_id': [f'SKU_{i:05d}' for i in range(1, n_skus + 1)],
        'category': np.random.choice(['Electronics', 'Food', 'Home', 'Clothing', 'Beauty'], n_skus),
        'region': np.random.choice(['Moscow', 'SPb', 'Region'], n_skus),
        'revenue_12m': np.random.lognormal(10, 2, n_skus),
        'margin_pct': np.clip(np.random.normal(25, 15, n_skus), 0, 100),
        'turnover_days': np.random.lognormal(3.5, 0.8, n_skus),
        'demand_cv': np.random.lognormal(-0.5, 0.6, n_skus),
        'promo_share_pct': np.random.beta(2, 5, n_skus) * 100,
        'basket_affinity': np.random.beta(1, 3, n_skus),
        'stock_days': np.random.lognormal(4, 1, n_skus),
        'unique_orders': np.random.poisson(200, n_skus),
    }
    df = pd.DataFrame(data)
    # Добавляем "проблемные" SKU для демонстрации
    df.loc[df.sample(frac=0.1).index, 'revenue_12m'] *= 0.1
    df.loc[df.sample(frac=0.1).index, 'margin_pct'] = np.random.uniform(0, 10, size=int(n_skus*0.1))
    return df

@st.cache_data
def compute_cluster_metrics(df, features, n_clusters_range=range(3, 8)):
    """Подбор оптимального числа кластеров по silhouette score"""
    scaler = StandardScaler()
    X = scaler.fit_transform(df[features].fillna(0))
    results = []
    for k in n_clusters_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        results.append({'n_clusters': k, 'silhouette': score})
    return pd.DataFrame(results)

@st.cache_data
def run_clustering(df, features, n_clusters=5, method='kmeans'):
    """Запуск кластеризации и добавление меток"""
    scaler = StandardScaler()
    X = scaler.fit_transform(df[features].fillna(0))
    
    if method == 'kmeans':
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    else:
        model = DBSCAN(eps=0.5, min_samples=10)
    
    labels = model.fit_predict(X)
    df_result = df.copy()
    df_result['cluster'] = labels
    
    # Бизнес-интерпретация кластеров (упрощённая)
    cluster_profiles = df_result.groupby('cluster').agg({
        'revenue_12m': 'median',
        'margin_pct': 'median', 
        'turnover_days': 'median',
        'demand_cv': 'median'
    }).round(2)
    
    def interpret_cluster(row):
        if row['margin_pct'] > 30 and row['turnover_days'] < 40:
            return '🟢 Cash Cow'
        elif row['revenue_12m'] > df_result['revenue_12m'].quantile(0.7):
            return '🔵 Traffic Driver'
        elif row['demand_cv'] > 1.0:
            return '🟡 Seasonal/Niche'
        elif row['margin_pct'] < 15 and row['turnover_days'] > 60:
            return '🔴 Candidate for Exit'
        else:
            return '🟡 Review Needed'
    
    profiles_named = cluster_profiles.apply(interpret_cluster, axis=1)
    df_result['cluster_name'] = df_result['cluster'].map(profiles_named)
    
    # Автоматические рекомендации
    def generate_recommendation(row):
        if row['margin_pct'] < 10 and row['turnover_days'] > 75 and row['revenue_12m'] < df_result['revenue_12m'].quantile(0.3):
            return 'EXIT', 'Низкая маржа + медленная оборачиваемость + низкий спрос'
        elif row['basket_affinity'] > 0.5 and row['revenue_12m'] > df_result['revenue_12m'].median():
            return 'KEEP', 'Высокая привязка к корзине — трафик-драйвер'
        elif row['demand_cv'] > 1.2:
            return 'SEASONAL', 'Высокая волатильность — настроить сезонное пополнение'
        elif row['promo_share_pct'] > 70 and row['margin_pct'] < 20:
            return 'REVIEW_PRICING', 'Продажи только в промо — пересмотреть контракт'
        else:
            return 'KEEP', 'Стабильные показатели'
    
    df_result[['recommendation', 'reason']] = df_result.apply(
        lambda row: pd.Series(generate_recommendation(row)), axis=1
    )
    
    return df_result, cluster_profiles, profiles_named

# ======================
# SIDEBAR
# ======================
st.sidebar.title("🛒 SKU Rationalization")
st.sidebar.markdown("---")

# Загрузка данных
data_source = st.sidebar.radio("Источник данных:", ["📊 Демо-данные", "📁 Загрузить CSV", "🗄️ Подключиться к БД"])

if data_source == "📊 Демо-данные":
    df_raw = load_sample_data()
    st.sidebar.success("✅ Загружены демо-данные (5000 SKU)")
    
elif data_source == "📁 Загрузить CSV":
    uploaded = st.sidebar.file_uploader("Выберите CSV-файл", type=['csv'])
    if uploaded:
        df_raw = pd.read_csv(uploaded)
        st.sidebar.success(f"✅ Загружено {len(df_raw)} строк")
    else:
        st.info("👆 Загрузите CSV-файл с метриками SKU")
        st.stop()
        
else:  # DB connection placeholder
    st.sidebar.warning("🔧 Подключение к БД требует настройки в production")
    df_raw = load_sample_data()  # fallback

# Фильтры
st.sidebar.markdown("### 🔍 Фильтры")
selected_categories = st.sidebar.multiselect("Категория:", options=df_raw['category'].unique(), default=df_raw['category'].unique())
selected_regions = st.sidebar.multiselect("Регион:", options=df_raw['region'].unique(), default=df_raw['region'].unique())

# Применение фильтров
df_filtered = df_raw[
    df_raw['category'].isin(selected_categories) & 
    df_raw['region'].isin(selected_regions)
].copy()

# ======================
# MAIN APP
# ======================
st.title("🎯 Анализ ассортимента: классификация SKU")
st.markdown(f"**Всего товаров:** {len(df_filtered):,} | **Категорий:** {df_filtered['category'].nunique()} | **Регионов:** {df_filtered['region'].nunique()}")

# KPI карточки
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Средняя маржа", f"{df_filtered['margin_pct'].median():.1f}%")
with col2:
    st.metric("Мед. оборачиваемость", f"{df_filtered['turnover_days'].median():.0f} дн.")
with col3:
    st.metric("Доля в промо", f"{df_filtered['promo_share_pct'].median():.1f}%")
with col4:
    st.metric("Кандидаты на вывод", f"{(df_filtered['recommendation']=='EXIT').sum() if 'recommendation' in df_filtered.columns else '—'}")

# Вкладки
tab1, tab2, tab3, tab4 = st.tabs(["📊 Визуализация", "🤖 Кластеризация", "📋 Таблица решений", "⚙️ Настройки"])

# ======================
# TAB 1: VISUALIZATION
# ======================
with tab1:
    st.subheader("📈 Аналитические графики")
    
    col_a, col_b = st.columns(2)
    with col_a:
        # Проверяем наличие колонки recommendation перед использованием
        hover_cols = ['sku_id']
        if 'recommendation' in df_filtered.columns:
            hover_cols.append('recommendation')
        
        fig_scatter = px.scatter(
            df_filtered.head(2000),  # ограничиваем для производительности
            x='turnover_days', y='margin_pct',
            color='category', size='revenue_12m',
            hover_data=hover_cols,
            title="Маржа vs Оборачиваемость",
            template='plotly_white',
            height=400
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with col_b:
        fig_hist = px.histogram(
            df_filtered, x='demand_cv', color='category',
            marginal='box', title="Распределение волатильности спроса",
            template='plotly_white', height=400
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    # Рекомендации по кластерам
    if 'recommendation' in df_filtered.columns:
        rec_counts = df_filtered['recommendation'].value_counts()
        fig_pie = px.pie(
            names=rec_counts.index, values=rec_counts.values,
            title="Распределение рекомендаций",
            color_discrete_map={
                'KEEP': '#28a745', 'EXIT': '#dc3545', 
                'REVIEW_PRICING': '#ffc107', 'SEASONAL': '#17a2b8'
            }
        )
        st.plotly_chart(fig_pie, use_container_width=True)

# ======================
# TAB 2: CLUSTERING
# ======================
with tab2:
    st.subheader("🤖 ML-кластеризация товаров")
    
    # Выбор фич
    available_features = ['revenue_12m', 'margin_pct', 'turnover_days', 'demand_cv', 'promo_share_pct', 'basket_affinity', 'stock_days']
    selected_features = st.multiselect("Метрики для кластеризации:", available_features, default=['revenue_12m', 'margin_pct', 'turnover_days'])
    
    col_opt, col_run = st.columns([1, 2])
    with col_opt:
        method = st.selectbox("Алгоритм:", ['kmeans', 'dbscan'])
        n_clusters = st.slider("Число кластеров (K-Means):", 3, 10, 5) if method == 'kmeans' else None
    
    if st.button("🚀 Запустить кластеризацию", type="primary"):
        with st.spinner("Выполняется кластеризация..."):
            df_clustered, profiles, profile_names = run_clustering(
                df_filtered, selected_features, n_clusters=n_clusters or 5, method=method
            )
            st.session_state['df_clustered'] = df_clustered
            
            # Silhouette score
            if method == 'kmeans':
                scores = compute_cluster_metrics(df_filtered, selected_features)
                best_k = scores.loc[scores['silhouette'].idxmax()]
                st.success(f"✅ Оптимальное K={best_k['n_clusters']:.0f} (silhouette={best_k['silhouette']:.3f})")
            
            # Визуализация кластеров (PCA для 2D)
            from sklearn.decomposition import PCA
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df_clustered[selected_features].fillna(0))
            pca = PCA(n_components=2)
            components = pca.fit_transform(X_scaled)
            
            df_viz = df_clustered.copy()
            df_viz['pc1'], df_viz['pc2'] = components[:, 0], components[:, 1]
            
            fig_cluster = px.scatter(
                df_viz.head(3000), x='pc1', y='pc2',
                color='cluster_name', hover_data=['sku_id', 'category', 'recommendation'],
                title=f"Кластеры в 2D (объяснённая дисперсия: {pca.explained_variance_ratio_.sum():.1%})",
                template='plotly_white', height=500
            )
            st.plotly_chart(fig_cluster, use_container_width=True)
            
            # Профили кластеров
            st.markdown("### 📋 Профили кластеров")
            st.dataframe(profiles.style.background_gradient(cmap='RdYlGn', subset=['margin_pct']), use_container_width=True)

# ======================
# TAB 3: DECISION TABLE
# ======================
with tab3:
    st.subheader("📋 Таблица решений по SKU")
    
    # Загружаем результаты кластеризации если есть
    df_show = st.session_state.get('df_clustered', df_filtered.copy())
    
    # Фильтр по рекомендации
    rec_filter = st.multiselect("Фильтр по рекомендации:", 
                                options=['KEEP', 'EXIT', 'SEASONAL', 'REVIEW_PRICING'],
                                default=['EXIT', 'REVIEW_PRICING'])
    
    df_display = df_show[df_show['recommendation'].isin(rec_filter)].copy() if 'recommendation' in df_show.columns else df_show
    
    # Интерактивная таблица с возможностью редактирования
    st.markdown("✅ **Интерактивная таблица**: отметьте финальные решения")
    
    # Добавляем колонку для ручного override
    if 'manual_decision' not in df_display.columns:
        df_display['manual_decision'] = df_display.get('recommendation', 'KEEP')
    
    # Показываем ключевые колонки
    display_cols = ['sku_id', 'category', 'region', 'revenue_12m', 'margin_pct', 
                   'turnover_days', 'recommendation', 'reason']
    available_cols = [c for c in display_cols if c in df_display.columns]
    
    # Применяем стилизацию только если колонка recommendation существует
    if 'recommendation' in df_display.columns:
        styled_df = df_display[available_cols].style.map(
            lambda x: 'background-color: #f8d7da' if x == 'EXIT' else 
                     ('background-color: #fff3cd' if 'REVIEW' in str(x) else ''),
            subset=['recommendation']
        )
    else:
        styled_df = df_display[available_cols]
    
    st.dataframe(
        styled_df,
        use_container_width=True,
        height=400
    )
    
    # Экспорт
    col_exp1, col_exp2 = st.columns(2)
    with col_exp1:
        csv = df_display.to_csv(index=False, encoding='utf-8-sig')
        st.download_button("📥 Скачать CSV", csv, "sku_decisions.csv", "text/csv")
    
    with col_exp2:
        if st.button("📄 Сформировать отчёт для руководства"):
            # Проверяем наличие колонки recommendation
            if 'recommendation' in df_display.columns:
                exit_count = (df_display['recommendation']=='EXIT').sum()
                review_count = (df_display['recommendation']=='REVIEW_PRICING').sum()
                seasonal_count = (df_display['recommendation']=='SEASONAL').sum()
                
                exit_df = df_display[df_display['recommendation']=='EXIT'].head(10)[['sku_id','category','reason']]
                report_text = f"""# Отчёт по оптимизации ассортимента
Дата: {pd.Timestamp.now().strftime('%Y-%m-%d')}
Всего проанализировано SKU: {len(df_filtered)}

## Рекомендации:
- К выводу: {exit_count} позиций
- Требуют пересмотра: {review_count} позиций
- Сезонные: {seasonal_count} позиций

## Топ-10 кандидатов на вывод:
{exit_df.to_markdown(index=False)}

*Сформировано автоматически в SKU Rationalization Dashboard*
"""
            else:
                report_text = f"""# Отчёт по оптимизации ассортимента
Дата: {pd.Timestamp.now().strftime('%Y-%m-%d')}
Всего проанализировано SKU: {len(df_filtered)}

## Статус:
Кластеризация ещё не выполнена. Пожалуйста, перейдите на вкладку "Кластеризация" и запустите анализ.

*Сформировано автоматически в SKU Rationalization Dashboard*
"""
            st.download_button("📥 Скачать отчёт (MD)", report_text, "assortment_report.md", "text/markdown")

# ======================
# TAB 4: SETTINGS & RULES
# ======================
with tab4:
    st.subheader("⚙️ Бизнес-правила и настройки")
    
    st.markdown("### 🎯 Пороговые значения для автоматических рекомендаций")
    
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        st.number_input("🔴 Мин. маржа для 'EXIT'", min_value=0, max_value=50, value=10, key='min_margin_exit')
        st.number_input("🔴 Макс. оборачиваемость для 'EXIT'", min_value=30, max_value=365, value=75, key='max_turnover_exit')
    
    with col_r2:
        st.number_input("🔵 Мин. корзина для 'Traffic Driver'", min_value=0.0, max_value=1.0, value=0.5, step=0.1, key='min_basket_driver')
        st.number_input("🟡 Порог волатильности для 'Seasonal'", min_value=0.5, max_value=3.0, value=1.2, step=0.1, key='cv_seasonal')
    
    st.markdown("### 💾 Сохранение конфигурации")
    if st.button("💾 Экспортировать настройки правил"):
        config = {
            'thresholds': {
                'min_margin_exit': st.session_state.min_margin_exit,
                'max_turnover_exit': st.session_state.max_turnover_exit,
                'min_basket_driver': st.session_state.min_basket_driver,
                'cv_seasonal': st.session_state.cv_seasonal
            },
            'features_used': selected_features if 'selected_features' in locals() else available_features
        }
        import json
        config_json = json.dumps(config, indent=2, ensure_ascii=False)
        st.download_button("📥 Скачать config.json", config_json, "rules_config.json", "application/json")
    
    st.info("💡 **Совет**: Настройте правила под вашу бизнес-модель и согласуйте с категорийными менеджерами перед запуском в production.")

# ======================
# FOOTER
# ======================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.9em;'>
    🔗 <b>SKU Rationalization Dashboard</b> | 
    Подготовлено для миграции на Greenplum | 
    <a href='https://github.com/serge3emskov/sku_rationalization' target='_blank'>GitHub репозиторий</a>
</div>
""", unsafe_allow_html=True)