import os
import pandas as pd
import numpy as np

from dash import Dash, html, dcc, Input, Output, State
import plotly.express as px


# ============================================================
# 1. Файлы
# ============================================================

DATA_FILE = "weekly_sla_incidents.csv"
ALL_REQUESTS_FILE = "table_end_v3.csv"


# ============================================================
# 2. Вспомогательные функции
# ============================================================

def safe_percent(part, total):
    if total == 0 or pd.isna(total):
        return 0.0
    return round(part / total * 100, 1)


def normalize_sla_fall(value):
    if pd.isna(value):
        return 0

    value = str(value).strip().lower()

    if value in ["1", "true", "yes", "y", "да"]:
        return 1

    return 0


def sla_status(value):
    if value == 1:
        return "Есть SLA-просрочка"
    return "Нет SLA-просрочки"


def normalize_priority(value):
    if pd.isna(value):
        return "Unknown"

    value = str(value).strip().lower()

    mapping = {
        "critical": "Critical",
        "crit": "Critical",
        "high": "High",
        "medium": "Medium",
        "med": "Medium",
        "low": "Low",
    }

    return mapping.get(value, str(value).strip())


def priority_weight(priority):
    priority = normalize_priority(priority)

    weights = {
        "Critical": 5,
        "High": 3,
        "Medium": 2,
        "Low": 1,
    }

    return weights.get(priority, 1)


def fix_time_bucket(hours):
    if pd.isna(hours):
        return "Не закрыто / нет данных"

    if hours <= 4:
        return "до 4 часов"
    elif hours <= 12:
        return "4–12 часов"
    elif hours <= 24:
        return "12–24 часа"
    elif hours <= 72:
        return "1–3 дня"
    else:
        return "более 3 дней"


def classify_reason(text):
    if pd.isna(text) or str(text).strip() == "":
        return "Не классифицировано"

    text = str(text).lower()

    if any(word in text for word in [
        "deadlock", "timeout", "субд", "бд", "database",
        "iops", "connection pool", "производительность"
    ]):
        return "Производительность БД"

    if any(word in text for word in [
        "json", "xml", "источник", "валидац", "поставщик",
        "primary key", "данн", "source", "schema"
    ]):
        return "Ошибка в данных / источнике"

    if any(word in text for word in [
        "cascade", "каскад", "upstream", "downstream",
        "зависим", "lineage"
    ]):
        return "Проблемы с зависимостями"

    if any(word in text for word in [
        "error", "exception", "valueerror", "zerodivision",
        "логическ", "код", "script"
    ]):
        return "Ошибка в коде / логике"

    if any(word in text for word in [
        "api", "gateway", "disk", "ssl", "connectionreset",
        "инфраструктур", "network", "сервер"
    ]):
        return "Инфраструктурный сбой"

    if any(word in text for word in [
        "ручн", "дежурн", "администратор", "manual", "paus",
        "человеческий"
    ]):
        return "Человеческий фактор"

    return "Не классифицировано"


# ============================================================
# 3. Загрузка данных
# ============================================================

def load_and_prepare_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE)
    df.columns = df.columns.str.upper()

    required_columns = [
        "ID",
        "SLA_FALL",
        "PROJECT_NAME",
        "OBJECT_NAME",
        "PRIORITY",
        "CREATE_DATE",
        "FIX_DATE",
        "DATA_DOMAIN",
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"В файле {DATA_FILE} не хватает столбцов: {missing_columns}")

    optional_columns = {
        "REQ_ID": np.nan,
        "SLA_FALL_DATE": pd.NaT,
        "ETA_FALL_DATE": pd.NaT,
        "INCIDENT_DATE": pd.NaT,
        "BUG_DETAILS": "Не указано",
        "REASON_CATEGORY": "",
        "FIX_DESCRIPTION": "Не указано",
        "RESPONSIBLE": "Не указано",
        "DOMAIN_CRITICALITY": "Не указано",
        "REQ_TYPE": "Не указано",
        "REQ_STATUS": "Не указано",
        "REQ_BEGIN_DATE": pd.NaT,
        "REQ_END_DATE": pd.NaT,
        "FIX_TIME_HOURS": np.nan,
        "WEEK_START": pd.NaT,
        "RISK_SCORE": np.nan,
        "I_REQ_ID": "",
        "UP_REQ_ID": np.nan,
        "I_SCHEMA_NAME": "",
        "COMMENTS": "",
        "DEPENDENCY_COUNT": 0,
        "HAS_DEPENDENCY": 0,
        "ROOT_CAUSE_REQ_ID": np.nan,
        "ROOT_CAUSE_PROJECT_NAME": "",
        "ROOT_CAUSE_OBJECT_NAME": "",
        "ROOT_CAUSE_DOMAIN": "",
        "ROOT_CAUSE_REASON_CATEGORY": "",
        "DEPENDENCY_DEPTH": 0,
        "IS_CASCADE_INCIDENT": 0,
        "DOWNSTREAM_IMPACT_COUNT": 0,
    }

    for col, default_value in optional_columns.items():
        if col not in df.columns:
            df[col] = default_value

    date_columns = [
        "SLA_FALL_DATE",
        "ETA_FALL_DATE",
        "CREATE_DATE",
        "FIX_DATE",
        "INCIDENT_DATE",
        "WEEK_START",
        "REQ_BEGIN_DATE",
        "REQ_END_DATE",
    ]

    for col in date_columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    df["SLA_FALL"] = df["SLA_FALL"].apply(normalize_sla_fall)
    df["SLA_STATUS"] = df["SLA_FALL"].apply(sla_status)

    df["PRIORITY"] = df["PRIORITY"].apply(normalize_priority)

    df["INCIDENT_DATE"] = df["INCIDENT_DATE"].fillna(
        df["SLA_FALL_DATE"].fillna(df["CREATE_DATE"])
    )

    df = df[df["INCIDENT_DATE"].notna()].copy()

    df["WEEK_START"] = df["WEEK_START"].fillna(
        df["INCIDENT_DATE"] - pd.to_timedelta(df["INCIDENT_DATE"].dt.weekday, unit="D")
    )

    df["FIX_TIME_HOURS"] = pd.to_numeric(df["FIX_TIME_HOURS"], errors="coerce")

    if df["FIX_TIME_HOURS"].isna().all():
        df["FIX_TIME_HOURS"] = (
            df["FIX_DATE"] - df["CREATE_DATE"]
        ).dt.total_seconds() / 3600

    df["RISK_SCORE"] = pd.to_numeric(df["RISK_SCORE"], errors="coerce")
    df["RISK_SCORE"] = df["RISK_SCORE"].fillna(df["PRIORITY"].apply(priority_weight))

    df["IS_OPEN"] = df["FIX_DATE"].isna().astype(int)
    df["IS_CRITICAL_OR_HIGH"] = df["PRIORITY"].isin(["Critical", "High"]).astype(int)

    if "РЕЗУЛЬТАТ_КАТЕГОРИЯ_1" in df.columns:
        df["REASON_CATEGORY"] = np.where(
            df["REASON_CATEGORY"].fillna("").astype(str).str.strip() == "",
            df["РЕЗУЛЬТАТ_КАТЕГОРИЯ_1"],
            df["REASON_CATEGORY"],
        )

    df["REASON_CATEGORY"] = (
        df["REASON_CATEGORY"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    empty_reason_mask = df["REASON_CATEGORY"].isin(
        ["", "nan", "NaN", "None", "не указано"]
    )

    df.loc[empty_reason_mask, "REASON_CATEGORY"] = (
        df.loc[empty_reason_mask, "BUG_DETAILS"].apply(classify_reason)
    )

    text_columns = [
        "PROJECT_NAME",
        "OBJECT_NAME",
        "DATA_DOMAIN",
        "BUG_DETAILS",
        "REASON_CATEGORY",
        "FIX_DESCRIPTION",
        "RESPONSIBLE",
        "DOMAIN_CRITICALITY",
        "REQ_TYPE",
        "REQ_STATUS",
        "ROOT_CAUSE_PROJECT_NAME",
        "ROOT_CAUSE_OBJECT_NAME",
        "ROOT_CAUSE_DOMAIN",
        "ROOT_CAUSE_REASON_CATEGORY",
        "I_SCHEMA_NAME",
        "COMMENTS",
    ]

    for col in text_columns:
        df[col] = df[col].fillna("Не указано").astype(str)

    numeric_columns = [
        "DEPENDENCY_COUNT",
        "HAS_DEPENDENCY",
        "IS_CASCADE_INCIDENT",
        "DEPENDENCY_DEPTH",
        "DOWNSTREAM_IMPACT_COUNT",
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    df["FIX_TIME_BUCKET"] = df["FIX_TIME_HOURS"].apply(fix_time_bucket)

    repeat_counts = (
        df.groupby(["OBJECT_NAME", "REASON_CATEGORY"])["ID"]
        .transform("count")
    )

    df["IS_REPEATED"] = (repeat_counts > 1).astype(int)

    return df


def load_all_requests_data() -> pd.DataFrame:
    if not os.path.exists(ALL_REQUESTS_FILE):
        return pd.DataFrame()

    df_all = pd.read_csv(ALL_REQUESTS_FILE)
    df_all.columns = df_all.columns.str.upper()

    if "INCIDENT_DATE" in df_all.columns:
        df_all["REQUEST_DATE"] = pd.to_datetime(df_all["INCIDENT_DATE"], errors="coerce")
    elif "CREATE_DATE" in df_all.columns:
        df_all["REQUEST_DATE"] = pd.to_datetime(df_all["CREATE_DATE"], errors="coerce")
    elif "REQ_BEGIN_DATE" in df_all.columns:
        df_all["REQUEST_DATE"] = pd.to_datetime(df_all["REQ_BEGIN_DATE"], errors="coerce")
    elif "UPDATEDT" in df_all.columns:
        df_all["REQUEST_DATE"] = pd.to_datetime(df_all["UPDATEDT"], errors="coerce")
    else:
        df_all["REQUEST_DATE"] = pd.NaT

    if "REQ_ID" not in df_all.columns:
        df_all["REQ_ID"] = range(1, len(df_all) + 1)

    if "DATA_DOMAIN" not in df_all.columns:
        df_all["DATA_DOMAIN"] = np.nan

    if "PROJECT_NAME" not in df_all.columns:
        df_all["PROJECT_NAME"] = "Не указано"

    if "OBJECT_NAME" not in df_all.columns:
        df_all["OBJECT_NAME"] = "Не указано"

    return df_all


def filter_data(df, start_date, end_date, domains, priorities, sla_status_value):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date) + pd.Timedelta(days=1)

    filtered = df[
        (df["INCIDENT_DATE"] >= start_date)
        & (df["INCIDENT_DATE"] < end_date)
    ].copy()

    if domains:
        filtered = filtered[filtered["DATA_DOMAIN"].isin(domains)]

    if priorities:
        filtered = filtered[filtered["PRIORITY"].isin(priorities)]

    if sla_status_value == "breach":
        filtered = filtered[filtered["SLA_FALL"] == 1]
    elif sla_status_value == "no_breach":
        filtered = filtered[filtered["SLA_FALL"] == 0]

    return filtered


def filter_all_requests_by_period(all_requests_df, start_date, end_date):
    if all_requests_df.empty:
        return all_requests_df

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date) + pd.Timedelta(days=1)

    if "REQUEST_DATE" in all_requests_df.columns and all_requests_df["REQUEST_DATE"].notna().any():
        return all_requests_df[
            (all_requests_df["REQUEST_DATE"] >= start_date)
            & (all_requests_df["REQUEST_DATE"] < end_date)
        ].copy()

    return all_requests_df.copy()


# ============================================================
# 4. Метрики
# ============================================================

def build_reason_top(df: pd.DataFrame, selected_domain="all", top_n=10) -> pd.DataFrame:
    temp = df.copy()

    if selected_domain and selected_domain != "all":
        temp = temp[temp["DATA_DOMAIN"] == selected_domain]

    if temp.empty:
        return pd.DataFrame()

    reason_df = (
        temp.groupby("REASON_CATEGORY")
        .agg(
            INCIDENTS=("ID", "count"),
            SLA_BREACHES=("SLA_FALL", "sum"),
            RISK_SCORE=("RISK_SCORE", "sum"),
            AFFECTED_OBJECTS=("OBJECT_NAME", "nunique"),
            AVG_FIX_TIME_HOURS=("FIX_TIME_HOURS", "mean"),
        )
        .reset_index()
        .sort_values(["RISK_SCORE", "INCIDENTS", "SLA_BREACHES"], ascending=False)
        .head(top_n)
    )

    reason_df["AVG_FIX_TIME_HOURS"] = reason_df["AVG_FIX_TIME_HOURS"].round(1)

    return reason_df


def build_domain_percent_metrics(incidents_df: pd.DataFrame, all_requests_df: pd.DataFrame) -> pd.DataFrame:
    if incidents_df.empty or all_requests_df.empty:
        return pd.DataFrame()

    incidents = incidents_df.copy()
    requests = all_requests_df.copy()

    if requests["DATA_DOMAIN"].isna().all():
        project_domain_map = (
            incidents[["PROJECT_NAME", "DATA_DOMAIN"]]
            .dropna()
            .drop_duplicates("PROJECT_NAME")
        )

        requests = requests.merge(
            project_domain_map,
            how="left",
            on="PROJECT_NAME",
            suffixes=("", "_FROM_INCIDENTS"),
        )

        if "DATA_DOMAIN_FROM_INCIDENTS" in requests.columns:
            requests["DATA_DOMAIN"] = requests["DATA_DOMAIN"].fillna(
                requests["DATA_DOMAIN_FROM_INCIDENTS"]
            )
            requests = requests.drop(columns=["DATA_DOMAIN_FROM_INCIDENTS"], errors="ignore")

    requests["DATA_DOMAIN"] = requests["DATA_DOMAIN"].fillna("НЕОПРЕДЕЛЕНО")
    requests["OBJECT_NAME"] = requests["OBJECT_NAME"].fillna("Не указано")

    total_requests_by_domain = (
        requests.groupby("DATA_DOMAIN")
        .agg(
            TOTAL_REQUESTS=("REQ_ID", "count"),
            TOTAL_OBJECTS=("OBJECT_NAME", "nunique"),
        )
        .reset_index()
    )

    incidents_by_domain = (
        incidents.groupby("DATA_DOMAIN")
        .agg(
            INCIDENTS=("ID", "count"),
            SLA_BREACHES=("SLA_FALL", "sum"),
            RISK_SCORE=("RISK_SCORE", "sum"),
            AFFECTED_OBJECTS=("OBJECT_NAME", "nunique"),
            AVG_FIX_TIME_HOURS=("FIX_TIME_HOURS", "mean"),
        )
        .reset_index()
    )

    result = total_requests_by_domain.merge(
        incidents_by_domain,
        how="left",
        on="DATA_DOMAIN",
    )

    for col in ["INCIDENTS", "SLA_BREACHES", "RISK_SCORE", "AFFECTED_OBJECTS"]:
        result[col] = result[col].fillna(0)

    result["AVG_FIX_TIME_HOURS"] = result["AVG_FIX_TIME_HOURS"].fillna(0)

    result["INCIDENT_RATE_PCT"] = (
        result["INCIDENTS"] / result["TOTAL_REQUESTS"] * 100
    ).round(2)

    result["SLA_BREACH_RATE_PCT"] = (
        result["SLA_BREACHES"] / result["TOTAL_REQUESTS"] * 100
    ).round(2)

    result["AFFECTED_OBJECTS_SHARE_PCT"] = (
        result["AFFECTED_OBJECTS"] / result["TOTAL_OBJECTS"] * 100
    ).replace([np.inf, -np.inf], 0).fillna(0).round(2)

    result["RISK_SCORE_PER_1000_REQUESTS"] = (
        result["RISK_SCORE"] / result["TOTAL_REQUESTS"] * 1000
    ).round(2)

    result["AVG_FIX_TIME_HOURS"] = result["AVG_FIX_TIME_HOURS"].round(1)

    result = result.sort_values(
        ["INCIDENT_RATE_PCT", "SLA_BREACH_RATE_PCT", "RISK_SCORE_PER_1000_REQUESTS"],
        ascending=False,
    )

    return result


def get_default_dates():
    try:
        df = load_and_prepare_data()
        return (
            df["INCIDENT_DATE"].min().date().isoformat(),
            df["INCIDENT_DATE"].max().date().isoformat(),
        )
    except Exception:
        return "2026-05-10", "2026-05-29"


# ============================================================
# 5. Визуальные функции
# ============================================================

def create_empty_figure(title):
    fig = px.bar(title=title)

    fig.update_layout(
        annotations=[
            dict(
                text="Нет данных",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=18),
            )
        ],
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    return fig


def style_figure(fig):
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=13),
        title_font=dict(size=18),
        margin=dict(l=30, r=30, t=60, b=30),
    )
    return fig


def create_kpi_card(title, value, subtitle=""):
    return html.Div(
        style={
            "backgroundColor": "white",
            "padding": "18px",
            "borderRadius": "14px",
            "boxShadow": "0 2px 10px rgba(0,0,0,0.08)",
            "textAlign": "center",
            "minHeight": "120px",
        },
        children=[
            html.H4(title, style={"margin": "0 0 10px 0", "color": "#555", "fontSize": "15px"}),
            html.H2(str(value), style={"margin": "0", "color": "#1f77b4", "fontSize": "28px"}),
            html.P(subtitle, style={"margin": "10px 0 0 0", "color": "#777", "fontSize": "13px"}),
        ],
    )


def create_table(df, columns, max_rows=15):
    if df.empty:
        return html.P(
            "Данных для таблицы нет.",
            style={
                "backgroundColor": "white",
                "padding": "20px",
                "borderRadius": "12px",
                "boxShadow": "0 2px 10px rgba(0,0,0,0.08)",
            },
        )

    df = df.head(max_rows)

    return html.Div(
        style={
            "overflowX": "auto",
            "backgroundColor": "white",
            "borderRadius": "12px",
            "boxShadow": "0 2px 10px rgba(0,0,0,0.08)",
        },
        children=[
            html.Table(
                style={"width": "100%", "borderCollapse": "collapse", "fontSize": "14px"},
                children=[
                    html.Thead(
                        html.Tr(
                            [
                                html.Th(
                                    col,
                                    style={
                                        "borderBottom": "1px solid #ddd",
                                        "padding": "10px",
                                        "textAlign": "left",
                                        "backgroundColor": "#f0f3f8",
                                    },
                                )
                                for col in columns
                            ]
                        )
                    ),
                    html.Tbody(
                        [
                            html.Tr(
                                [
                                    html.Td(
                                        row[col],
                                        style={
                                            "borderBottom": "1px solid #eee",
                                            "padding": "10px",
                                            "verticalAlign": "top",
                                        },
                                    )
                                    for col in columns
                                ]
                            )
                            for _, row in df.iterrows()
                        ]
                    ),
                ],
            )
        ],
    )


# ============================================================
# 6. Создание приложения
# ============================================================

default_start, default_end = get_default_dates()

app = Dash(__name__)
server = app.server
app.title = "MOEX SLA Dashboard"


# ============================================================
# 7. Layout
# ============================================================

app.layout = html.Div(
    style={
        "fontFamily": "Arial, sans-serif",
        "backgroundColor": "#f5f7fb",
        "padding": "30px",
    },
    children=[
        html.H1(
            "Дашборд по инцидентам, SLA и причинам сбоев",
            style={"textAlign": "center", "marginBottom": "10px"},
        ),

        html.P(
            "Анализ SLA-инцидентов, корневых причин, зависимостей и проблемности доменов",
            style={"textAlign": "center", "color": "#555", "marginBottom": "30px"},
        ),

        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "1fr 1fr 1fr 1fr 1fr auto",
                "gap": "18px",
                "alignItems": "end",
                "marginBottom": "30px",
                "backgroundColor": "white",
                "padding": "20px",
                "borderRadius": "12px",
                "boxShadow": "0 2px 10px rgba(0,0,0,0.08)",
            },
            children=[
                html.Div([
                    html.Label("Дата начала"),
                    dcc.DatePickerSingle(
                        id="start-date",
                        date=default_start,
                        display_format="YYYY-MM-DD",
                    ),
                ]),

                html.Div([
                    html.Label("Дата окончания"),
                    dcc.DatePickerSingle(
                        id="end-date",
                        date=default_end,
                        display_format="YYYY-MM-DD",
                    ),
                ]),

                html.Div([
                    html.Label("Домен"),
                    dcc.Dropdown(
                        id="domain-filter",
                        multi=True,
                        placeholder="Все домены",
                    ),
                ]),

                html.Div([
                    html.Label("Критичность"),
                    dcc.Dropdown(
                        id="priority-filter",
                        multi=True,
                        placeholder="Все уровни",
                    ),
                ]),

                html.Div([
                    html.Label("Факт SLA-просрочки"),
                    dcc.Dropdown(
                        id="sla-status-filter",
                        options=[
                            {"label": "Все инциденты", "value": "all"},
                            {"label": "Есть SLA-просрочка", "value": "breach"},
                            {"label": "Нет SLA-просрочки", "value": "no_breach"},
                        ],
                        value="all",
                        clearable=False,
                    ),
                ]),

                html.Button(
                    "Сформировать отчёт",
                    id="generate-button",
                    n_clicks=0,
                    style={
                        "height": "40px",
                        "backgroundColor": "#1f77b4",
                        "color": "white",
                        "border": "none",
                        "borderRadius": "8px",
                        "padding": "0 20px",
                        "cursor": "pointer",
                        "fontWeight": "bold",
                    },
                ),
            ],
        ),

        html.Div(
            id="error-message",
            style={
                "color": "red",
                "textAlign": "center",
                "marginBottom": "20px",
                "fontWeight": "bold",
            },
        ),

        html.Div(
            id="kpi-cards",
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(5, 1fr)",
                "gap": "18px",
                "marginBottom": "30px",
            },
        ),

        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "1fr 1fr",
                "gap": "25px",
            },
            children=[
                dcc.Graph(id="weekly-chart"),
                dcc.Graph(id="sla-status-chart"),
                dcc.Graph(id="domain-chart"),
                dcc.Graph(id="reason-chart"),
                dcc.Graph(id="object-chart"),
                dcc.Graph(id="root-cause-chart"),
                dcc.Graph(id="fix-time-chart"),
                dcc.Graph(id="risk-domain-chart"),
            ],
        ),

        html.Div(
            style={"marginTop": "35px"},
            children=[
                html.H2("ТОП причин инцидентов"),
                html.P(
                    "Можно посмотреть причины в целом или отдельно по выбранному домену.",
                    style={"color": "#666"},
                ),

                html.Div(
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "1fr 180px",
                        "gap": "18px",
                        "alignItems": "end",
                        "marginBottom": "20px",
                        "backgroundColor": "white",
                        "padding": "20px",
                        "borderRadius": "12px",
                        "boxShadow": "0 2px 10px rgba(0,0,0,0.08)",
                    },
                    children=[
                        html.Div([
                            html.Label("Домен для анализа причин"),
                            dcc.Dropdown(
                                id="reason-domain-filter",
                                placeholder="Все домены",
                                clearable=False,
                            ),
                        ]),

                        html.Div([
                            html.Label("Показать"),
                            dcc.Dropdown(
                                id="reason-top-n",
                                options=[
                                    {"label": "ТОП-5", "value": 5},
                                    {"label": "ТОП-10", "value": 10},
                                    {"label": "ТОП-15", "value": 15},
                                ],
                                value=10,
                                clearable=False,
                            ),
                        ]),
                    ],
                ),

                dcc.Graph(id="reason-top-chart"),
                html.Div(id="reason-top-table"),
            ],
        ),

        html.Div(
            style={"marginTop": "35px"},
            children=[
                html.H2("Проблемность доменов с учётом общего числа заявок"),
                html.P(
                    "Сравнение доменов не только по количеству инцидентов, но и по доле от всех заявок / загрузок.",
                    style={"color": "#666"},
                ),
                dcc.Graph(id="domain-percent-chart"),
                html.Div(id="domain-percent-table"),
            ],
        ),

        html.Div(
            style={"marginTop": "35px"},
            children=[
                html.H2("ТОП корневых причин по зависимостям"),
                html.P(
                    "Показывает объекты, сбой которых мог вызвать наибольшее число зависимых инцидентов.",
                    style={"color": "#666"},
                ),
                html.Div(id="root-cause-table"),
            ],
        ),

        html.Div(
            style={"marginTop": "35px"},
            children=[
                html.H2("ТОП проблемных зон"),
                html.P(
                    "Домен → проект → объект → причина. Сортировка по индексу риска и количеству инцидентов.",
                    style={"color": "#666"},
                ),
                html.Div(id="problem-areas-table"),
            ],
        ),

        html.Div(
            style={"marginTop": "35px"},
            children=[
                html.H2("Повторяющиеся инциденты"),
                html.Div(id="repeated-failures-table"),
            ],
        ),
    ],
)


# ============================================================
# 8. Callback для фильтров
# ============================================================

@app.callback(
    Output("domain-filter", "options"),
    Output("priority-filter", "options"),
    Output("reason-domain-filter", "options"),
    Output("reason-domain-filter", "value"),
    Input("generate-button", "n_clicks"),
)
def init_filters(_):
    try:
        df = load_and_prepare_data()

        domain_values = sorted(df["DATA_DOMAIN"].dropna().unique())

        domain_options = [
            {"label": domain, "value": domain}
            for domain in domain_values
        ]

        reason_domain_options = [{"label": "Все домены", "value": "all"}] + domain_options

        priority_order = ["Critical", "High", "Medium", "Low", "Unknown"]
        existing_priorities = df["PRIORITY"].dropna().unique().tolist()
        priorities = [p for p in priority_order if p in existing_priorities]

        priority_options = [
            {"label": priority, "value": priority}
            for priority in priorities
        ]

        return domain_options, priority_options, reason_domain_options, "all"

    except Exception:
        return [], [], [{"label": "Все домены", "value": "all"}], "all"


# ============================================================
# 9. Основной callback
# ============================================================

@app.callback(
    Output("error-message", "children"),
    Output("kpi-cards", "children"),

    Output("weekly-chart", "figure"),
    Output("sla-status-chart", "figure"),
    Output("domain-chart", "figure"),
    Output("reason-chart", "figure"),
    Output("object-chart", "figure"),
    Output("root-cause-chart", "figure"),
    Output("fix-time-chart", "figure"),
    Output("risk-domain-chart", "figure"),

    Output("reason-top-chart", "figure"),
    Output("reason-top-table", "children"),

    Output("domain-percent-chart", "figure"),
    Output("domain-percent-table", "children"),

    Output("root-cause-table", "children"),
    Output("problem-areas-table", "children"),
    Output("repeated-failures-table", "children"),

    Input("generate-button", "n_clicks"),
    State("start-date", "date"),
    State("end-date", "date"),
    State("domain-filter", "value"),
    State("priority-filter", "value"),
    State("sla-status-filter", "value"),
    State("reason-domain-filter", "value"),
    State("reason-top-n", "value"),
)
def update_dashboard(
    n_clicks,
    start_date,
    end_date,
    domains,
    priorities,
    sla_status_value,
    reason_domain_value,
    reason_top_n,
):
    empty_fig = create_empty_figure("Нажмите кнопку, чтобы сформировать отчёт")

    if n_clicks == 0:
        return (
            "",
            [],
            empty_fig, empty_fig, empty_fig, empty_fig,
            empty_fig, empty_fig, empty_fig, empty_fig,
            empty_fig, "",
            empty_fig, "",
            "", "", "",
        )

    try:
        df_all = load_and_prepare_data()
        df = filter_data(df_all, start_date, end_date, domains, priorities, sla_status_value)

        all_requests_df = load_all_requests_data()
        all_requests_period_df = filter_all_requests_by_period(
            all_requests_df,
            start_date,
            end_date,
        )

        if df.empty:
            no_data_fig = create_empty_figure("Нет данных")

            return (
                "За выбранный период нет инцидентов",
                [],
                no_data_fig, no_data_fig, no_data_fig, no_data_fig,
                no_data_fig, no_data_fig, no_data_fig, no_data_fig,
                no_data_fig, "",
                no_data_fig, "",
                "", "", "",
            )

        # KPI
        total_incidents = len(df)
        sla_breach_count = int((df["SLA_FALL"] == 1).sum())
        no_sla_breach_count = int((df["SLA_FALL"] == 0).sum())
        sla_breach_share = safe_percent(sla_breach_count, total_incidents)

        critical_high_count = int(df["IS_CRITICAL_OR_HIGH"].sum())
        critical_high_share = safe_percent(critical_high_count, total_incidents)

        cascade_count = int(df["IS_CASCADE_INCIDENT"].sum())
        cascade_share = safe_percent(cascade_count, total_incidents)

        open_count = int(df["IS_OPEN"].sum())
        open_share = safe_percent(open_count, total_incidents)

        risk_score = int(df["RISK_SCORE"].sum())

        avg_fix_time = df["FIX_TIME_HOURS"].mean()
        p90_fix_time = df["FIX_TIME_HOURS"].quantile(0.9)

        avg_fix_time = round(avg_fix_time, 1) if pd.notna(avg_fix_time) else 0
        p90_fix_time = round(p90_fix_time, 1) if pd.notna(p90_fix_time) else 0

        max_downstream_impact = int(df["DOWNSTREAM_IMPACT_COUNT"].max()) if not df.empty else 0

        top_root_df_for_kpi = (
            df[df["IS_CASCADE_INCIDENT"] == 1]
            .groupby("ROOT_CAUSE_OBJECT_NAME")
            .agg(DEPENDENT_INCIDENTS=("ID", "count"))
            .reset_index()
            .sort_values("DEPENDENT_INCIDENTS", ascending=False)
        )

        if top_root_df_for_kpi.empty:
            top_root_object = "Нет данных"
        else:
            top_root_object = top_root_df_for_kpi.iloc[0]["ROOT_CAUSE_OBJECT_NAME"]

        kpi_cards = [
            create_kpi_card("Всего инцидентов", total_incidents, "SLA_FALL = 0 или 1"),
            create_kpi_card("SLA-просрочки", sla_breach_count, f"{sla_breach_share}% от выбранных инцидентов"),
            create_kpi_card("Без SLA-просрочки", no_sla_breach_count, "Сбой был, но закрыт до SLA"),
            create_kpi_card("Критичные и высокие", critical_high_count, f"{critical_high_share}% от выбранных инцидентов"),
            create_kpi_card("Каскадные инциденты", cascade_count, f"{cascade_share}% от выбранных инцидентов"),
            create_kpi_card("Макс. число зависимых инцидентов", max_downstream_impact, f"Корневая причина: {top_root_object}"),
            create_kpi_card("Индекс риска", risk_score, "Чем выше значение, тем выше бизнес-риск"),
            create_kpi_card("Открытые инциденты", open_count, f"{open_share}% ещё не закрыто"),
            create_kpi_card("Среднее устранение", f"{avg_fix_time} ч", "Среднее FIX_DATE - CREATE_DATE"),
            create_kpi_card("90% инцидентов устранены за", f"{p90_fix_time} ч", "Не дольше указанного времени"),
        ]

        # Основные графики
        weekly_df = (
            df.groupby("WEEK_START")
            .agg(COUNT=("ID", "count"))
            .reset_index()
            .sort_values("WEEK_START")
        )

        sla_status_df = (
            df.groupby("SLA_STATUS")
            .agg(COUNT=("ID", "count"))
            .reset_index()
        )

        domain_df = (
            df.groupby("DATA_DOMAIN")
            .agg(
                COUNT=("ID", "count"),
                SLA_BREACHES=("SLA_FALL", "sum"),
                RISK_SCORE=("RISK_SCORE", "sum"),
            )
            .reset_index()
            .sort_values("COUNT", ascending=False)
        )

        reason_df = (
            df.groupby("REASON_CATEGORY")
            .agg(
                COUNT=("ID", "count"),
                RISK_SCORE=("RISK_SCORE", "sum"),
            )
            .reset_index()
            .sort_values(["COUNT", "RISK_SCORE"], ascending=False)
        )

        object_df = (
            df.groupby("OBJECT_NAME")
            .agg(
                COUNT=("ID", "count"),
                RISK_SCORE=("RISK_SCORE", "sum"),
            )
            .reset_index()
            .sort_values(["RISK_SCORE", "COUNT"], ascending=False)
            .head(10)
        )

        root_cause_chart_df = (
            df[df["IS_CASCADE_INCIDENT"] == 1]
            .groupby("ROOT_CAUSE_OBJECT_NAME")
            .agg(
                DEPENDENT_INCIDENTS=("ID", "count"),
                RISK_SCORE=("RISK_SCORE", "sum"),
            )
            .reset_index()
            .sort_values(["DEPENDENT_INCIDENTS", "RISK_SCORE"], ascending=False)
            .head(10)
        )

        fix_bucket_order = [
            "до 4 часов",
            "4–12 часов",
            "12–24 часа",
            "1–3 дня",
            "более 3 дней",
            "Не закрыто / нет данных",
        ]

        fix_time_df = (
            df.groupby("FIX_TIME_BUCKET")
            .agg(COUNT=("ID", "count"))
            .reset_index()
        )

        fix_time_df["FIX_TIME_BUCKET"] = pd.Categorical(
            fix_time_df["FIX_TIME_BUCKET"],
            categories=fix_bucket_order,
            ordered=True,
        )

        fix_time_df = fix_time_df.sort_values("FIX_TIME_BUCKET")

        risk_domain_df = (
            df.groupby("DATA_DOMAIN")
            .agg(RISK_SCORE=("RISK_SCORE", "sum"))
            .reset_index()
            .sort_values("RISK_SCORE", ascending=False)
        )

        weekly_fig = px.line(
            weekly_df,
            x="WEEK_START",
            y="COUNT",
            markers=True,
            title="Динамика инцидентов по неделям",
            labels={"WEEK_START": "Неделя", "COUNT": "Инцидентов"},
        )

        sla_status_fig = px.pie(
            sla_status_df,
            names="SLA_STATUS",
            values="COUNT",
            title="Инциденты с SLA-просрочкой и без",
            hole=0.45,
        )

        domain_fig = px.bar(
            domain_df,
            x="DATA_DOMAIN",
            y="COUNT",
            title="Инциденты по доменам",
            labels={"DATA_DOMAIN": "Домен", "COUNT": "Инцидентов"},
            text="COUNT",
        )

        reason_fig = px.bar(
            reason_df.sort_values("COUNT"),
            x="COUNT",
            y="REASON_CATEGORY",
            title="Причины инцидентов",
            labels={"COUNT": "Инцидентов", "REASON_CATEGORY": "Причина"},
            text="COUNT",
            orientation="h",
        )

        object_fig = px.bar(
            object_df.sort_values("RISK_SCORE"),
            x="RISK_SCORE",
            y="OBJECT_NAME",
            title="ТОП-10 объектов по индексу риска",
            labels={"RISK_SCORE": "Индекс риска", "OBJECT_NAME": "ETL-объект"},
            text="RISK_SCORE",
            orientation="h",
        )

        if root_cause_chart_df.empty:
            root_cause_fig = create_empty_figure("ТОП объектов — корневых причин")
        else:
            root_cause_fig = px.bar(
                root_cause_chart_df.sort_values("DEPENDENT_INCIDENTS"),
                x="DEPENDENT_INCIDENTS",
                y="ROOT_CAUSE_OBJECT_NAME",
                title="ТОП объектов по числу зависимых инцидентов",
                labels={
                    "DEPENDENT_INCIDENTS": "Зависимых инцидентов",
                    "ROOT_CAUSE_OBJECT_NAME": "Объект — корневая причина",
                },
                text="DEPENDENT_INCIDENTS",
                orientation="h",
            )

        fix_time_fig = px.bar(
            fix_time_df,
            x="FIX_TIME_BUCKET",
            y="COUNT",
            title="Распределение времени устранения",
            labels={"FIX_TIME_BUCKET": "Время устранения", "COUNT": "Инцидентов"},
            text="COUNT",
        )

        risk_domain_fig = px.bar(
            risk_domain_df,
            x="DATA_DOMAIN",
            y="RISK_SCORE",
            title="Индекс риска по доменам",
            labels={"DATA_DOMAIN": "Домен", "RISK_SCORE": "Индекс риска"},
            text="RISK_SCORE",
        )

        # ТОП причин
        reason_top_df = build_reason_top(
            df=df,
            selected_domain=reason_domain_value,
            top_n=int(reason_top_n or 10),
        )

        if reason_top_df.empty:
            reason_top_fig = create_empty_figure("ТОП причин инцидентов")
            reason_top_table = html.P(
                "Нет данных по причинам для выбранного фильтра.",
                style={
                    "backgroundColor": "white",
                    "padding": "20px",
                    "borderRadius": "12px",
                    "boxShadow": "0 2px 10px rgba(0,0,0,0.08)",
                },
            )
        else:
            reason_top_fig = px.bar(
                reason_top_df.sort_values("RISK_SCORE"),
                x="RISK_SCORE",
                y="REASON_CATEGORY",
                title="ТОП причин по индексу риска",
                labels={"RISK_SCORE": "Индекс риска", "REASON_CATEGORY": "Причина"},
                text="RISK_SCORE",
                orientation="h",
            )

            reason_top_table_df = reason_top_df.rename(
                columns={
                    "REASON_CATEGORY": "Причина",
                    "INCIDENTS": "Инцидентов",
                    "SLA_BREACHES": "SLA-просрочек",
                    "RISK_SCORE": "Индекс риска",
                    "AFFECTED_OBJECTS": "Затронуто объектов",
                    "AVG_FIX_TIME_HOURS": "Среднее устранение, ч",
                }
            )

            reason_top_table = create_table(
                reason_top_table_df,
                [
                    "Причина",
                    "Инцидентов",
                    "SLA-просрочек",
                    "Индекс риска",
                    "Затронуто объектов",
                    "Среднее устранение, ч",
                ],
                max_rows=int(reason_top_n or 10),
            )

        # Процентные метрики доменов
        domain_percent_df = build_domain_percent_metrics(
            incidents_df=df,
            all_requests_df=all_requests_period_df,
        )

        if domain_percent_df.empty:
            domain_percent_fig = create_empty_figure("Проблемность доменов в процентах")
            domain_percent_table = html.P(
                "Файл table_end_v3.csv не найден или нет данных для расчёта процентных метрик.",
                style={
                    "backgroundColor": "white",
                    "padding": "20px",
                    "borderRadius": "12px",
                    "boxShadow": "0 2px 10px rgba(0,0,0,0.08)",
                },
            )
        else:
            domain_percent_fig = px.bar(
                domain_percent_df.sort_values("INCIDENT_RATE_PCT", ascending=False),
                x="DATA_DOMAIN",
                y="INCIDENT_RATE_PCT",
                title="Доля инцидентов от всех заявок по доменам",
                labels={
                    "DATA_DOMAIN": "Домен",
                    "INCIDENT_RATE_PCT": "Инциденты от всех заявок, %",
                },
                text="INCIDENT_RATE_PCT",
            )

            domain_percent_table_df = domain_percent_df.rename(
                columns={
                    "DATA_DOMAIN": "Домен",
                    "TOTAL_REQUESTS": "Всего заявок",
                    "TOTAL_OBJECTS": "Всего объектов",
                    "INCIDENTS": "Инцидентов",
                    "SLA_BREACHES": "SLA-просрочек",
                    "INCIDENT_RATE_PCT": "Инциденты от заявок, %",
                    "SLA_BREACH_RATE_PCT": "SLA-просрочки от заявок, %",
                    "AFFECTED_OBJECTS": "Затронуто объектов",
                    "AFFECTED_OBJECTS_SHARE_PCT": "Затронуто объектов, %",
                    "RISK_SCORE": "Индекс риска",
                    "RISK_SCORE_PER_1000_REQUESTS": "Индекс риска на 1000 заявок",
                    "AVG_FIX_TIME_HOURS": "Среднее устранение, ч",
                }
            )

            domain_percent_table = create_table(
                domain_percent_table_df,
                [
                    "Домен",
                    "Всего заявок",
                    "Инцидентов",
                    "Инциденты от заявок, %",
                    "SLA-просрочек",
                    "SLA-просрочки от заявок, %",
                    "Всего объектов",
                    "Затронуто объектов",
                    "Затронуто объектов, %",
                    "Индекс риска",
                    "Индекс риска на 1000 заявок",
                    "Среднее устранение, ч",
                ],
                max_rows=20,
            )

        # Таблица root cause
        root_cause_df = (
            df[df["IS_CASCADE_INCIDENT"] == 1]
            .groupby(
                [
                    "ROOT_CAUSE_REQ_ID",
                    "ROOT_CAUSE_DOMAIN",
                    "ROOT_CAUSE_OBJECT_NAME",
                    "ROOT_CAUSE_REASON_CATEGORY",
                ]
            )
            .agg(
                DEPENDENT_INCIDENTS=("ID", "count"),
                SLA_BREACHES=("SLA_FALL", "sum"),
                RISK_SCORE=("RISK_SCORE", "sum"),
                AFFECTED_OBJECTS=("OBJECT_NAME", "nunique"),
                MAX_DEPTH=("DEPENDENCY_DEPTH", "max"),
                AVG_FIX_TIME_HOURS=("FIX_TIME_HOURS", "mean"),
            )
            .reset_index()
            .sort_values(["DEPENDENT_INCIDENTS", "RISK_SCORE", "SLA_BREACHES"], ascending=False)
        )

        if not root_cause_df.empty:
            root_cause_df["AVG_FIX_TIME_HOURS"] = root_cause_df["AVG_FIX_TIME_HOURS"].round(1)

        root_cause_df = root_cause_df.rename(
            columns={
                "ROOT_CAUSE_REQ_ID": "REQ_ID корневой причины",
                "ROOT_CAUSE_DOMAIN": "Домен корневой причины",
                "ROOT_CAUSE_OBJECT_NAME": "Объект — корневая причина",
                "ROOT_CAUSE_REASON_CATEGORY": "Причина корневого сбоя",
                "DEPENDENT_INCIDENTS": "Зависимых инцидентов",
                "SLA_BREACHES": "SLA-просрочек",
                "RISK_SCORE": "Индекс риска",
                "AFFECTED_OBJECTS": "Затронуто объектов",
                "MAX_DEPTH": "Макс. глубина цепочки",
                "AVG_FIX_TIME_HOURS": "Среднее устранение, ч",
            }
        )

        root_cause_table = create_table(
            root_cause_df,
            [
                "REQ_ID корневой причины",
                "Домен корневой причины",
                "Объект — корневая причина",
                "Причина корневого сбоя",
                "Зависимых инцидентов",
                "SLA-просрочек",
                "Индекс риска",
                "Затронуто объектов",
                "Макс. глубина цепочки",
                "Среднее устранение, ч",
            ],
            max_rows=15,
        )

        # Таблица проблемных зон
        problem_areas_df = (
            df.groupby(["DATA_DOMAIN", "PROJECT_NAME", "OBJECT_NAME", "REASON_CATEGORY"])
            .agg(
                COUNT=("ID", "count"),
                SLA_BREACHES=("SLA_FALL", "sum"),
                CASCADE_INCIDENTS=("IS_CASCADE_INCIDENT", "sum"),
                RISK_SCORE=("RISK_SCORE", "sum"),
                AVG_FIX_TIME_HOURS=("FIX_TIME_HOURS", "mean"),
            )
            .reset_index()
            .sort_values(["RISK_SCORE", "COUNT"], ascending=False)
        )

        problem_areas_df["AVG_FIX_TIME_HOURS"] = problem_areas_df["AVG_FIX_TIME_HOURS"].round(1)

        problem_areas_df = problem_areas_df.rename(
            columns={
                "DATA_DOMAIN": "Домен",
                "PROJECT_NAME": "Проект",
                "OBJECT_NAME": "ETL-объект",
                "REASON_CATEGORY": "Причина",
                "COUNT": "Инцидентов",
                "SLA_BREACHES": "SLA-просрочек",
                "CASCADE_INCIDENTS": "Каскадных",
                "RISK_SCORE": "Индекс риска",
                "AVG_FIX_TIME_HOURS": "Среднее устранение, ч",
            }
        )

        problem_table = create_table(
            problem_areas_df,
            [
                "Домен",
                "Проект",
                "ETL-объект",
                "Причина",
                "Инцидентов",
                "SLA-просрочек",
                "Каскадных",
                "Индекс риска",
                "Среднее устранение, ч",
            ],
            max_rows=15,
        )

        # Повторяющиеся инциденты
        repeated_df = (
            df.groupby(["OBJECT_NAME", "REASON_CATEGORY"])
            .agg(
                REPEAT_COUNT=("ID", "count"),
                SLA_BREACHES=("SLA_FALL", "sum"),
                RISK_SCORE=("RISK_SCORE", "sum"),
            )
            .reset_index()
            .sort_values(["REPEAT_COUNT", "RISK_SCORE"], ascending=False)
        )

        repeated_df = repeated_df[repeated_df["REPEAT_COUNT"] > 1]

        repeated_df = repeated_df.rename(
            columns={
                "OBJECT_NAME": "ETL-объект",
                "REASON_CATEGORY": "Причина",
                "REPEAT_COUNT": "Количество повторов",
                "SLA_BREACHES": "SLA-просрочек",
                "RISK_SCORE": "Индекс риска",
            }
        )

        repeated_table = create_table(
            repeated_df,
            [
                "ETL-объект",
                "Причина",
                "Количество повторов",
                "SLA-просрочек",
                "Индекс риска",
            ],
            max_rows=15,
        )

        figures = [
            weekly_fig,
            sla_status_fig,
            domain_fig,
            reason_fig,
            object_fig,
            root_cause_fig,
            fix_time_fig,
            risk_domain_fig,
            reason_top_fig,
            domain_percent_fig,
        ]

        figures = [style_figure(fig) for fig in figures]

        (
            weekly_fig,
            sla_status_fig,
            domain_fig,
            reason_fig,
            object_fig,
            root_cause_fig,
            fix_time_fig,
            risk_domain_fig,
            reason_top_fig,
            domain_percent_fig,
        ) = figures

        return (
            "",
            kpi_cards,

            weekly_fig,
            sla_status_fig,
            domain_fig,
            reason_fig,
            object_fig,
            root_cause_fig,
            fix_time_fig,
            risk_domain_fig,

            reason_top_fig,
            reason_top_table,

            domain_percent_fig,
            domain_percent_table,

            root_cause_table,
            problem_table,
            repeated_table,
        )

    except Exception as error:
        error_fig = create_empty_figure("Ошибка загрузки данных")

        return (
            f"Ошибка: {error}",
            [],

            error_fig, error_fig, error_fig, error_fig,
            error_fig, error_fig, error_fig, error_fig,

            error_fig, "",

            error_fig, "",

            "", "", "",
        )


# ============================================================
# 10. Локальный запуск
# ============================================================

if __name__ == "__main__":
    app.run(debug=True)
