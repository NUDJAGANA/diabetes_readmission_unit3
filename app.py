import os
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
        page_title="Diabetes Hospital Readmission Analysis",
        page_icon="🏥",
        layout="wide",
)

st.markdown(
        """
        <style>
        .stMetric, div[data-testid="stMetric"] {
            background-color: #1E293B;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }
        </style>
        """,
        unsafe_allow_html=True,
)


@st.cache_data
def load_data():
    path = "data/processed/cleaned_diabetic_data.csv"

    if os.path.exists(path):
        return pd.read_csv(path)

    if os.path.exists("cleaned_diabetic_data.csv"):
        return pd.read_csv("cleaned_diabetic_data.csv")

    return None


@st.cache_resource
def load_model():
    model_path = "data/processed/prediction_model_6_features.joblib"

    if not os.path.exists(model_path):
        return None

    return joblib.load(model_path)


df = load_data()
prediction_artifact = load_model()

if df is None:
    st.error(
            "⚠️ **File Not Found:** Could not locate "
            "`cleaned_diabetic_data.csv` in `data/processed/`."
    )
    st.stop()


st.title("🏥 Diabetes Hospital Readmission Analysis")
st.markdown(
        "### Readmission Risk Analytics, Population Insights & ML Prediction"
)
st.write("---")


st.sidebar.header("Navigation & Controls")
app_mode = st.sidebar.radio(
        "Select View",
        [
            "📊 Population Dashboard",
            "🤖 Patient Risk Prediction Simulator",
        ],
)

st.sidebar.write("---")
st.sidebar.header("🔍 Global Cohort Filters")


if "age" in df.columns:
    unique_ages = sorted(df["age"].dropna().unique().tolist())
    selected_ages = st.sidebar.multiselect(
            "Filter Age Groups",
            options=unique_ages,
            default=unique_ages,
    )
    df_filtered = df[df["age"].isin(selected_ages)]
else:
    df_filtered = df.copy()


if "time_in_hospital" in df_filtered.columns and not df_filtered.empty:
    max_days = int(df_filtered["time_in_hospital"].max())
    min_days = int(df_filtered["time_in_hospital"].min())

    selected_time = st.sidebar.slider(
            "Max Time in Hospital (Days)",
            min_value=min_days,
            max_value=max_days if max_days > min_days else min_days + 1,
            value=max_days,
    )
    df_filtered = df_filtered[
            df_filtered["time_in_hospital"] <= selected_time
    ]


if app_mode == "📊 Population Dashboard":
    total_patients = len(df_filtered)

    if total_patients > 0 and "readmitted_30" in df_filtered.columns:
        readmit_rate = (
                df_filtered["readmitted_30"].sum()
                / total_patients
                * 100
        )
    else:
        readmit_rate = 0.0

    if "readmitted_30" in df_filtered.columns:
        high_risk_count = df_filtered[
                df_filtered["readmitted_30"] == 1
        ].shape[0]
    else:
        high_risk_count = 0

    penalty_avoidance = high_risk_count * 3000

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
                label="Actual Readmit Rate",
                value=f"{readmit_rate:.1f}%",
                delta="-1.2% vs target",
                delta_color="inverse",
        )

    with col2:
        st.metric(
                label="High Risk Patients Flagged",
                value=f"{high_risk_count:,}",
                delta="Action Required",
                delta_color="off",
        )

    with col3:
        st.metric(
                label="Estimated Penalty Avoidance",
                value=f"${penalty_avoidance:,.0f}",
                delta="Projected Savings",
        )

    with col4:
        st.metric(
                label="Total Patients (Filtered)",
                value=f"{total_patients:,}",
        )

    st.write("")
    mid_col1, mid_col2 = st.columns(2)

    with mid_col1:
        st.markdown("#### Population Distribution by Age Tier")

        if "age" in df_filtered.columns and not df_filtered.empty:
            age_counts = df_filtered["age"].value_counts().reset_index()
            age_counts.columns = ["age", "count"]

            fig_doughnut = px.pie(
                    age_counts,
                    names="age",
                    values="count",
                    hole=0.5,
                    color_discrete_sequence=px.colors.sequential.Tealgrn,
            )
            fig_doughnut.update_layout(
                    paper_bgcolor="#1E293B",
                    font_color="#FFFFFF",
                    margin=dict(t=20, b=20, l=20, r=20),
                    legend=dict(
                            orientation="v",
                            yanchor="middle",
                            y=0.5,
                    ),
            )
            st.plotly_chart(
                    fig_doughnut,
                    use_container_width=True,
            )
        else:
            st.warning("No data available for the selected filters.")

    with mid_col2:
        st.markdown(
                "#### 30-Day Readmission Rate by Prior Inpatient Visits"
        )
        required_columns = ["number_inpatient", "readmitted_30"]

        if all(
                column in df_filtered.columns
                for column in required_columns
        ):
            chart_data = df_filtered[required_columns].dropna().copy()

            chart_data["prior_visit_group"] = pd.cut(
                    chart_data["number_inpatient"],
                    bins=[-1, 0, 1, 2, float("inf")],
                    labels=["0", "1", "2", "3 or more"],
            )

            summary = (
                    chart_data.groupby(
                            "prior_visit_group",
                            observed=False,
                    )
                    .agg(
                            encounters=("readmitted_30", "size"),
                            readmission_rate=("readmitted_30", "mean"),
                    )
                    .reset_index()
            )
            summary = summary[summary["encounters"] > 0]

            if not summary.empty:
                fig_bar = px.bar(
                        summary,
                        x="prior_visit_group",
                        y="readmission_rate",
                        text="readmission_rate",
                        labels={
                                "prior_visit_group": "Prior inpatient visits",
                                "readmission_rate": "30-day readmission rate",
                        },
                        title="Readmission rate by prior inpatient visits",
                )

                fig_bar.update_traces(
                        texttemplate="%{y:.1%}",
                        textposition="outside",
                        customdata=summary[["encounters"]],
                        hovertemplate=(
                                "Prior inpatient visits: %{x}<br>"
                                "30-day readmission rate: %{y:.1%}<br>"
                                "Encounters: %{customdata[0]:,}"
                                "<extra></extra>"
                        ),
                )

                fig_bar.update_layout(
                        yaxis_tickformat=".0%",
                        yaxis_title="Readmitted within 30 days",
                        xaxis_title=(
                                "Prior inpatient visits in the previous year"
                        ),
                        showlegend=False,
                        margin=dict(t=60, b=40, l=40, r=20),
                )

                st.plotly_chart(
                        fig_bar,
                        use_container_width=True,
                )

                accessible_table = summary.copy()
                accessible_table["readmission_rate"] = (
                        accessible_table["readmission_rate"].map(
                                lambda rate: f"{rate:.1%}"
                        )
                )
                accessible_table.columns = [
                        "Prior inpatient visits",
                        "Number of encounters",
                        "30-day readmission rate",
                ]

                st.dataframe(
                        accessible_table,
                        hide_index=True,
                        use_container_width=True,
                )

                st.caption(
                        "This chart shows an association in historical data. "
                        "It does not show that prior inpatient visits cause "
                        "readmission."
                )
            else:
                st.warning(
                        "There is no data to display for these filters."
                )
        else:
            st.warning(
                    "The prior inpatient visit or readmission outcome "
                    "column is missing."
            )

    st.write("---")
    st.markdown("#### 📋 Patient Scorecard and Export")

    def highlight_risk(row):
        if "readmitted_30" in row and row["readmitted_30"] == 1:
            return [
                    "background-color: #991B1B; color: white"
            ] * len(row)

        return [
                "background-color: #1E293B; color: white"
        ] * len(row)

    display_cols = [
            column
            for column in [
                "age",
                "admission_type_id",
                "time_in_hospital",
                "num_medications",
                "readmitted_30",
            ]
            if column in df_filtered.columns
    ]

    if display_cols and not df_filtered.empty:
        subset_df = df_filtered[display_cols].head(100)
        styled_df = subset_df.style.apply(
                highlight_risk,
                axis=1,
        ).format(na_rep="-")

        st.dataframe(
                styled_df,
                use_container_width=True,
        )

        csv_data = subset_df.to_csv(index=False).encode("utf-8")
        st.download_button(
                label="📥 Download Filtered Patient Scorecard (CSV)",
                data=csv_data,
                file_name="filtered_patient_scorecard.csv",
                mime="text/csv",
        )
    else:
        st.warning("No records match the selected filters.")


elif app_mode == "🤖 Patient Risk Prediction Simulator":
    st.markdown("### 30-Day Readmission Model Demonstration")
    st.write(
            "Enter the six features used by this model to see an educational "
            "estimate. This is not a clinical prediction or care recommendation."
    )

    expected_features = [
            "time_in_hospital",
            "num_medications",
            "number_inpatient",
            "number_outpatient",
            "number_emergency",
            "num_lab_procedures",
    ]

    model = None
    feature_columns = expected_features
    threshold = 0.5
    metrics = {}

    if prediction_artifact is None:
        st.warning(
                "The compatible six-input model is not available. Run "
                "`train_prediction_model.py` from the project root to create "
                "it. No simulated result will be shown."
        )
    else:
        model = prediction_artifact.get("model")
        feature_columns = prediction_artifact.get(
                "feature_columns",
                [],
        )
        threshold = float(
                prediction_artifact.get("optimal_threshold", 0.5)
        )
        metrics = prediction_artifact.get("metrics", {})

        if model is None or feature_columns != expected_features:
            st.error(
                    "The saved model does not match the six inputs shown here. "
                    "Retrain it with `train_prediction_model.py`."
            )
            st.stop()

        if metrics:
            st.caption(
                    "Held-out test results for this model: "
                    f"ROC-AUC "
                    f"{metrics.get('test_roc_auc', float('nan')):.3f}; "
                    f"precision "
                    f"{metrics.get('test_precision', float('nan')):.3f}; "
                    f"recall "
                    f"{metrics.get('test_recall', float('nan')):.3f}. "
                    "These are dataset evaluation metrics, not a guarantee "
                    "for an individual."
            )

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            time_in_hospital = st.slider(
                    "Time in Hospital (Days)",
                    1,
                    14,
                    3,
            )
            num_medications = st.slider(
                    "Number of Medications",
                    1,
                    40,
                    15,
            )

        with col2:
            number_inpatient = st.slider(
                    "Prior Inpatient Visits",
                    0,
                    10,
                    1,
            )
            number_outpatient = st.slider(
                    "Prior Outpatient Visits",
                    0,
                    10,
                    0,
            )

        with col3:
            number_emergency = st.slider(
                    "Prior Emergency Visits",
                    0,
                    10,
                    0,
            )
            num_lab_procedures = st.slider(
                    "Number of Lab Procedures",
                    1,
                    120,
                    43,
            )

        submitted = st.form_submit_button(
                "Run Risk Prediction",
                disabled=model is None,
        )

        if submitted:
            input_data = pd.DataFrame(
                    {
                        "time_in_hospital": [time_in_hospital],
                        "num_medications": [num_medications],
                        "number_inpatient": [number_inpatient],
                        "number_outpatient": [number_outpatient],
                        "number_emergency": [number_emergency],
                        "num_lab_procedures": [num_lab_procedures],
                    }
            )

            try:
                input_data = input_data[feature_columns]
                probability = float(
                        model.predict_proba(input_data)[0][1]
                )

                st.write("---")
                st.metric(
                        "Estimated probability in this model",
                        f"{probability * 100:.1f}%",
                )

                if probability >= threshold:
                    st.info(
                            "The estimate is at or above the model's "
                            f"validation-selected threshold ({threshold:.2f})."
                    )
                else:
                    st.info(
                            "The estimate is below the model's "
                            f"validation-selected threshold ({threshold:.2f})."
                    )

                st.caption(
                        "For educational analysis only. Do not use this result "
                        "to make decisions about an individual patient's care."
                )
            except Exception as error:
                st.error(
                        f"The model could not generate a prediction: {error}"
                )