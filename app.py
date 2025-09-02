import os
import json
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from model import load_data, solve_production_plan, save_solution

st.set_page_config(page_title="🏭 Production Planning Optimization", layout="wide")

# --- Ensure data folder and sample CSVs exist ---
os.makedirs("data", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

sample_products = """product,price,min_demand,max_demand
WidgetA,120,0,500
WidgetB,150,0,400
WidgetC,200,0,300
"""

sample_resources = """resource,available,unit_cost
RawMaterial1,1000,10
RawMaterial2,800,8
MachineHours,1200,5
"""

sample_bom = """product,resource,units_required
WidgetA,RawMaterial1,2
WidgetA,MachineHours,1
WidgetB,RawMaterial2,3
WidgetB,MachineHours,2
WidgetC,RawMaterial1,1
WidgetC,RawMaterial2,1
"""

# Save files if missing
if not os.path.exists("data/products.csv"):
    with open("data/products.csv", "w") as f:
        f.write(sample_products)
if not os.path.exists("data/resources.csv"):
    with open("data/resources.csv", "w") as f:
        f.write(sample_resources)
if not os.path.exists("data/bom.csv"):
    with open("data/bom.csv", "w") as f:
        f.write(sample_bom)

# --- Styles ---
st.markdown(
    """
    <style>
      .kpi-card {
        padding: 14px 16px;
        border-radius: 14px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        background: #ffffff;
        border-left: 6px solid #6366F1;
      }
      .kpi-value { font-size: 1.4rem; font-weight: 700; margin-top: 2px; }
      .kpi-label { color: #6b7280; font-size: 0.85rem; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🏭 Production Planning Optimization (Linear Programming)")

# --- Sidebar ---
st.sidebar.header("⚙️ Settings")
use_sample = st.sidebar.checkbox("Use sample data (in /data)", value=True)

products_file = st.sidebar.file_uploader("Upload products.csv", type=["csv"])
resources_file = st.sidebar.file_uploader("Upload resources.csv", type=["csv"])
bom_file = st.sidebar.file_uploader("Upload bom.csv", type=["csv"])

integer_vars = st.sidebar.checkbox("Force integer production quantities", value=False)
time_limit = st.sidebar.number_input("Solver time limit (sec)", min_value=0, max_value=600, value=0, step=5)

st.sidebar.markdown("---")
st.sidebar.subheader("👨‍💻 About the Developer")
st.sidebar.markdown(
    """
**Sherriff Abdul-Hamid**  
AI Engineer | Data Scientist/Analyst | Economist  

**Contact:**  
[GitHub](https://github.com/S-ABDUL-AI) •
[LinkedIn](https://www.linkedin.com/in/abdul-hamid-sherriff-08583354/)  
📧 Sherriffhamid001@gmail.com
"""
)

# --- Load data ---
def get_dfs():
    if use_sample:
        p = "data/products.csv"
        r = "data/resources.csv"
        b = "data/bom.csv"
        return load_data(p, r, b)
    else:
        if not products_file or not resources_file or not bom_file:
            st.warning("Upload all three files: products.csv, resources.csv, bom.csv")
            return None, None, None
        return load_data(products_file, resources_file, bom_file)

products_df, resources_df, bom_df = get_dfs()

# Stop if data is missing
if products_df is None or resources_df is None or bom_df is None:
    st.stop()

# --- Preview data ---
col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("Products")
    st.dataframe(products_df, use_container_width=True, height=220)
with col2:
    st.subheader("Resources")
    st.dataframe(resources_df, use_container_width=True, height=220)
with col3:
    st.subheader("BOM")
    st.dataframe(bom_df, use_container_width=True, height=220)

# --- Solve ---
if st.button("🚀 Optimize Plan", type="primary", use_container_width=True):
    with st.spinner("Solving optimization model..."):
        sol = solve_production_plan(
            products_df, resources_df, bom_df,
            integer=integer_vars,
            time_limit=(int(time_limit) if time_limit > 0 else None)
        )

    # --- Solver Status Banner ---
    status = sol.get("status", "Undefined")
    if status == 1:  # Optimal
        st.markdown(
            """
            <div style="background-color:#d4edda; padding:18px; border-radius:12px; margin-bottom:25px;">
                <h3 style="color:#155724; margin:0;">✅ Solver status: Optimal solution found!</h3>
            </div>
            """,
            unsafe_allow_html=True
        )
    elif "plan" not in sol:
        st.markdown(
            """
            <div style="background-color:#f8d7da; padding:18px; border-radius:12px; margin-bottom:25px;">
                <h3 style="color:#721c24; margin:0;">❌ Solver failed: No feasible plan found.</h3>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.stop()
    else:
        st.markdown(
            f"""
            <div style="background-color:#fff3cd; padding:18px; border-radius:12px; margin-bottom:25px;">
                <h3 style="color:#856404; margin:0;">⚠️ Solver status: {status}</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

    # --- KPIs ---
    profit = sol["profit"]
    revenue = sol["total_revenue"]
    mat_cost = sol["total_material_cost"]

    k1, k2, k3 = st.columns(3)
    with k1:
        st.markdown('<div class="kpi-card"><div class="kpi-label">Total Profit</div>'
                    f'<div class="kpi-value">${profit:,.2f}</div></div>', unsafe_allow_html=True)
        st.caption("Profit = Revenue − Material Cost")
    with k2:
        st.markdown('<div class="kpi-card"><div class="kpi-label">Total Revenue</div>'
                    f'<div class="kpi-value">${revenue:,.2f}</div></div>', unsafe_allow_html=True)
    with k3:
        st.markdown('<div class="kpi-card"><div class="kpi-label">Material Cost</div>'
                    f'<div class="kpi-value">${mat_cost:,.2f}</div></div>', unsafe_allow_html=True)

    st.markdown("#### Units & Conventions")
    st.markdown("- **Production quantities**: units of finished goods\n"
                "- **Materials**: units (e.g., kg)\n"
                "- **Machines**: hours\n"
                "- **Currency**: as per `price` and `unit_cost` values in CSVs")

    # --- Plan Table ---
    plan_df = pd.DataFrame([
        {"product": p, "quantity": q} for p, q in sol["plan"].items()
    ]).sort_values("product")
    st.subheader("📦 Optimal Production Plan")
    st.dataframe(plan_df, use_container_width=True)

    # --- Resource Usage ---
    res_usage_df = pd.DataFrame([
        {
            "resource": r,
            "used": sol["resource_usage"][r],
            "available": float(resources_df.loc[r, "available"]),
            "utilization_%": sol["resource_utilization_pct"][r]
        }
        for r in sol["resource_usage"].keys()
    ]).sort_values("utilization_%", ascending=False)

    st.subheader("🔧 Resource Usage & Utilization")
    st.dataframe(res_usage_df, use_container_width=True)

    # --- Bar charts ---
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(x=plan_df["product"], y=plan_df["quantity"], name="Qty"))
    fig1.update_layout(title="Production Quantities", xaxis_title="Product", yaxis_title="Units")
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=res_usage_df["resource"], y=res_usage_df["utilization_%"], name="Utilization %"))
    fig2.update_layout(title="Resource Utilization (%)", xaxis_title="Resource", yaxis_title="%")
    st.plotly_chart(fig2, use_container_width=True)

    # --- Save outputs ---
    plan_path = "outputs/plan.csv"
    usage_path = "outputs/resource_usage.csv"
    sol_path = save_solution(sol, "outputs/solution.json")
    plan_df.to_csv(plan_path, index=False)
    res_usage_df.to_csv(usage_path, index=False)

    # --- Downloads ---
    st.download_button("📥 Download Plan (CSV)", data=plan_df.to_csv(index=False), file_name="plan.csv", mime="text/csv")
    st.download_button("📥 Download Resource Usage (CSV)", data=res_usage_df.to_csv(index=False), file_name="resource_usage.csv", mime="text/csv")
    st.download_button("📥 Download Full Solution (JSON)", data=json.dumps(sol, indent=2), file_name="solution.json", mime="application/json")

else:
    st.info("Set options in the sidebar, review the data, then click **Optimize Plan**.")
