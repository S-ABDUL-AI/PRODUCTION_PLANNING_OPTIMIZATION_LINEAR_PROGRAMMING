import json
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from model import load_data, solve_production_plan, save_solution

st.set_page_config(
    page_title="Production Planning — Optimization Console",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

_TRUST_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"]  {
        font-family: 'Inter', 'Segoe UI', Roboto, Helvetica, Arial, sans-serif !important;
    }
    .block-container { padding-top: 1rem; max-width: 100%; }
    div[data-testid="stMetricValue"] { font-size: 1.45rem; font-weight: 600; color: #253858; }
    h1 { color: #0052CC !important; font-weight: 700 !important; }
    h2, h3 { color: #253858 !important; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="stToolbar"] {visibility: hidden;}
</style>
"""
st.markdown(_TRUST_CSS, unsafe_allow_html=True)

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

if not os.path.exists("data/products.csv"):
    with open("data/products.csv", "w") as f:
        f.write(sample_products)
if not os.path.exists("data/resources.csv"):
    with open("data/resources.csv", "w") as f:
        f.write(sample_resources)
if not os.path.exists("data/bom.csv"):
    with open("data/bom.csv", "w") as f:
        f.write(sample_bom)


@st.cache_data(show_spinner=False)
def cached_load_sample():
    return load_data("data/products.csv", "data/resources.csv", "data/bom.csv")


def get_dfs(use_sample, products_file, resources_file, bom_file):
    if use_sample:
        return cached_load_sample()
    if not products_file or not resources_file or not bom_file:
        return None, None, None
    return load_data(products_file, resources_file, bom_file)


st.title("Production Planning — LP Optimization Console")
st.caption(
    "Maximize profit subject to BOM and resource constraints. "
    "All inputs and the **Run optimization** control are in the sidebar."
)

# --- Sidebar ---
st.sidebar.header("Data")
use_sample = st.sidebar.checkbox("Use bundled sample CSVs in `/data`", value=True)

products_file = st.sidebar.file_uploader("products.csv", type=["csv"])
resources_file = st.sidebar.file_uploader("resources.csv", type=["csv"])
bom_file = st.sidebar.file_uploader("bom.csv", type=["csv"])

st.sidebar.header("Solver")
integer_vars = st.sidebar.checkbox("Integer production quantities", value=False)
time_limit = st.sidebar.number_input("Time limit (seconds, 0 = default)", min_value=0, max_value=600, value=0, step=5)

run_opt = st.sidebar.button("Run optimization", type="primary", use_container_width=True)

st.sidebar.divider()
st.sidebar.subheader("About")
st.sidebar.markdown(
    """
**Sherriff Abdul-Hamid**  
AI Engineer · Data Scientist · Economist  

[GitHub](https://github.com/S-ABDUL-AI) ·
[LinkedIn](https://www.linkedin.com/in/abdul-hamid-sherriff-08583354/)  
📧 Sherriffhamid001@gmail.com
"""
)

products_df, resources_df, bom_df = get_dfs(use_sample, products_file, resources_file, bom_file)

if products_df is None or resources_df is None or bom_df is None:
    st.warning("Upload **products.csv**, **resources.csv**, and **bom.csv** in the sidebar, or use sample data.")
    st.stop()

ceiling_revenue = float((products_df["max_demand"] * products_df["price"]).sum())
n_bom = len(bom_df)

# --- Structural KPI row (always visible) ---
s1, s2, s3, s4 = st.columns(4)
s1.metric("SKU count", f"{len(products_df):,}")
s2.metric("Resource types", f"{len(resources_df):,}")
s3.metric("BOM lines", f"{n_bom:,}")
s4.metric(
    "Demand-ceiling revenue",
    f"${ceiling_revenue:,.0f}",
    help="Σ(max_demand × price) — upper bound if capacity were unlimited.",
)

st.divider()

st.subheader("Input parameters")
st.caption("Review CSV-driven inputs before running the solver.")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("**Products**")
    st.dataframe(products_df, use_container_width=True, height=220)
with c2:
    st.markdown("**Resources**")
    st.dataframe(resources_df.reset_index(), use_container_width=True, height=220)
with c3:
    st.markdown("**Bill of materials**")
    st.dataframe(bom_df, use_container_width=True, height=220)

st.download_button(
    "Download inputs snapshot (JSON bundle)",
    data=json.dumps(
        {
            "products": products_df.to_csv(index=False),
            "resources": resources_df.reset_index().to_csv(index=False),
            "bom": bom_df.to_csv(index=False),
        },
        indent=0,
    ).encode("utf-8"),
    file_name="inputs_snapshot.json",
    mime="application/json",
)

with st.expander("Technical methodology & assumptions"):
    st.markdown(
        """
- **Formulation:** Linear program (mixed-integer optional) maximizing revenue minus procurement cost of materials,
  subject to BOM consumption and resource availability caps.
- **Data:** `products` (price, min/max demand), `resources` (availability, unit cost), `bom` (units of resource per product).
- **Integrality:** Continuous relaxation may overstate feasibility for discrete units; enable integer variables for discrete lots.
- **Validation:** Compare shadow prices / duals in advanced workflows; this UI surfaces primal plan and utilization only.
        """
    )

st.divider()

st.subheader("Visual analysis")
if not run_opt:
    st.info("Configure the model in the sidebar, then click **Run optimization**.")
    st.stop()

with st.spinner("Solving linear program…"):
    sol = solve_production_plan(
        products_df,
        resources_df,
        bom_df,
        integer=integer_vars,
        time_limit=(int(time_limit) if time_limit > 0 else None),
    )

status = sol.get("status", "Undefined")
if status == 1 and "plan" in sol:
    st.success("**Go:** Optimal solution found — review utilization and downloads for hand-off.")
elif "plan" not in sol:
    st.error("**No-go:** Solver did not return a feasible production plan. Check constraints and data.")
    st.stop()
else:
    st.warning(f"**Check status:** solver returned status code `{status}` — validate results before production use.")

profit = sol["profit"]
revenue = sol["total_revenue"]
mat_cost = sol["total_material_cost"]
margin_pct = (profit / revenue * 100.0) if revenue else 0.0
ceiling_capture = (revenue / ceiling_revenue * 100.0) if ceiling_revenue else 0.0

plan_df = pd.DataFrame([{"product": p, "quantity": q} for p, q in sol["plan"].items()]).sort_values("product")

res_usage_df = pd.DataFrame(
    [
        {
            "resource": r,
            "used": sol["resource_usage"][r],
            "available": float(resources_df.loc[r, "available"]),
            "utilization_%": sol["resource_utilization_pct"][r],
        }
        for r in sol["resource_usage"].keys()
    ]
).sort_values("utilization_%", ascending=False)
avg_util = float(res_usage_df["utilization_%"].mean()) if len(res_usage_df) else 0.0

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric(
        "Predicted profit",
        f"${profit:,.2f}",
        delta=f"{margin_pct:.1f}% margin",
        help="Revenue minus material cost at optimal quantities.",
    )
with k2:
    st.metric(
        "Revenue (optimal)",
        f"${revenue:,.2f}",
        delta=f"{ceiling_capture:.1f}% of demand-ceiling revenue",
        help="Realized revenue vs Σ(max_demand × price).",
    )
with k3:
    st.metric("Material cost", f"${mat_cost:,.2f}", delta_color="inverse")
with k4:
    st.metric(
        "Avg resource utilization",
        f"{avg_util:.1f}%",
        delta="Bottleneck signal" if avg_util > 85 else "Headroom available",
        delta_color="off",
    )

fig1 = go.Figure()
fig1.add_trace(go.Bar(x=plan_df["product"], y=plan_df["quantity"], marker_color="#0052CC", name="Units"))
fig1.update_layout(
    title="Optimal production quantities",
    xaxis_title="Product",
    yaxis_title="Units",
    template="plotly_white",
    height=420,
)
st.plotly_chart(fig1, use_container_width=True)
_dl1, _dl2 = st.columns([1, 5])
with _dl1:
    st.download_button(
        "Download chart data (plan CSV)",
        data=plan_df.to_csv(index=False).encode("utf-8"),
        file_name="plan_quantities.csv",
        mime="text/csv",
    )

fig2 = go.Figure()
fig2.add_trace(
    go.Bar(
        x=res_usage_df["resource"],
        y=res_usage_df["utilization_%"],
        marker_color="#253858",
        name="Utilization %",
    )
)
fig2.update_layout(
    title="Resource utilization",
    xaxis_title="Resource",
    yaxis_title="Utilization (%)",
    template="plotly_white",
    height=420,
)
st.plotly_chart(fig2, use_container_width=True)
_dl3, _dl4 = st.columns([1, 5])
with _dl3:
    st.download_button(
        "Download utilization CSV",
        data=res_usage_df.to_csv(index=False).encode("utf-8"),
        file_name="resource_utilization.csv",
        mime="text/csv",
    )

st.divider()

st.subheader("Raw data & exports")
st.markdown("#### Optimal plan")
st.dataframe(plan_df, use_container_width=True)
st.markdown("#### Resource usage")
st.dataframe(res_usage_df, use_container_width=True)

plan_path = "outputs/plan.csv"
usage_path = "outputs/resource_usage.csv"
save_solution(sol, "outputs/solution.json")
plan_df.to_csv(plan_path, index=False)
res_usage_df.to_csv(usage_path, index=False)

b1, b2, b3 = st.columns(3)
with b1:
    st.download_button("Download plan (CSV)", data=plan_df.to_csv(index=False), file_name="plan.csv", mime="text/csv")
with b2:
    st.download_button(
        "Download resource usage (CSV)",
        data=res_usage_df.to_csv(index=False),
        file_name="resource_usage.csv",
        mime="text/csv",
    )
with b3:
    st.download_button(
        "Download full solution (JSON)",
        data=json.dumps(sol, indent=2, default=str),
        file_name="solution.json",
        mime="application/json",
    )
