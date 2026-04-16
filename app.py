import io
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


def _total_material_cost(plan: dict, bom_df: pd.DataFrame, resources_df: pd.DataFrame) -> float:
    """Sum procurement cost for a production mix (same accounting as the LP objective)."""
    total = 0.0
    for p, q in plan.items():
        if q is None:
            continue
        qf = float(q)
        for r in resources_df.index:
            sub = bom_df.loc[(bom_df["product"] == p) & (bom_df["resource"] == r)]
            if sub.empty:
                continue
            u = float(sub["units_required"].iloc[0])
            total += qf * u * float(resources_df.loc[r, "unit_cost"])
    return total


def _total_revenue_from_plan(plan: dict, products_df: pd.DataFrame) -> float:
    """Revenue Σ price_i × q_i for a quantity vector."""
    pid = products_df.set_index("product")
    total = 0.0
    for p, q in plan.items():
        if q is None:
            continue
        if p not in pid.index:
            continue
        total += float(q) * float(pid.loc[p, "price"])
    return total


def _max_uniform_scale_feasible(products_df: pd.DataFrame, resources_df: pd.DataFrame, bom_df: pd.DataFrame) -> float:
    """Largest s in [0, 1] such that producing s * max_demand for every SKU stays within resource caps."""

    def feasible(s: float) -> bool:
        if s <= 0:
            return True
        for r in resources_df.index:
            used = 0.0
            for _, row in products_df.iterrows():
                p = row["product"]
                q = float(row["max_demand"]) * s
                sub = bom_df.loc[(bom_df["product"] == p) & (bom_df["resource"] == r)]
                if sub.empty:
                    continue
                used += q * float(sub["units_required"].iloc[0])
            if used > float(resources_df.loc[r, "available"]) + 1e-5:
                return False
        return True

    if feasible(1.0):
        return 1.0
    if not feasible(1e-9):
        return 0.0
    lo, hi = 0.0, 1.0
    for _ in range(56):
        mid = (lo + hi) / 2
        if feasible(mid):
            lo = mid
        else:
            hi = mid
    return lo


def _baseline_uniform_max_plan(
    products_df: pd.DataFrame, resources_df: pd.DataFrame, bom_df: pd.DataFrame, scale: float
) -> dict[str, float]:
    return {str(row["product"]): float(row["max_demand"]) * scale for _, row in products_df.iterrows()}


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

st.sidebar.header("What-if (stress test)")
cap_mult = st.sidebar.slider(
    "Resource capacity ×",
    min_value=0.5,
    max_value=1.5,
    value=1.0,
    step=0.05,
    help="Scales every resource availability before baseline + LP (scenario planning).",
)
cost_mult = st.sidebar.slider(
    "Resource unit cost ×",
    min_value=0.5,
    max_value=1.5,
    value=1.0,
    step=0.05,
    help="Scales procurement $/unit for every resource (cost inflation / deflation scenario).",
)

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

# Stressed resources (same LP + baseline accounting)
resources_work = resources_df.copy()
resources_work["available"] = (resources_work["available"] * cap_mult).astype(float)
resources_work["unit_cost"] = (resources_work["unit_cost"] * cost_mult).astype(float)

# Baseline push scenario (scaled max-demand mix) — computed before solve for KPI deltas
baseline_scale = _max_uniform_scale_feasible(products_df, resources_work, bom_df)
baseline_plan = _baseline_uniform_max_plan(products_df, resources_work, bom_df, baseline_scale)
baseline_material = _total_material_cost(baseline_plan, bom_df, resources_work)
baseline_revenue = _total_revenue_from_plan(baseline_plan, products_df)
baseline_profit = baseline_revenue - baseline_material

total_demand_units = float(products_df["max_demand"].sum())
total_capacity_units = float(resources_work["available"].sum())

# --- Executive scale KPIs (always visible) ---
s1, s2, s3, s4 = st.columns(4)
s1.metric(
    "Total demand (max)",
    f"{total_demand_units:,.0f}",
    delta=f"{len(products_df)} SKUs",
    delta_color="off",
    help="Sum of max_demand — scale of requested throughput.",
)
s2.metric(
    "Total capacity (Σ avail.)",
    f"{total_capacity_units:,.0f}",
    delta=f"×{cap_mult:.2f} capacity slider",
    delta_color="off",
    help="Sum of resource availability after the capacity multiplier.",
)
s3.metric(
    "Baseline material (ref.)",
    f"${baseline_material:,.0f}",
    delta=f"push @ {baseline_scale:.0%} of max",
    delta_color="off",
    help="Material $ for uniform max-demand mix scaled to feasibility — anchor for savings deltas after solve.",
)
s4.metric(
    "Demand-ceiling revenue",
    f"${ceiling_revenue:,.0f}",
    delta=f"{n_bom} BOM lines",
    delta_color="off",
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
    st.markdown("**Resources (after what-if multipliers)**")
    st.caption(f"Capacity ×{cap_mult:.2f} · Unit cost ×{cost_mult:.2f} — used for baseline + optimization.")
    st.dataframe(resources_work.reset_index(), use_container_width=True, height=220)
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

with st.expander("Objective & constraints (algorithm audit)", expanded=False):
    st.markdown(
        """
**Decision variables**  
- `q_i` = production quantity for SKU *i*, bounded by `min_demand_i ≤ q_i ≤ max_demand_i`.

**Objective (maximize)**  
- Sum of `price_i × q_i` **minus** total procurement spend: for each resource *r*,  
  `unit_cost_r × (sum over i of q_i × BOM[i,r])`.  
  i.e. **revenue minus material cost**.

**Constraints**  
- Each resource *r*: total consumption from the BOM **≤** `available_r` (after your **capacity ×** slider).  
- `BOM[i,r]` = units of resource *r* required per unit of product *i*.

**Baseline reference (for savings deltas)**  
- “Push” scenario: produce `s × max_demand_i` for every SKU, using the **largest** `s` in `[0, 1]` that keeps all resource constraints feasible; material $ uses the same **unit cost ×** multipliers as the LP.
        """
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
        resources_work,
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
            "available": float(resources_work.loc[r, "available"]),
            "utilization_%": sol["resource_utilization_pct"][r],
        }
        for r in sol["resource_usage"].keys()
    ]
).sort_values("utilization_%", ascending=False)
avg_util = float(res_usage_df["utilization_%"].mean()) if len(res_usage_df) else 0.0

if baseline_material > 1e-9:
    cost_reduction_pct = (baseline_material - mat_cost) / baseline_material * 100.0
else:
    cost_reduction_pct = 0.0
saved_usd = baseline_material - mat_cost
profit_vs_baseline = profit - baseline_profit

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric(
        "Predicted profit",
        f"${profit:,.2f}",
        delta=f"{margin_pct:.1f}% margin · Δ ${profit_vs_baseline:+,.0f} vs baseline push",
        delta_color="normal" if profit_vs_baseline > 1 else ("inverse" if profit_vs_baseline < -1 else "off"),
        help="LP optimum vs the same-stress reference plan (scaled max-demand mix).",
    )
with k2:
    st.metric(
        "Revenue (optimal)",
        f"${revenue:,.2f}",
        delta=f"{ceiling_capture:.1f}% of demand-ceiling revenue",
        help="Realized revenue vs Σ(max_demand × price).",
    )
with k3:
    st.metric(
        "Material cost (optimized)",
        f"${mat_cost:,.2f}",
        delta=(
            f"{cost_reduction_pct:.1f}% vs baseline (−${saved_usd:,.0f})"
            if saved_usd > 0.01
            else (
                f"+{-cost_reduction_pct:.1f}% vs baseline (+${-saved_usd:,.0f})"
                if saved_usd < -0.01
                else "Flat vs baseline"
            )
        ),
        delta_color="normal" if saved_usd > 0.01 else ("inverse" if saved_usd < -0.01 else "off"),
        help="Green when optimized procurement is below the scaled max-demand baseline.",
    )
with k4:
    st.metric(
        "Avg resource utilization",
        f"{avg_util:.1f}%",
        delta="Bottleneck signal" if avg_util > 85 else "Headroom available",
        delta_color="off",
    )

# --- Scenario comparison: baseline vs optimized material spend ---

st.subheader("Scenario comparison — material spend")
st.caption(
    "Baseline: each SKU at max demand, scaled uniformly by "
    f"{baseline_scale:.1%} until the mix is resource-feasible (status-quo push scenario). "
    "Optimized: profit-maximizing LP solution on the same inputs."
)
c_sc_a, c_sc_mid, c_sc_b = st.columns([1, 0.55, 1])
with c_sc_a:
    st.markdown("##### Baseline costs")
    st.metric(
        label="Total material spend",
        value=f"${baseline_material:,.2f}",
        help="Procurement cost under the scaled max-demand reference mix.",
    )
with c_sc_mid:
    st.markdown(
        f"<div style='text-align:center;padding-top:1.6rem;font-size:2rem;color:#0052CC;font-weight:700;'>→</div>",
        unsafe_allow_html=True,
    )
    st.metric(
        label="Cost vs baseline",
        value=f"{cost_reduction_pct:+.1f}%",
        delta=f"${saved_usd:,.2f} material" if abs(saved_usd) > 0.01 else "No change",
        delta_color="normal" if saved_usd > 0 else ("inverse" if saved_usd < 0 else "off"),
        help="Positive % = optimized plan uses less material than the baseline push scenario.",
    )
with c_sc_b:
    st.markdown("##### Optimized costs")
    st.metric(
        label="Total material spend",
        value=f"${mat_cost:,.2f}",
        help="Procurement cost at the optimal production plan.",
    )

if 12 <= cost_reduction_pct <= 18:
    st.success(
        f"**Executive headline:** material spend is down **~{cost_reduction_pct:.0f}%** vs the baseline "
        "push scenario — in line with common **10–15%** procurement efficiency targets in briefing decks."
    )
elif cost_reduction_pct > 0:
    st.info(
        f"**Material efficiency:** **{cost_reduction_pct:.1f}%** lower procurement cost than the baseline "
        f"scenario (~**${saved_usd:,.0f}** saved at this mix)."
    )
else:
    st.warning(
        "**Material spend:** the profit-optimal plan can **exceed** the naive baseline on procurement "
        "when revenue upside justifies higher input use — review BOM costs and margins by SKU."
    )

plan_compare = plan_df.copy()
plan_compare = plan_compare.rename(columns={"quantity": "optimized_qty"})
plan_compare["baseline_qty"] = plan_compare["product"].map(baseline_plan).astype(float)
plan_compare["qty_delta_vs_baseline"] = plan_compare["optimized_qty"] - plan_compare["baseline_qty"]

summary_exec = pd.DataFrame(
    [
        {
            "metric": "Baseline_total_material_USD",
            "value": round(baseline_material, 2),
        },
        {
            "metric": "Optimized_total_material_USD",
            "value": round(mat_cost, 2),
        },
        {
            "metric": "Cost_reduction_vs_baseline_pct",
            "value": round(cost_reduction_pct, 3),
        },
        {
            "metric": "Baseline_max_demand_uniform_scale",
            "value": round(baseline_scale, 6),
        },
        {
            "metric": "Total_profit_USD",
            "value": round(profit, 2),
        },
        {
            "metric": "Total_revenue_USD",
            "value": round(revenue, 2),
        },
    ]
)

buf_exec = io.StringIO()
buf_exec.write("# EXECUTIVE_REVIEW — scenario snapshot (append to board pack)\n")
summary_exec.to_csv(buf_exec, index=False)
buf_exec.write("\n# PRODUCT_PLAN — baseline vs optimized quantities\n")
plan_compare.to_csv(buf_exec, index=False)
buf_exec.write("\n# RESOURCE_UTILIZATION\n")
res_usage_df.to_csv(buf_exec, index=False)
exec_csv_bytes = buf_exec.getvalue().encode("utf-8")

st.download_button(
    label="Download executive review (CSV)",
    data=exec_csv_bytes,
    file_name="executive_review_supply_chain_scenario.csv",
    mime="text/csv",
    help="One file: KPI snapshot, plan comparison (baseline vs optimized qty), and utilization for audit.",
)

fig_cost = go.Figure()
fig_cost.add_trace(
    go.Bar(
        x=["Baseline push (material)", "Optimized LP (material)"],
        y=[baseline_material, mat_cost],
        marker_color=["#64748b", "#0052CC"],
        text=[f"${baseline_material:,.0f}", f"${mat_cost:,.0f}"],
        textposition="outside",
        textfont=dict(size=13, color="#253858"),
    )
)
fig_cost.update_layout(
    title="Cost story — procurement spend vs baseline reference",
    yaxis_title="USD",
    xaxis=dict(tickangle=0, tickfont=dict(size=13, color="#253858")),
    template="plotly_white",
    height=380,
    showlegend=False,
)
st.plotly_chart(fig_cost, use_container_width=True)

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
