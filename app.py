import io
import json
import os
import tempfile
import zipfile

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
    .pp-insight-box {
        border-radius: 12px;
        padding: 20px 22px;
        margin: 14px 0 20px 0;
        border: 1px solid #e2e8f0;
        background: #f8fafc;
        border-left-width: 5px;
        border-left-style: solid;
    }
    .pp-insight-kicker { font-size: 0.72rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; }
    .pp-insight-lead { color: #253858; font-size: 1.2rem; font-weight: 800; line-height: 1.35; margin: 10px 0 12px 0; }
    .pp-insight-body { color: #334155; font-size: 0.98rem; line-height: 1.55; }
    .pp-hero {
        border-radius: 14px;
        border: 1px solid #bfdbfe;
        background: linear-gradient(90deg, #eff6ff 0%, #f8fafc 100%);
        padding: 14px 16px;
        color: #0f172a;
        margin-bottom: 12px;
    }
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
    """Return (products_df, resources_df, bom_df, error_message). error_message is None on success."""
    if use_sample:
        try:
            return (*cached_load_sample(), None)
        except Exception as exc:
            return None, None, None, f"Could not load bundled sample data: {exc}"
    if not products_file or not resources_file or not bom_file:
        return None, None, None, "Upload all three CSV files (products, resources, BOM), or enable bundled sample data."
    try:
        for handle in (products_file, resources_file, bom_file):
            if hasattr(handle, "seek"):
                handle.seek(0)
        products = pd.read_csv(products_file)
        resources = pd.read_csv(resources_file)
        bom = pd.read_csv(bom_file)
        return (*load_data_from_frames(products, resources, bom), None)
    except ValueError as exc:
        return None, None, None, str(exc)
    except Exception as exc:
        return None, None, None, f"Could not read uploaded CSVs: {exc}"


def load_data_from_frames(products: pd.DataFrame, resources: pd.DataFrame, bom: pd.DataFrame):
    """Validate and normalize uploaded frames using the same rules as model.load_data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        pp = os.path.join(tmpdir, "products.csv")
        rr = os.path.join(tmpdir, "resources.csv")
        bb = os.path.join(tmpdir, "bom.csv")
        products.to_csv(pp, index=False)
        resources.to_csv(rr, index=False)
        bom.to_csv(bb, index=False)
        return load_data(pp, rr, bb)


def build_csv_template_zip_bytes() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("products.csv", sample_products)
        zf.writestr("resources.csv", sample_resources)
        zf.writestr("bom.csv", sample_bom)
    return buf.getvalue()


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


def _unit_material_cost_per_unit(product: str, bom_df: pd.DataFrame, resources_df: pd.DataFrame) -> float:
    """True variable procurement $ for one unit of SKU (BOM × unit cost)."""
    total = 0.0
    for r in resources_df.index:
        sub = bom_df.loc[(bom_df["product"] == product) & (bom_df["resource"] == r)]
        if sub.empty:
            continue
        u = float(sub["units_required"].iloc[0])
        total += u * float(resources_df.loc[r, "unit_cost"])
    return total


def _max_feasible_increment(
    product: str, rem: pd.Series, bom_df: pd.DataFrame, headroom: float
) -> float:
    """Max extra units of `product` without violating remaining capacities or headroom."""
    caps = [max(0.0, float(headroom))]
    for r in rem.index:
        sub = bom_df.loc[(bom_df["product"] == product) & (bom_df["resource"] == r)]
        if sub.empty:
            continue
        u = float(sub["units_required"].iloc[0])
        if u <= 1e-12:
            continue
        caps.append(float(rem[r]) / u)
    return max(0.0, min(caps))


def build_shadow_price_dataframe(sol: dict, resources_work: pd.DataFrame) -> tuple[pd.DataFrame | None, str | None]:
    """
    Assemble sensitivity table: shadow price ≈ marginal value of relaxing one unit of RHS
    (extra capacity) on the optimal LP basis — continuous model only.
    """
    if not sol.get("duals_valid", True):
        return None, (
            "Shadow prices are reported for the **continuous LP relaxation**. "
            "Turn off **Integer production quantities** in the sidebar to enable duals."
        )
    sp = sol.get("shadow_prices") or {}
    if not sp:
        return None, "Solver did not return shadow prices for this run (try continuous variables)."

    rows = []
    for r in resources_work.index:
        rs = str(r)
        used = float(sol["resource_usage"].get(r, sol["resource_usage"].get(rs, 0.0)))
        avail = float(resources_work.loc[r, "available"])
        slack = max(0.0, avail - used)
        util = float(sol["resource_utilization_pct"].get(r, sol["resource_utilization_pct"].get(rs, 0.0)))
        pi = sp.get(rs)
        if pi is None:
            pi = sp.get(r)
        bind_thr = max(1e-6, 0.002 * max(avail, 1.0))
        binding = "Tight (slack ≈ 0)" if slack < bind_thr else "Has slack"
        rows.append(
            {
                "Resource": rs,
                "Shadow price ($ / extra unit of capacity)": pi,
                "Utilization %": util,
                "Unused capacity (units)": slack,
                "Capacity posture": binding,
            }
        )
    df = pd.DataFrame(rows)
    df["_pi_sort"] = df["Shadow price ($ / extra unit of capacity)"].fillna(0.0)
    df = df.sort_values("_pi_sort", ascending=False).drop(columns=["_pi_sort"])
    if df["Shadow price ($ / extra unit of capacity)"].isna().all():
        return None, "Shadow prices were not exposed by CBC for this model instance."
    return df, None


def compute_naive_greedy_high_cost_bias_plan(
    products_df: pd.DataFrame, resources_df: pd.DataFrame, bom_df: pd.DataFrame
) -> dict:
    """
    Naïve operational baseline (no LP): satisfy min demand, then repeatedly add volume
    prioritizing SKUs with the **worst** material economics (highest procurement $ per $ of price).

    Interpretation for stakeholders: analogous to pushing volume through **cost-heavy SKUs first**
    (a common failure mode if a plant or sourcing lane behaves like a “highest-cost facility”).
    """
    rem = resources_df["available"].astype(float).copy()
    plan: dict[str, float] = {}
    for _, row in products_df.iterrows():
        p = str(row["product"])
        q0 = float(row["min_demand"])
        plan[p] = q0
        for r in rem.index:
            sub = bom_df.loc[(bom_df["product"] == p) & (bom_df["resource"] == r)]
            if sub.empty:
                continue
            rem[r] -= q0 * float(sub["units_required"].iloc[0])
    if float(rem.min()) < -1e-5:
        return {"feasible": False, "reason": "Minimum demand exceeds capacity for this scenario."}

    pindex = products_df.set_index("product")
    prod_rows = []
    for _, row in products_df.iterrows():
        p = str(row["product"])
        price = float(row["price"])
        umc = _unit_material_cost_per_unit(p, bom_df, resources_df)
        badness = umc / max(price, 1e-9)
        prod_rows.append((badness, p, float(row["max_demand"])))

    prod_rows.sort(key=lambda t: t[0], reverse=True)
    order = [p for _, p, _ in prod_rows]

    max_passes = max(50, len(order) * 80)
    for _ in range(max_passes):
        progressed = False
        for p in order:
            headroom = float(pindex.loc[p, "max_demand"]) - plan[p]
            if headroom <= 1e-9:
                continue
            d = _max_feasible_increment(p, rem, bom_df, headroom)
            if d <= 1e-9:
                continue
            plan[p] += d
            for r in rem.index:
                sub = bom_df.loc[(bom_df["product"] == p) & (bom_df["resource"] == r)]
                if sub.empty:
                    continue
                rem[r] -= d * float(sub["units_required"].iloc[0])
            progressed = True
        if not progressed:
            break

    revenue = _total_revenue_from_plan(plan, products_df)
    material = _total_material_cost(plan, bom_df, resources_df)
    profit = revenue - material
    return {
        "feasible": True,
        "plan": plan,
        "total_revenue": revenue,
        "total_material_cost": material,
        "profit": profit,
    }


# --- Sidebar ---
st.sidebar.header("Data")
use_sample = st.sidebar.checkbox("Use bundled sample CSVs in `/data`", value=True)

products_file = st.sidebar.file_uploader("products.csv", type=["csv"])
resources_file = st.sidebar.file_uploader("resources.csv", type=["csv"])
bom_file = st.sidebar.file_uploader("bom.csv", type=["csv"])
st.sidebar.download_button(
    "Download CSV template pack (ZIP)",
    data=build_csv_template_zip_bytes(),
    file_name="production_planning_template.zip",
    mime="application/zip",
    help="products.csv, resources.csv, and bom.csv with the expected columns.",
)

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

run_opt_sidebar = st.sidebar.button("Run optimization", type="primary", use_container_width=True)

with st.sidebar.expander("How to use this app", expanded=False):
    st.markdown(
        "1. Use **bundled sample** data or upload three CSVs (see ZIP template).\n"
        "2. Adjust **capacity ×** and **cost ×** for what-if scenarios.\n"
        "3. Click **Run optimization** (sidebar or main) — sample mode auto-runs once per session.\n"
        "4. Review executive KPIs, mix vs baseline, and exports.\n"
        "5. Open **Sensitivity analysis** for shadow prices (continuous LP only)."
    )

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

products_df, resources_df, bom_df, load_err = get_dfs(use_sample, products_file, resources_file, bom_file)

if load_err:
    st.error(load_err)
    st.stop()

st.title("Production Planning — LP Optimization Console")
st.caption(
    "Decision support for S&OP and plant planning: maximize profit subject to BOM and resource constraints."
)
st.markdown(
    "<div class='pp-hero'><strong>Challenge / problem statement:</strong> With limited materials and machine time, "
    "choosing production quantities by instinct leaves margin on the table. This console runs a transparent LP, "
    "compares baselines, and exports a board-ready snapshot.</div>",
    unsafe_allow_html=True,
)

btn_a, btn_b, _sp = st.columns([1, 1, 3])
with btn_a:
    run_opt_main = st.button("Run optimization", type="primary", use_container_width=True)
with btn_b:
    st.download_button(
        label="Template pack (ZIP)",
        data=build_csv_template_zip_bytes(),
        file_name="production_planning_template.zip",
        mime="application/zip",
        use_container_width=True,
    )

if not use_sample:
    st.session_state["_pp_autosolve_sample_done"] = False

auto_run_sample = bool(
    use_sample and not st.session_state.get("_pp_autosolve_sample_done", False)
)
if auto_run_sample:
    st.session_state["_pp_autosolve_sample_done"] = True

run_effective = bool(run_opt_sidebar or run_opt_main or auto_run_sample)

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

naive_pack = compute_naive_greedy_high_cost_bias_plan(products_df, resources_work, bom_df)

total_demand_units = float(products_df["max_demand"].sum())
n_resource_lanes = int(len(resources_work))

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
    "Resource lanes",
    f"{n_resource_lanes:,}",
    delta=f"×{cap_mult:.2f} capacity slider",
    delta_color="off",
    help="Count of capacity rows. Each row uses its own unit (kg, hours, etc.); do not sum availability across different units.",
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
            "resources": resources_work.reset_index().to_csv(index=False),
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
- **Sensitivity:** After a continuous solve, expand **Sensitivity analysis — shadow prices** to see which capacity rows bind hardest on the optimal basis (dual values from CBC).
        """
    )

st.divider()

# --- Pre-solve ministerial brief (uses baseline reference only) ---
mb1, mb2, mb3 = st.columns(3)
risk_txt = (
    f"Demand push is capped at **{baseline_scale:.0%}** of max demand under current capacity ×{cap_mult:.2f} — "
    "capacity binds before the full demand ceiling."
    if baseline_scale < 0.999
    else "Demand push reaches **100%** of max demand in the reference scenario — check binding resources after solve."
)
impl_txt = (
    "Baseline reference material spend anchors procurement deltas after optimization. "
    f"Reference mix material ≈ **${baseline_material:,.0f}**."
)
action_txt = "Click **Run optimization** (sidebar or above) to compute the profit-maximizing plan and bottleneck signals."
with mb1:
    st.markdown(f"**Risk**\n\n{risk_txt}")
with mb2:
    st.markdown(f"**Implication**\n\n{impl_txt}")
with mb3:
    st.markdown(f"**Action now**\n\n{action_txt}")

st.divider()

st.subheader("Optimization results")
if not run_effective:
    st.info(
        "Configure inputs in the sidebar, then click **Run optimization** here or in the sidebar. "
        "Bundled sample data runs automatically on first load."
    )
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

# --- Naïve baseline vs LP (Prompt 1: scenario delta vs “bad operations”) ---
st.subheader("Scenario comparison — LP vs naïve baseline")
st.caption(
    "**Naïve baseline (no optimizer):** meet **min demand**, then add volume repeatedly, always prioritizing SKUs with the "
    "**highest procurement cost per dollar of price** — a stand‑in for “push everything through your worst‑economics lane / "
    "highest‑cost facility behavior.” The **linear program** then maximizes **profit** on the same capacities and BOM."
)
if naive_pack.get("feasible"):
    naive_profit = float(naive_pack["profit"])
    naive_material = float(naive_pack["total_material_cost"])
    naive_revenue = float(naive_pack["total_revenue"])
    profit_lift_naive = float(profit - naive_profit)
    material_saved_naive = float(naive_material - mat_cost)
    n1, n2, n3 = st.columns(3)
    with n1:
        st.metric(
            "Naïve baseline profit",
            f"${naive_profit:,.2f}",
            help="Greedy plan that favors cost-heavy SKUs first — same capacities & BOM as the LP.",
        )
    with n2:
        st.metric(
            "Optimized profit (LP)",
            f"${profit:,.2f}",
            delta=f"${profit_lift_naive:+,.0f} vs naïve baseline",
            delta_color="normal" if profit_lift_naive > 1 else ("inverse" if profit_lift_naive < -1 else "off"),
            help="How much more profit the LP captures versus the naïve greedy baseline.",
        )
    with n3:
        st.metric(
            "Material spend (LP)",
            f"${mat_cost:,.2f}",
            delta=f"${material_saved_naive:+,.0f} vs naïve baseline",
            delta_color="normal" if material_saved_naive > 1 else ("inverse" if material_saved_naive < -1 else "off"),
            help="Positive delta means the LP buys less material than the naïve plan (often true, not guaranteed).",
        )
    st.caption(
        f"Naïve baseline realized **${naive_revenue:,.0f}** revenue on **${naive_material:,.0f}** material; "
        f"LP realized **${revenue:,.0f}** revenue on **${mat_cost:,.0f}** material."
    )
else:
    st.warning(naive_pack.get("reason", "Naïve baseline is infeasible for this scenario (check min demand vs capacity)."))

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
        f"**Scenario headline (this run only):** material spend is down **~{cost_reduction_pct:.0f}%** vs the baseline "
        "push scenario. Treat this as an illustration for discussion — not an industry benchmark."
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

plan_compare = plan_df.copy().rename(columns={"quantity": "optimized_qty"})
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

# --- Executive insight & recommendation (after cost evidence) ---
shift_idx = plan_compare["qty_delta_vs_baseline"].abs().idxmax()
shift_row = plan_compare.loc[shift_idx]
shift_prod = str(shift_row["product"])
shift_delta = float(shift_row["qty_delta_vs_baseline"])

top_res = res_usage_df.iloc[0]["resource"] if len(res_usage_df) else "—"
top_util = float(res_usage_df.iloc[0]["utilization_%"]) if len(res_usage_df) else 0.0

if saved_usd > 1.0 and profit_vs_baseline > 1.0:
    narrative_accent = "#0d9488"
    insight_lead = (
        f"<strong>Approve the optimized mix</strong> — it beats the scaled max-demand baseline on both "
        f"<strong>profit (+${profit_vs_baseline:,.0f})</strong> and "
        f"<strong>procurement (−${saved_usd:,.0f}, {cost_reduction_pct:.1f}%)</strong> under your current stress test."
    )
    insight_body = (
        f"<strong>Recommendation:</strong> socialize this plan with Ops + Finance using the "
        "<strong>Executive review CSV</strong>. Next: lock assumptions on capacity × "
        f"{cap_mult:.2f} and cost × {cost_mult:.2f}, then schedule a follow‑up run if demand or vendor rates move.<br><br>"
        f"<strong>Bottleneck watch:</strong> <strong>{top_res}</strong> is at <strong>{top_util:.1f}%</strong> utilization — "
        "any supply shock there binds first.<br><br>"
        f"<strong>Mix shift:</strong> largest move vs baseline is <strong>{shift_prod}</strong> "
        f"(<strong>{shift_delta:+,.0f}</strong> units) — confirm changeovers and MOQs before rollout."
    )
elif saved_usd > 1.0:
    narrative_accent = "#0052CC"
    insight_lead = (
        f"<strong>Procurement-led win</strong> — material spend is down <strong>${saved_usd:,.0f}</strong> "
        f"(<strong>{cost_reduction_pct:.1f}%</strong> vs baseline) even if headline profit vs baseline is modest."
    )
    insight_body = (
        "<strong>Recommendation:</strong> pair this view with SKU margin checks so savings are not masking "
        "revenue left on the table. Export the <strong>Executive review CSV</strong> for audit.<br><br>"
        f"<strong>Capacity story:</strong> tightest resource remains <strong>{top_res}</strong> at "
        f"<strong>{top_util:.1f}%</strong> — if you relax capacity ×, re‑run to see shadow price relief."
    )
elif profit_vs_baseline > 1.0 and saved_usd < -1.0:
    narrative_accent = "#0052CC"
    insight_lead = (
        "<strong>Revenue-first plan</strong> — the LP spends more on inputs than the naive baseline because "
        f"it buys <strong>+${-saved_usd:,.0f}</strong> of margin‑accretive volume (profit vs baseline "
        f"<strong>+${profit_vs_baseline:,.0f}</strong>)."
    )
    insight_body = (
        "<strong>Recommendation:</strong> treat this as a <strong>growth / mix</strong> decision, not a cost‑cut "
        "story. Validate service levels and supplier payment terms before committing.<br><br>"
        f"<strong>Watch:</strong> <strong>{top_res}</strong> @ <strong>{top_util:.1f}%</strong>; "
        f"largest quantity swing: <strong>{shift_prod}</strong> (<strong>{shift_delta:+,.0f}</strong> units vs baseline)."
    )
else:
    narrative_accent = "#64748b"
    insight_lead = (
        "<strong>Neutral / marginal trade</strong> — vs the baseline push, the LP does not deliver a large "
        "separable win on procurement or profit at the current stress settings."
    )
    insight_body = (
        "<strong>Recommendation:</strong> widen what‑if ranges, revisit <strong>prices / BOM</strong>, or "
        "tighten min–max demand bands so the model has more room to separate scenarios. "
        "Always attach the <strong>Executive review CSV</strong> when escalating.<br><br>"
        f"<strong>Diagnostics:</strong> mean utilization <strong>{avg_util:.1f}%</strong>; "
        f"highest binding resource <strong>{top_res}</strong> (<strong>{top_util:.1f}%</strong>)."
    )

st.markdown(
    f"""
<div class="pp-insight-box" style="border-left-color:{narrative_accent};">
  <div class="pp-insight-kicker" style="color:{narrative_accent};">Executive insight</div>
  <div class="pp-insight-lead">{insight_lead}</div>
  <div class="pp-insight-body">{insight_body}</div>
</div>
""",
    unsafe_allow_html=True,
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

with st.expander("Sensitivity analysis — shadow prices (dual values)", expanded=False):
    st.markdown(
        """
**What this means (decision science)**  
For each resource capacity row (`consumption ≤ available`), the **shadow price** is the **approximate**
increase in optimal **profit** if you relax that capacity by **one more unit** — holding everything else fixed.

- **Higher shadow price** → that capacity is a **more expensive bottleneck** on the margin (freeing it pays more).
- **Unused capacity (slack)** → shadow price is typically **~0** (extra units would not change the optimum yet).

Shadow prices come from the **LP dual**; they are most interpretable for the **continuous** formulation.
        """
    )
    sh_df, sh_msg = build_shadow_price_dataframe(sol, resources_work)
    if sh_msg:
        st.info(sh_msg)
    if sh_df is not None:
        st.dataframe(sh_df, use_container_width=True, hide_index=True)
        top = sh_df.iloc[0]
        pi0 = top["Shadow price ($ / extra unit of capacity)"]
        if pd.notna(pi0) and float(pi0) > 1e-8:
            st.success(
                f"**Tightest economic bottleneck (on margin):** **{top['Resource']}** — shadow price "
                f"**${float(pi0):,.2f}** per extra unit of capacity vs **{top['Capacity posture'].lower()}**."
            )
        else:
            st.caption("No strongly positive shadow prices returned — capacity may be abundant vs demand at this optimum.")

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
