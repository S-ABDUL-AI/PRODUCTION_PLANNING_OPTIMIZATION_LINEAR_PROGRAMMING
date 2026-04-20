import html
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
    page_title="Production planning assistant",
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
    .pp-plain-insight {
        background: #f1f5f9;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 14px 16px;
        color: #1e293b;
        font-size: 1.02rem;
        line-height: 1.55;
        margin-bottom: 14px;
    }
    h3 { margin-top: 0.35rem; }
    .pp-triage-band {
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        padding: 20px 24px 18px;
        margin: 6px 0 20px 0;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06);
    }
    .pp-triage-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: clamp(16px, 3vw, 28px);
        align-items: start;
    }
    @media (max-width: 900px) {
        .pp-triage-grid { grid-template-columns: 1fr; }
    }
    .pp-triage-title {
        font-weight: 700;
        color: #0f172a;
        font-size: 0.82rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin: 0 0 10px 0;
    }
    .pp-triage-body {
        color: #475569;
        font-size: 0.96rem;
        line-height: 1.55;
        margin: 0;
    }
    .pp-triage-body strong { color: #0f172a; font-weight: 600; }
    .pp-triage-rule {
        height: 1px;
        background: #e2e8f0;
        margin-top: 18px;
    }
    .pp-snapshot-band {
        border: 1px solid #c7d2fe;
        border-radius: 14px;
        background: linear-gradient(135deg, #f8fafc 0%, #eff6ff 55%, #f1f5f9 100%);
        padding: 20px 22px 18px;
        margin: 0 0 16px 0;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.06);
    }
    .pp-snapshot-title {
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.11em;
        text-transform: uppercase;
        color: #0052CC;
        margin: 0 0 14px 0;
    }
    .pp-snapshot-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: clamp(12px, 2vw, 20px);
    }
    @media (max-width: 1100px) {
        .pp-snapshot-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
    @media (max-width: 560px) {
        .pp-snapshot-grid { grid-template-columns: 1fr; }
    }
    .pp-snapshot-cell {
        background: #ffffffcc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 14px 14px 12px;
    }
    .pp-snapshot-label {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #64748b;
        margin: 0 0 6px 0;
    }
    .pp-snapshot-value {
        font-size: 1.35rem;
        font-weight: 800;
        color: #0f172a;
        line-height: 1.2;
        margin: 0 0 6px 0;
    }
    .pp-snapshot-sub {
        font-size: 0.88rem;
        color: #475569;
        line-height: 1.45;
        margin: 0;
    }
    .pp-snapshot-foot {
        margin-top: 16px;
        padding-top: 14px;
        border-top: 1px solid #e2e8f0;
        font-size: 0.92rem;
        color: #334155;
        line-height: 1.5;
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


def render_input_parameters_expander(
    products_df: pd.DataFrame, resources_work: pd.DataFrame, bom_df: pd.DataFrame
) -> None:
    """Collapsed tables + JSON snapshot (same data passed to the solver)."""
    with st.expander("Input parameters", expanded=False):
        st.caption(
            "The three tables the app reads: **products**, **limits** (after your sidebar multipliers), and **recipes** "
            "(which inputs each product uses)."
        )
        t1, t2, t3 = st.columns(3)
        with t1:
            st.markdown("**Products**")
            st.dataframe(products_df, use_container_width=True, height=200)
        with t2:
            st.markdown("**Limits**")
            st.dataframe(resources_work.reset_index(), use_container_width=True, height=200)
        with t3:
            st.markdown("**Recipes (product → inputs)**")
            st.dataframe(bom_df, use_container_width=True, height=200)
        st.download_button(
            "Download input parameters (JSON snapshot)",
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
            "These “extra profit per unit of room” numbers work best when **Whole-number production units (rounding)** is **unchecked** "
            "in the sidebar."
        )
    sp = sol.get("shadow_prices") or {}
    if not sp:
        return None, "No sensitivity table for this run—try unchecking **Whole-number production units (rounding)** and run again."

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
        return None, "Sensitivity numbers were not available for this run from the solver."
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
with st.sidebar.expander("Built-in sample or your 3 CSVs", expanded=False):
    st.caption("Turn off the sample to upload **products**, **resources**, and **BOM** (recipe) files.")
    use_sample = st.checkbox("Use bundled sample CSVs in `/data`", value=True)

    products_file = st.file_uploader("products.csv", type=["csv"])
    resources_file = st.file_uploader("resources.csv", type=["csv"])
    bom_file = st.file_uploader("bom.csv", type=["csv"])
    st.download_button(
        "Download CSV template pack (ZIP)",
        data=build_csv_template_zip_bytes(),
        file_name="production_planning_template.zip",
        mime="application/zip",
        help="products.csv, resources.csv, and bom.csv with the expected columns.",
        use_container_width=True,
    )

st.sidebar.header("What-if (stress test)")
cap_mult = st.sidebar.slider(
    "How tight are limits?",
    min_value=0.5,
    max_value=1.5,
    value=1.0,
    step=0.05,
    help="1.0 = use the capacities in your file. Lower = pretend everything is tighter (e.g. downtime); higher = more room.",
)
cost_mult = st.sidebar.slider(
    "How expensive are inputs?",
    min_value=0.5,
    max_value=1.5,
    value=1.0,
    step=0.05,
    help="1.0 = use the $/unit costs in your file. Move up or down to stress supplier price changes.",
)

st.sidebar.header("Solver")
integer_vars = st.sidebar.checkbox("Whole-number production units (rounding)", value=False)
time_limit = st.sidebar.number_input("Time limit (seconds, 0 = default)", min_value=0, max_value=600, value=0, step=5)

run_opt_sidebar = st.sidebar.button("Run optimization", type="primary", use_container_width=True)

with st.sidebar.expander("How to use this app", expanded=False):
    st.markdown(
        "1. Open **Data → Built-in sample or your 3 CSVs** for the sample toggle, uploads, and ZIP template.\n"
        "2. Adjust **capacity ×** and **cost ×** for what-if scenarios.\n"
        "3. Click **Run optimization** (sidebar or main) — sample mode auto-runs once per session.\n"
        "4. Read **Overview** for the headline, **Compare scenarios** for contrasts, **Charts** for visuals, "
        "**Tables & downloads** for CSV/JSON.\n"
        "5. Optional: under **Charts**, open **Which limit pays off to loosen?** (clearest with **Whole-number production units** unchecked)."
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

st.title("Production planning assistant")
st.caption(
    "Turn your product list, limits (materials, hours, etc.), and bill of materials into a profit-minded production mix."
)
st.markdown(
    "<div class='pp-hero'><strong>Challenge / problem statement:</strong> You have several products and limited inputs. "
    "Guessing how much to make on each line leaves money on the table. This tool proposes a mix that aims for the best profit "
    "while respecting your limits—and shows where those limits pinch hardest.</div>",
    unsafe_allow_html=True,
)

if not use_sample:
    st.session_state["_pp_autosolve_sample_done"] = False

auto_run_sample = bool(
    use_sample and not st.session_state.get("_pp_autosolve_sample_done", False)
)
if auto_run_sample:
    st.session_state["_pp_autosolve_sample_done"] = True

ceiling_revenue = float((products_df["max_demand"] * products_df["price"]).sum())

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

n_resource_lanes = int(len(resources_work))
total_max_demand_units = float(products_df["max_demand"].sum())
n_bom_lines = int(len(bom_df))

st.markdown("### Key numbers")
st.caption(
    f"**{len(products_df)} products** · **{n_resource_lanes} kinds of limits** in your file (each row is one thing—hours, kg, etc.—don’t add unlike units). "
    f"**Stress test (sidebar):** limits are **×{cap_mult:.2f}** and input **prices** are **×{cost_mult:.2f}** compared with your CSV. "
    "Later we use a **simple reference** (grow every product in step until something runs out) only to **compare numbers**, not as a real operating plan."
)
pk1, pk2, pk3, pk4 = st.columns(4)
with pk1:
    st.metric(
        "Total max demand (units)",
        f"{total_max_demand_units:,.0f}",
        delta=f"{len(products_df)} products",
        delta_color="off",
        help="Add up each product’s “if we could sell the most” quantity—shows the size of the case.",
    )
with pk2:
    st.metric(
        "Kinds of limits in file",
        f"{n_resource_lanes:,}",
        delta=f"limits ×{cap_mult:.2f}",
        delta_color="off",
        help="One row per limit in your data (machine hours, material, etc.). Each uses its own unit.",
    )
with pk3:
    st.metric(
        "Simple reference: input spend",
        f"${baseline_material:,.0f}",
        delta=f"even growth stops ~{baseline_scale:.0%} of full demand",
        delta_color="off",
        help="If every product grew the same way until something ran out, this is about what you’d spend on bought inputs.",
    )
with pk4:
    st.metric(
        "Dream-team sales ($)",
        f"${ceiling_revenue:,.0f}",
        delta=f"{n_bom_lines} product–input links",
        delta_color="off",
        help="If you sold every SKU at its top quantity at list price, with no capacity problem—an upper bound, not a forecast.",
    )

_risk_html = (
    f"If every product grew in lockstep, you’d only get to about <strong>{baseline_scale:.0%}</strong> of full demand before "
    f"a limit stops you (with limits at <strong>×{cap_mult:.2f}</strong> in the sidebar). Something runs out before you “max out” every line."
    if baseline_scale < 0.999
    else (
        f"With limits at <strong>×{cap_mult:.2f}</strong>, that even-growth story could reach <strong>full</strong> max demand in this file—"
        "still run optimization for the best-profit mix."
    )
)
_impl_html = (
    f"After you run, we compare to a simple **baseline spend on inputs** of about <strong>${baseline_material:,.0f}</strong> "
    "(that even-growth story above)—so you can see whether the recommended plan spends more or less and still wins on profit."
)
_action_html = (
    "Click <strong>Run optimization</strong> (sidebar or below) to get the best-profit mix and see **which limit is tightest**."
)

st.subheader("Recommendations")
st.markdown(
    f"""
<div class="pp-triage-band">
  <div class="pp-triage-grid">
    <div>
      <div class="pp-triage-title">Risk</div>
      <p class="pp-triage-body">{_risk_html}</p>
    </div>
    <div>
      <div class="pp-triage-title">Implication</div>
      <p class="pp-triage-body">{_impl_html}</p>
    </div>
    <div>
      <div class="pp-triage-title">Action now</div>
      <p class="pp-triage-body">{_action_html}</p>
    </div>
  </div>
  <div class="pp-triage-rule" aria-hidden="true"></div>
</div>
""",
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

run_effective = bool(run_opt_sidebar or run_opt_main or auto_run_sample)

st.subheader("Optimization results")
if not run_effective:
    st.info(
        "Pick your files or use the sample data in the sidebar, then click **Run optimization**. "
        "Sample mode runs once automatically the first time you open the app."
    )
    render_input_parameters_expander(products_df, resources_work, bom_df)
    st.stop()

with st.spinner("Working out the best-profit mix for your limits…"):
    sol = solve_production_plan(
        products_df,
        resources_work,
        bom_df,
        integer=integer_vars,
        time_limit=(int(time_limit) if time_limit > 0 else None),
    )

status = sol.get("status", "Undefined")
if status == 1 and "plan" in sol:
    st.success("Found a best-profit plan for this scenario—numbers below.")
elif "plan" not in sol:
    st.error(
        "No workable plan with these limits and minimums. Try raising capacity, relaxing minimum demand, "
        "or checking the bill of materials."
    )
    render_input_parameters_expander(products_df, resources_work, bom_df)
    st.stop()
else:
    st.warning(
        "The solver finished with an unusual status—treat these numbers as provisional until you validate the case."
    )

profit = sol["profit"]
revenue = sol["total_revenue"]
mat_cost = sol["total_material_cost"]
margin_pct = (profit / revenue * 100.0) if revenue else 0.0
ceiling_capture = (revenue / ceiling_revenue * 100.0) if ceiling_revenue else 0.0

plan_df = pd.DataFrame([{"product": p, "quantity": q} for p, q in sol["plan"].items()]).sort_values("product")
plan_compare = plan_df.copy().rename(columns={"quantity": "optimized_qty"})
plan_compare["baseline_qty"] = plan_compare["product"].map(baseline_plan).astype(float)
plan_compare["qty_delta_vs_baseline"] = plan_compare["optimized_qty"] - plan_compare["baseline_qty"]

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
    input_pct_vs_ref = (mat_cost - baseline_material) / baseline_material * 100.0
else:
    cost_reduction_pct = 0.0
    input_pct_vs_ref = 0.0
saved_usd = baseline_material - mat_cost
profit_vs_baseline = profit - baseline_profit

_top_row = res_usage_df.iloc[0] if len(res_usage_df) else None
_top_res_name = str(_top_row["resource"]) if _top_row is not None else "—"
_top_util_pct = float(_top_row["utilization_%"]) if _top_row is not None else 0.0
_esc_res = html.escape(_top_res_name)

_naive_ok = bool(naive_pack.get("feasible"))
if _naive_ok:
    _nv_p = float(naive_pack["profit"])
    _nv_m = float(naive_pack["total_material_cost"])
    _lift_nv = profit - _nv_p
    _mat_saved_vs_naive = _nv_m - mat_cost
else:
    _nv_p = _nv_m = _lift_nv = _mat_saved_vs_naive = None

if abs(saved_usd) < 0.01:
    _inp_vs_ref_txt = "About the <strong>same</strong> on purchased inputs as the simple reference mix."
elif saved_usd > 0.01:
    _inp_vs_ref_txt = (
        f"<strong>${saved_usd:,.0f}</strong> <strong>less</strong> on inputs than the simple reference "
        f"(about <strong>{-input_pct_vs_ref:.1f}%</strong> lower spend)."
    )
else:
    _inp_vs_ref_txt = (
        f"<strong>${-saved_usd:,.0f}</strong> <strong>more</strong> on inputs than the simple reference "
        f"(<strong>+{input_pct_vs_ref:.1f}%</strong> spend)—often paired with <strong>higher sales</strong> when the plan pushes margin-rich volume."
    )

if _naive_ok:
    if abs(_mat_saved_vs_naive) < 0.01 and abs(_lift_nv) < 0.01:
        _foot_greedy = (
            "Compared to a <strong>careless greedy</strong> plan (same limits): about the same profit and input spend in this toy case."
        )
    else:
        _foot_greedy = (
            f"Compared to a <strong>careless greedy</strong> plan on the same numbers: profit <strong>${_lift_nv:+,.0f}</strong>; "
            f"inputs <strong>${_mat_saved_vs_naive:+,.0f}</strong> vs greedy (positive means you buy <strong>less</strong> than greedy)."
        )
else:
    _foot_greedy = "Greedy baseline not available for this scenario—see <strong>Compare scenarios</strong> for detail."

st.markdown(
    f"""
<div class="pp-snapshot-band">
  <div class="pp-snapshot-title">Results at a glance</div>
  <div class="pp-snapshot-grid">
    <div class="pp-snapshot-cell">
      <div class="pp-snapshot-label">Profit</div>
      <div class="pp-snapshot-value">${profit:,.0f}</div>
      <p class="pp-snapshot-sub">{margin_pct:.1f}% margin · <strong>${profit_vs_baseline:+,.0f}</strong> vs the simple even-growth story’s profit</p>
    </div>
    <div class="pp-snapshot-cell">
      <div class="pp-snapshot-label">Sales</div>
      <div class="pp-snapshot-value">${revenue:,.0f}</div>
      <p class="pp-snapshot-sub">About <strong>{ceiling_capture:.0f}%</strong> of “<strong>every</strong> product at max demand at list price” (~<strong>${ceiling_revenue:,.0f}</strong>)—a ceiling, not a forecast.</p>
    </div>
    <div class="pp-snapshot-cell">
      <div class="pp-snapshot-label">Purchased inputs</div>
      <div class="pp-snapshot-value">${mat_cost:,.0f}</div>
      <p class="pp-snapshot-sub">That simple story would spend ~<strong>${baseline_material:,.0f}</strong> on inputs. {_inp_vs_ref_txt}</p>
    </div>
    <div class="pp-snapshot-cell">
      <div class="pp-snapshot-label">Tightest limit</div>
      <div class="pp-snapshot-value">{_esc_res}</div>
      <p class="pp-snapshot-sub">Using about <strong>{_top_util_pct:.0f}%</strong> of what you allowed for that line · simple average across lines <strong>{avg_util:.0f}%</strong> (mixed units—rough health check only)</p>
    </div>
  </div>
  <div class="pp-snapshot-foot">{_foot_greedy}</div>
</div>
""",
    unsafe_allow_html=True,
)

tab_overview, tab_compare, tab_charts, tab_data = st.tabs(
    ["Overview", "Compare scenarios", "Charts", "Tables & downloads"]
)

with tab_overview:
    with st.expander("How this works (plain English)", expanded=False):
        st.markdown(
            """
The app chooses **how many units** of each product to make within your **minimum and maximum** demand bands.  
It **adds up sales** from your prices, **subtracts** what you pay for each bought input in your recipe table, and picks the mix that **maximizes profit** while **staying under** each limit in your file (after the sidebar stress sliders).

The **simple reference plan** is only a storytelling anchor: grow every SKU in the same pattern until something runs out. The **recommended plan** is the profit-minded answer on the same numbers.
            """
        )

    with st.expander("For analysts (formulation & duals)", expanded=False):
        st.markdown(
            """
**Decision variables** — `q_i` = production quantity for SKU *i*, with `min_demand_i ≤ q_i ≤ max_demand_i`.

**Objective** — maximize Σ `price_i × q_i` minus procurement: for each resource *r*, `unit_cost_r × Σ_i q_i × BOM[i,r]`.

**Constraints** — for each *r*, consumption ≤ `available_r` (after the capacity multiplier).

**Reference mix** — proportional max-demand push scaled by the largest factor in `[0,1]` that keeps the mix feasible; material $ uses the same unit-cost multipliers.

**Shadow prices (“duals”)** — for a **continuous** run, roughly **extra profit** from **one more unit** of that limit; **~0** usually means spare room. See **Charts → Which limit pays off to loosen?** after solving.
            """
        )

    shift_idx = plan_compare["qty_delta_vs_baseline"].abs().idxmax()
    shift_row = plan_compare.loc[shift_idx]
    shift_prod = str(shift_row["product"])
    shift_delta = float(shift_row["qty_delta_vs_baseline"])
    top_res = _top_res_name
    top_util = _top_util_pct

    if saved_usd > 1.0 and profit_vs_baseline > 1.0:
        narrative_accent = "#0d9488"
        insight_lead = (
            f"<strong>Strong story on both profit and inputs:</strong> about <strong>+${profit_vs_baseline:,.0f}</strong> more profit "
            f"than the simple reference plan, and roughly <strong>${saved_usd:,.0f}</strong> less on purchased inputs "
            f"(<strong>{cost_reduction_pct:.1f}%</strong>) at your current slider settings."
        )
        insight_body = (
            f"<strong>What to do next:</strong> walk this through ops and finance using the <strong>Tables & downloads</strong> tab. "
            f"Lock in the sidebar stress settings (limits ×{cap_mult:.2f}, prices ×{cost_mult:.2f}), then re-run if demand or supplier prices move.<br><br>"
            f"<strong>Where it gets tight first:</strong> <strong>{top_res}</strong> is running at about <strong>{top_util:.0f}%</strong> of what you allowed—"
            "any slip there hits the plan fastest.<br><br>"
            f"<strong>Biggest quantity swing vs the reference:</strong> <strong>{shift_prod}</strong> "
            f"(about <strong>{shift_delta:+,.0f}</strong> units)—sanity-check changeovers and minimum order sizes before you commit."
        )
    elif saved_usd > 1.0:
        narrative_accent = "#0052CC"
        insight_lead = (
            f"<strong>Input spend looks better:</strong> roughly <strong>${saved_usd:,.0f}</strong> less on purchased inputs "
            f"(<strong>{cost_reduction_pct:.1f}%</strong> vs the simple reference), even if headline profit vs that reference is small."
        )
        insight_body = (
            "<strong>What to do next:</strong> double-check margins by product so cheaper inputs are not hiding weaker sales. "
            "Use the export bundle under <strong>Tables & downloads</strong> for a paper trail.<br><br>"
            f"<strong>Capacity:</strong> the tightest line is still <strong>{top_res}</strong> at about <strong>{top_util:.0f}%</strong>—"
            "try moving the **How tight are limits?** slider up and re-running to see how much room opens up."
        )
    elif profit_vs_baseline > 1.0 and saved_usd < -1.0:
        narrative_accent = "#0052CC"
        insight_lead = (
            "<strong>Revenue-led plan:</strong> this run spends more on inputs than the simple reference because it buys "
            f"about <strong>${-saved_usd:,.0f}</strong> more material to capture higher-margin volume "
            f"(profit vs reference about <strong>+${profit_vs_baseline:,.0f}</strong>)."
        )
        insight_body = (
            "<strong>What to do next:</strong> treat this as a growth or mix decision, not a pure cost-down story. "
            "Check customer promises and payment terms before locking volumes.<br><br>"
            f"<strong>Watch:</strong> <strong>{top_res}</strong> at about <strong>{top_util:.0f}%</strong>; "
            f"largest swing vs reference: <strong>{shift_prod}</strong> (<strong>{shift_delta:+,.0f}</strong> units)."
        )
    else:
        narrative_accent = "#64748b"
        insight_lead = (
            "<strong>Modest move vs the reference plan</strong> — at these slider settings the recommended mix does not separate "
            "much from the simple proportional story on profit or input spend."
        )
        insight_body = (
            "<strong>What to do next:</strong> widen the what-if sliders, revisit prices or recipes, or tighten min/max demand bands "
            "so the case has more room to breathe. Attach the CSV bundle from <strong>Tables & downloads</strong> if you escalate.<br><br>"
            f"<strong>Pulse check:</strong> simple average use across limit lines about <strong>{avg_util:.0f}%</strong>; "
            f"busiest line <strong>{top_res}</strong> at about <strong>{top_util:.0f}%</strong>."
        )

    st.markdown(
        f"""
<div class="pp-insight-box" style="border-left-color:{narrative_accent};">
  <div class="pp-insight-kicker" style="color:{narrative_accent};">Read on the story</div>
  <div class="pp-insight-lead">{insight_lead}</div>
  <div class="pp-insight-body">{insight_body}</div>
</div>
""",
        unsafe_allow_html=True,
    )

# --- Naïve baseline vs LP (Prompt 1: scenario delta vs “bad operations”) ---
with tab_compare:
    st.subheader("Versus a careless greedy plan")
    st.caption(
        "**Greedy “bad habits” plan (no optimizer):** satisfy **minimum** demand, then keep adding volume while always picking "
        "the products that burn the **most input cost per sales dollar** first—like favoring your most expensive lanes. "
        "The **recommended plan** uses the same limits and recipes but aims for the **best profit**."
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
                "Greedy plan profit",
                f"${naive_profit:,.2f}",
                help="Same limits and bill of materials; prioritizes expensive-to-serve volume first.",
            )
        with n2:
            st.metric(
                "Recommended plan profit",
                f"${profit:,.2f}",
                delta=f"${profit_lift_naive:+,.0f} vs greedy plan",
                delta_color="normal" if profit_lift_naive > 1 else ("inverse" if profit_lift_naive < -1 else "off"),
                help="How much more profit the recommended mix captures versus the greedy baseline.",
            )
        with n3:
            st.metric(
                "Purchased inputs (recommended)",
                f"${mat_cost:,.2f}",
                delta=f"${material_saved_naive:+,.0f} vs greedy plan",
                delta_color="normal" if material_saved_naive > 1 else ("inverse" if material_saved_naive < -1 else "off"),
                help="Positive means this run buys less material than the greedy plan (common, not guaranteed).",
            )
        st.caption(
            f"Greedy plan: about **${naive_revenue:,.0f}** sales and **${naive_material:,.0f}** on inputs. "
            f"Recommended: **${revenue:,.0f}** sales and **${mat_cost:,.0f}** on inputs."
        )
    else:
        st.warning(
            naive_pack.get(
                "reason",
                "Greedy baseline cannot meet minimums with these limits—check minimum demand vs capacity.",
            )
        )

    st.subheader("Versus the simple reference spend")
    st.caption(
        "**Reference:** pretend every product grows together until limits bite (scaled to stay feasible). "
        "**Recommended:** the profit-minded mix on the same numbers."
    )
    c_sc_a, c_sc_mid, c_sc_b = st.columns([1, 0.55, 1])
    with c_sc_a:
        st.markdown("##### Reference input spend")
        st.metric(
            label="Purchased inputs",
            value=f"${baseline_material:,.2f}",
            help="What the simple proportional reference would spend on inputs.",
        )
    with c_sc_mid:
        st.markdown(
            "<div style='text-align:center;padding-top:1.6rem;font-size:2rem;color:#0052CC;font-weight:700;'>→</div>",
            unsafe_allow_html=True,
        )
        st.metric(
            label="Input spend vs reference",
            value=f"{input_pct_vs_ref:+.1f}%",
            delta=(
                f"${mat_cost - baseline_material:+,.0f} vs reference"
                if abs(mat_cost - baseline_material) > 0.01
                else "No change"
            ),
            delta_color="inverse" if mat_cost > baseline_material + 0.01 else ("normal" if mat_cost < baseline_material - 0.01 else "off"),
            help="Positive % = this run spends more on inputs than the simple reference; negative % = spends less.",
        )
    with c_sc_b:
        st.markdown("##### Recommended input spend")
        st.metric(
            label="Purchased inputs",
            value=f"${mat_cost:,.2f}",
            help="Input spend at the recommended production mix.",
        )

    if 12 <= cost_reduction_pct <= 18:
        st.success(
            f"**Illustration only:** input spend is about **{cost_reduction_pct:.0f}%** lower than the reference push—"
            "good for discussion, not a benchmark by itself."
        )
    elif cost_reduction_pct > 0:
        st.info(
            f"**Inputs vs reference:** about **{cost_reduction_pct:.1f}%** less spend on purchased inputs "
            f"(~**${saved_usd:,.0f}** at this mix)."
        )
    else:
        st.warning(
            "**Inputs can be higher than the reference** when the recommended mix chases stronger margins—"
            "spot-check prices and recipes by product."
        )

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

fig_cost = go.Figure()
fig_cost.add_trace(
    go.Bar(
        x=["Reference input spend", "Recommended input spend"],
        y=[baseline_material, mat_cost],
        marker_color=["#64748b", "#0052CC"],
        text=[f"${baseline_material:,.0f}", f"${mat_cost:,.0f}"],
        textposition="outside",
        textfont=dict(size=13, color="#253858"),
    )
)
fig_cost.update_layout(
    title="Purchased inputs — reference vs recommended",
    yaxis_title="USD",
    xaxis=dict(tickangle=0, tickfont=dict(size=13, color="#253858")),
    template="plotly_white",
    height=380,
    showlegend=False,
)

fig1 = go.Figure()
fig1.add_trace(go.Bar(x=plan_df["product"], y=plan_df["quantity"], marker_color="#0052CC", name="Units"))
fig1.update_layout(
    title="Recommended quantities by product",
    xaxis_title="Product",
    yaxis_title="Units",
    template="plotly_white",
    height=420,
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
    title="How hard each limit is being used",
    xaxis_title="Limit row",
    yaxis_title="Use (%)",
    template="plotly_white",
    height=420,
)

with tab_charts:
    st.caption(
        "Visual readout for the same run: money on inputs, recommended volumes, and how hard each limit from your file is working."
    )
    st.plotly_chart(fig_cost, use_container_width=True)
    st.plotly_chart(fig1, use_container_width=True)
    _dl1, _dl2 = st.columns([1, 5])
    with _dl1:
        st.download_button(
            "Plan quantities (CSV)",
            data=plan_df.to_csv(index=False).encode("utf-8"),
            file_name="plan_quantities.csv",
            mime="text/csv",
        )
    st.plotly_chart(fig2, use_container_width=True)
    _dl3, _dl4 = st.columns([1, 5])
    with _dl3:
        st.download_button(
            "Limit use (CSV)",
            data=res_usage_df.to_csv(index=False).encode("utf-8"),
            file_name="resource_utilization.csv",
            mime="text/csv",
        )

    with st.expander("Which limit pays off to loosen? (advanced)", expanded=False):
        st.markdown(
            """
For each **limit from your file**, the table estimates how much **extra profit** you might get from **one more unit** of room there—
**if nothing else changed**. A **higher dollar value** usually means easing that limit pays off more on the margin.
**Near zero** often means you still have slack there.

Clearest when **Whole-number production units (rounding)** is **unchecked** in the sidebar.
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
                    f"On the margin, **{top['Resource']}** looks like the most valuable extra unit of capacity—about "
                    f"**${float(pi0):,.2f}** more profit per extra unit vs **{top['Capacity posture'].lower()}**."
                )
            else:
                st.caption("No strong ‘extra capacity is worth a lot’ signal here—demand or other limits may be the real story.")

with tab_data:
    st.download_button(
        label="Board-style snapshot (one CSV)",
        data=exec_csv_bytes,
        file_name="executive_review_supply_chain_scenario.csv",
        mime="text/csv",
        help="KPI snapshot, reference vs recommended quantities, and limit use in one file.",
    )
    st.subheader("Tables & file exports")
    st.markdown("#### Recommended plan")
    st.dataframe(plan_df, use_container_width=True)
    st.markdown("#### Limit use")
    st.dataframe(res_usage_df, use_container_width=True)

    plan_path = "outputs/plan.csv"
    usage_path = "outputs/resource_usage.csv"
    save_solution(sol, "outputs/solution.json")
    plan_df.to_csv(plan_path, index=False)
    res_usage_df.to_csv(usage_path, index=False)

    b1, b2, b3 = st.columns(3)
    with b1:
        st.download_button("Plan (CSV)", data=plan_df.to_csv(index=False), file_name="plan.csv", mime="text/csv")
    with b2:
        st.download_button(
            "Limit use (CSV)",
            data=res_usage_df.to_csv(index=False),
            file_name="resource_usage.csv",
            mime="text/csv",
        )
    with b3:
        st.download_button(
            "Full detail (JSON)",
            data=json.dumps(sol, indent=2, default=str),
            file_name="solution.json",
            mime="application/json",
        )

render_input_parameters_expander(products_df, resources_work, bom_df)
