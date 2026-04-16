# model.py — LP formulation + data load only. Charts belong in app.py (Plotly); do not import matplotlib here.
import json
import re

import pandas as pd
from pulp import LpMaximize, LpProblem, LpVariable, PULP_CBC_CMD, lpSum


def _safe_constraint_name(resource: str) -> str:
    """PuLP constraint names should avoid spaces / odd characters."""
    s = re.sub(r"[^0-9A-Za-z_]+", "_", str(resource)).strip("_")
    return f"ResourceCap__{s or 'r'}"

# --- Load CSVs ---
def load_data(products_path, resources_path, bom_path):
    products = pd.read_csv(products_path)
    resources = pd.read_csv(resources_path)
    bom = pd.read_csv(bom_path)

    # Validate required columns
    missing_p = set(["product", "price", "min_demand", "max_demand"]) - set(products.columns)
    if missing_p:
        raise ValueError(f"products.csv missing columns: {missing_p}")

    missing_r = set(["resource", "available", "unit_cost"]) - set(resources.columns)
    if missing_r:
        raise ValueError(f"resources.csv missing columns: {missing_r}")

    missing_b = set(["product", "resource", "units_required"]) - set(bom.columns)
    if missing_b:
        raise ValueError(f"bom.csv missing columns: {missing_b}")

    # Ensure correct dtypes
    products["price"] = products["price"].astype(float)
    products["min_demand"] = products["min_demand"].astype(float)
    products["max_demand"] = products["max_demand"].astype(float)

    resources["available"] = resources["available"].astype(float)
    resources["unit_cost"] = resources["unit_cost"].astype(float)
    resources["resource"] = resources["resource"].astype(str)

    bom["units_required"] = bom["units_required"].astype(float)

    # Stable string IDs avoid dtype mismatches (e.g. int SKU in CSV vs str keys) and simplify masks.
    products["product"] = products["product"].astype(str)
    bom["product"] = bom["product"].astype(str)
    bom["resource"] = bom["resource"].astype(str)

    # Set resource index for easy access
    resources.set_index("resource", inplace=True)

    return products, resources, bom


def _scalar_price(products: pd.DataFrame, product) -> float:
    """Single SKU price as Python float (.loc on a column can return a Series)."""
    pid = str(product)
    sel = products.loc[products["product"] == pid, "price"]
    if sel.empty:
        raise KeyError(f"Unknown product in products table: {product!r}")
    return float(sel.iloc[0])


def _scalar_bom_units(bom: pd.DataFrame, product, resource) -> float:
    """BOM units_required as Python float."""
    sel = bom.loc[(bom["product"] == str(product)) & (bom["resource"] == str(resource)), "units_required"]
    if sel.empty:
        raise KeyError(f"No BOM row for product={product!r}, resource={resource!r}")
    return float(sel.iloc[0])


# --- Solve production plan ---
def solve_production_plan(products, resources, bom, integer=False, time_limit=None):
    # Initialize LP
    model = LpProblem(name="production-plan", sense=LpMaximize)

    # Decision variables: production quantities
    qty_vars = {}
    for _, row in products.iterrows():
        p = str(row["product"])
        if integer:
            qty_vars[p] = LpVariable(
                name=f"prod_{p}",
                lowBound=row["min_demand"],
                upBound=row["max_demand"],
                cat="Integer",
            )
        else:
            qty_vars[p] = LpVariable(
                name=f"prod_{p}",
                lowBound=row["min_demand"],
                upBound=row["max_demand"],
                cat="Continuous",
            )

    # Objective: Maximize profit = revenue - material cost
    revenue = lpSum(qty_vars[p] * _scalar_price(products, p) for p in qty_vars)
    material_cost = lpSum(
        qty_vars[p] * _scalar_bom_units(bom, p, r) * float(resources.loc[r, "unit_cost"])
        for p in qty_vars
        for r in resources.index
        if not bom.loc[(bom["product"] == str(p)) & (bom["resource"] == str(r))].empty
    )
    model += revenue - material_cost, "Total_Profit"

    # Constraints: Resource availability (named for dual / shadow price lookup)
    constraint_name_by_resource = {}
    for r in resources.index:
        cn = _safe_constraint_name(r)
        constraint_name_by_resource[r] = cn
        model += (
            lpSum(
                qty_vars[p] * _scalar_bom_units(bom, p, r)
                for p in qty_vars
                if not bom.loc[(bom["product"] == str(p)) & (bom["resource"] == str(r))].empty
            )
            <= resources.loc[r, "available"],
            cn,
        )

    # Solve
    solver = PULP_CBC_CMD(msg=False, timeLimit=time_limit)
    status = model.solve(solver)

    # Prepare solution
    plan = {p: qty_vars[p].value() for p in qty_vars}

    resource_usage = {}
    resource_utilization_pct = {}
    for r in resources.index:
        used = sum(
            plan[p] * _scalar_bom_units(bom, p, r)
            for p in qty_vars
            if not bom.loc[(bom["product"] == str(p)) & (bom["resource"] == str(r))].empty
        )
        resource_usage[r] = used
        resource_utilization_pct[r] = round((used / resources.loc[r, "available"]) * 100, 2) if resources.loc[r, "available"] > 0 else 0.0

    total_revenue = sum(plan[p] * _scalar_price(products, p) for p in qty_vars)
    total_material_cost = sum(
        plan[p] * _scalar_bom_units(bom, p, r) * float(resources.loc[r, "unit_cost"])
        for p in qty_vars
        for r in resources.index
        if not bom.loc[(bom["product"] == str(p)) & (bom["resource"] == str(r))].empty
    )
    total_profit = total_revenue - total_material_cost

    # Dual values (shadow prices) — meaningful for the LP relaxation only
    shadow_prices = {}
    duals_valid = not bool(integer)
    if duals_valid:
        for r, cn in constraint_name_by_resource.items():
            con = model.constraints.get(cn)
            if con is None:
                continue
            pi = getattr(con, "pi", None)
            if pi is not None:
                try:
                    shadow_prices[str(r)] = float(pi)
                except (TypeError, ValueError):
                    shadow_prices[str(r)] = None

    return {
        "status": status,
        "plan": plan,
        "resource_usage": resource_usage,
        "resource_utilization_pct": resource_utilization_pct,
        "total_revenue": total_revenue,
        "total_material_cost": total_material_cost,
        "profit": total_profit,
        "shadow_prices": shadow_prices,
        "duals_valid": duals_valid,
        "constraint_names": {str(k): str(v) for k, v in constraint_name_by_resource.items()},
    }

# --- Save solution to JSON ---
def save_solution(sol, path="outputs/solution.json"):
    with open(path, "w") as f:
        json.dump(sol, f, indent=2)
    return path
