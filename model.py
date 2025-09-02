# model.py
import pandas as pd
import json
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, PULP_CBC_CMD

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

    bom["units_required"] = bom["units_required"].astype(float)

    # Set resource index for easy access
    resources.set_index("resource", inplace=True)

    return products, resources, bom

# --- Solve production plan ---
def solve_production_plan(products, resources, bom, integer=False, time_limit=None):
    # Initialize LP
    model = LpProblem(name="production-plan", sense=LpMaximize)

    # Decision variables: production quantities
    qty_vars = {}
    for _, row in products.iterrows():
        if integer:
            qty_vars[row["product"]] = LpVariable(name=f"prod_{row['product']}", lowBound=row["min_demand"],
                                                  upBound=row["max_demand"], cat="Integer")
        else:
            qty_vars[row["product"]] = LpVariable(name=f"prod_{row['product']}", lowBound=row["min_demand"],
                                                  upBound=row["max_demand"], cat="Continuous")

    # Objective: Maximize profit = revenue - material cost
    revenue = lpSum(qty_vars[p] * float(products.loc[products["product"]==p, "price"]) for p in qty_vars)
    material_cost = lpSum(
        qty_vars[p] * float(bom.loc[(bom["product"]==p) & (bom["resource"]==r), "units_required"]) * resources.loc[r, "unit_cost"]
        for p in qty_vars for r in resources.index if not bom.loc[(bom["product"]==p) & (bom["resource"]==r)].empty
    )
    model += revenue - material_cost, "Total_Profit"

    # Constraints: Resource availability
    for r in resources.index:
        model += lpSum(
            qty_vars[p] * float(bom.loc[(bom["product"]==p) & (bom["resource"]==r), "units_required"])
            for p in qty_vars if not bom.loc[(bom["product"]==p) & (bom["resource"]==r)].empty
        ) <= resources.loc[r, "available"], f"Resource_{r}_Constraint"

    # Solve
    solver = PULP_CBC_CMD(msg=False, timeLimit=time_limit)
    status = model.solve(solver)

    # Prepare solution
    plan = {p: qty_vars[p].value() for p in qty_vars}

    resource_usage = {}
    resource_utilization_pct = {}
    for r in resources.index:
        used = sum(
            plan[p] * float(bom.loc[(bom["product"]==p) & (bom["resource"]==r), "units_required"])
            for p in qty_vars if not bom.loc[(bom["product"]==p) & (bom["resource"]==r)].empty
        )
        resource_usage[r] = used
        resource_utilization_pct[r] = round((used / resources.loc[r, "available"]) * 100, 2) if resources.loc[r, "available"] > 0 else 0.0

    total_revenue = sum(plan[p] * float(products.loc[products["product"]==p, "price"]) for p in qty_vars)
    total_material_cost = sum(
        plan[p] * float(bom.loc[(bom["product"]==p) & (bom["resource"]==r), "units_required"]) * resources.loc[r, "unit_cost"]
        for p in qty_vars for r in resources.index if not bom.loc[(bom["product"]==p) & (bom["resource"]==r)].empty
    )
    total_profit = total_revenue - total_material_cost

    return {
        "status": status,
        "plan": plan,
        "resource_usage": resource_usage,
        "resource_utilization_pct": resource_utilization_pct,
        "total_revenue": total_revenue,
        "total_material_cost": total_material_cost,
        "profit": total_profit
    }

# --- Save solution to JSON ---
def save_solution(sol, path="outputs/solution.json"):
    with open(path, "w") as f:
        json.dump(sol, f, indent=2)
    return path
