# 🏭 Production Planning & Strategy Optimization Engine

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_svg)](https://appuctionplanningoptimizationlinearprogramming-6wd4sg3wmwesrxp.streamlit.app/)

## 📌 Strategic Overview
In high-stakes manufacturing, manual scheduling often results in **sub-optimal resource allocation** and hidden **operational costs**. This application leverages **Linear Programming (LP)** to solve complex production planning problems, maximizing profitability while respecting rigid resource constraints.

## 🛑 Business Problem
- **Resource Underutilization:** Difficulty in balancing machine hours, labor, and raw material availability.
- **Complexity at Scale:** Traditional spreadsheet-based planning fails as product variants and constraint sets grow.
- **Decision Latency:** Planners need a real-time "What-If" simulator to understand the impact of supply chain disruptions.

## 🎯 Solution Objectives
1. **Mathematical Optimization:** Utilize the Simplex algorithm (via PuLP/SciPy) to find the absolute global maximum for profit.
2. **Dynamic Constraint Modeling:** Enable real-time adjustment of labor hours, material costs, and machine capacity.
3. **Executive Visualization:** Convert abstract linear inequalities into intuitive graphical dashboards for C-suite stakeholders.
4. **Staff-Level Transparency:** Provide shadow price analysis to identify which constraints are most limiting to growth.

## ⚙️ Modular Architecture
This system is architected as a **Decision Support System (DSS)** with a clean separation between the optimization engine and the presentation layer.

```mermaid

graph TD
    A[Stakeholder Input] -->|Constraint Parameters| B(Optimization Engine)
    B -->|Linear Programming Solver| C{Feasibility Check}
    C -->|Optimal Solution| D[Decision Dashboard]
    C -->|Infeasible| E[Constraint Conflict Warning]
    D -->|Export| F[Production Schedule PDF/CSV]


    Core Features

    Multi-Product Optimization: Simultaneously balance multiple product lines with varying profit margins.

    Sensitivity Analysis: Visualize how changes in resource availability (e.g., a 10% labor shortage) impact the bottom line.

    Validation Gates: Built-in logic to prevent non-physical solutions (e.g., negative production).

    Interactive UI: Built with Streamlit for rapid prototyping and stakeholder alignment.

    Technical Stack

    Engine: Python (PuLP / SciPy / NumPy)

    UI/UX: Streamlit

    Visualization: Plotly / Matplotlib

    Deployment: GitHub Actions CI/CD for automated testing and deployment.

Author
Sherriff Abdul-Hamid Staff Data Scientist & Decision Architect

