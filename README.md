# db-zpark-pyf 🐍

A lightweight framework for writing modular, testable, and expressive **data workflows** in Python using PySpark and [pyfecto](https://github.com/databrickslabs/pyfecto).

This is the **Python sibling of [`db-zpark`](https://github.com/fernanluyano/db-zpark)**, following similar principles for separation of concerns, functional programming, and structured workflow execution.

---

## ✨ Key Concepts

- **WorkflowTask**: Top-level job orchestrator (e.g. for a pipeline or table group).
- **WorkflowSubtask**: A reusable unit of work representing a single table or logical step.
- **WorkflowSubtasksRunner**: A strategy to execute a collection of subtasks (e.g. sequentially).
- **TaskEnvironment**: Shared resources like `SparkSession`, passed to all components.
- **Effect system**: All execution is managed through `PYIO` (from `pyfecto`) for clean logging, retrying, and chaining.

---

## 🚀 Example: Simulating a Delta Pipeline

Each table (`users`, `orders`, `products`) is handled by its own `WorkflowSubtask`, and all are coordinated by a `WorkflowTask` with a sequential runner.

🧪 Check the example here:  
➡️ [`examples/delta_tables_workflow.py`](./src/examples/delta_tables_workflow.py)

---

## 🔁 Databricks Runtime Compatibility

| db-zpark-pyf | Pyfecto | Python | Spark | DBR      |
|--------------|---------|--------|-------|----------|
| 0.1.0        | 0.2.0   | 3.11   | 3.5.x | 15.4 LTS |

---

## 🛠 Development Setup

## 📦 Install via pip

To use `db-zpark-pyf` in your own project:

```bash
pip install db_zpark_pyf
```