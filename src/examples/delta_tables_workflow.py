"""
Delta Tables Workflow Example

This example simulates processing multiple Delta tables using the WorkflowTask,
WorkflowSubtask, and SequentialRunner infrastructure. Each subtask represents
the processing logic for a different Delta table. This pipeline runs them sequentially.
"""

from typing import List
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import lit
from db_zpark_pyf.workflow_task import WorkflowTask, TaskEnvironment
from db_zpark_pyf.workflow_subtask import WorkflowSubtask, SubtaskContext
from db_zpark_pyf.workflow_subtasks_runner import WorkflowSubtasksRunner, SequentialRunner
from pyfecto.pyio import PYIO


# ----------- Environment -----------

class DeltaTaskEnv(TaskEnvironment):
    """
    Simple TaskEnvironment that provides a Spark session for all subtasks.
    """

    def __init__(self):
        self._spark = SparkSession.builder.appName("DeltaTablesWorkflow").getOrCreate()

    @property
    def spark_session(self) -> SparkSession:
        return self._spark

    @property
    def app_name(self) -> str:
        return self._spark.sparkContext.appName


# ----------- Subtask Definition -----------

class DeltaTableSubtask(WorkflowSubtask):
    """
    A subtask that simulates reading, transforming, and writing a Delta table.
    """

    def __init__(self, table_name: str, shared_env: TaskEnvironment):
        self._table_name = table_name
        self._context = SubtaskContext(name=f"Process_{table_name}", group_id=1)
        self._env = shared_env

    @property
    def context(self) -> SubtaskContext:
        return self._context

    def _get_env(self) -> TaskEnvironment:
        return self._env

    def read_source(self, env: TaskEnvironment) -> DataFrame:
        print(f"🔍 Reading source for table: {self._table_name}")
        if self._table_name == "users":
            return env.spark_session.createDataFrame(
                [(1, "Alice"), (2, "Bob")], ["user_id", "name"]
            )
        elif self._table_name == "orders":
            return env.spark_session.createDataFrame(
                [(100, 1, 23.45), (101, 2, 54.32)], ["order_id", "user_id", "total"]
            )
        elif self._table_name == "products":
            return env.spark_session.createDataFrame(
                [(10, "Widget", 9.99), (11, "Gadget", 19.99)], ["product_id", "name", "price"]
            )
        else:
            return env.spark_session.createDataFrame(
                [], []
            )

    def transformer(self, env: TaskEnvironment, input_df: DataFrame) -> DataFrame:
        print(f"🔧 Transforming data for table: {self._table_name}")
        return input_df.withColumn("source", lit(self._table_name))

    def sink(self, env: TaskEnvironment, output_df: DataFrame) -> None:
        print(f"📦 Writing data for table: {self._table_name}")
        output_df.show()


# ----------- Main Task -----------

class DeltaTablesWorkflowTask(WorkflowTask):
    """
    Main task that coordinates execution of subtasks for multiple Delta tables.
    """

    def build_task_environment(self) -> TaskEnvironment:
        return DeltaTaskEnv()

    def start_task(self, env: TaskEnvironment) -> PYIO[Exception, None]:
        # List of mock Delta table names
        tables = ["users", "orders", "products"]

        # Create subtasks for each table
        subtasks: List[WorkflowSubtask] = [
            DeltaTableSubtask(table_name=table, shared_env=env) for table in tables
        ]

        # Use the SequentialRunner to run each table subtask
        runner: WorkflowSubtasksRunner = SequentialRunner(subtasks=subtasks)
        return runner.run()
