"""
This example demonstrates how to use the WorkflowTask base class.

It defines a simple task that creates a small Spark DataFrame and prints the number of rows.
The task is executed using the run_as_app() method from the Pyfecto-based workflow framework.
"""
from pyspark.sql import SparkSession
from db_zpark_pyf.workflow_task import WorkflowTask, TaskEnvironment
from pyfecto.pyio import PYIO


class SimpleEnv(TaskEnvironment):
    def __init__(self):
        self._spark = SparkSession.builder.appName("CountRowsTask").getOrCreate()

    @property
    def spark_session(self) -> SparkSession:
        return self._spark

    @property
    def app_name(self) -> str:
        return self._spark.sparkContext.appName


class CountRowsTask(WorkflowTask):
    def build_task_environment(self) -> TaskEnvironment:
        return SimpleEnv()

    def start_task(self, env: TaskEnvironment) -> PYIO[Exception, None]:
        def logic():
            # Create a small test DataFrame with three rows
            df = env.spark_session.createDataFrame(
                [(1, "Alice"), (2, "Bob"), (3, "Charlie")],
                ["id", "name"]
            )
            # Count the number of rows in the DataFrame
            count = df.count()
            # Print the result to stdout
            print(f"Row count: {count}")

        return PYIO.attempt(logic)
