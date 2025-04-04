"""
This example demonstrates how to implement and run a WorkflowSubtask
using the db_zpark_pyf framework.

The subtask reads a small DataFrame of users, filters out users younger
than 18, and shows the resulting DataFrame.
"""

from pyspark.sql import SparkSession, DataFrame

from db_zpark_pyf.workflow_subtask import WorkflowSubtask, SubtaskContext
from db_zpark_pyf.workflow_task import TaskEnvironment


class ExampleEnv(TaskEnvironment):
    """
    Simple TaskEnvironment that provides a Spark session.
    """

    def __init__(self):
        self._spark = SparkSession.builder.appName("FilterAdultsSubtask").getOrCreate()

    @property
    def spark_session(self) -> SparkSession:
        return self._spark

    @property
    def app_name(self) -> str:
        return self._spark.sparkContext.appName


class FilterAdultsSubtask(WorkflowSubtask):
    """
    A WorkflowSubtask that filters out users under 18 years old from a dataset.
    """

    def __init__(self):
        self._context = SubtaskContext(name="FilterAdults", group_id=1)
        self._env = ExampleEnv()

    @property
    def context(self) -> SubtaskContext:
        return self._context

    def _get_env(self) -> TaskEnvironment:
        # Provide the environment needed to run the subtask
        return self._env

    def read_source(self, env: TaskEnvironment) -> DataFrame:
        """
        Creates a DataFrame with sample user data.

        Args:
            env (TaskEnvironment): The environment containing the Spark session.

        Returns:
            DataFrame: A DataFrame with user id, name, and age.
        """
        return env.spark_session.createDataFrame(
            [(1, "Alice", 17), (2, "Bob", 25), (3, "Charlie", 16), (4, "Dana", 30)],
            ["id", "name", "age"]
        )

    def transformer(self, env: TaskEnvironment, input_df: DataFrame) -> DataFrame:
        """
        Filters out users under the age of 18.

        Args:
            env (TaskEnvironment): The task environment.
            input_df (DataFrame): The input DataFrame.

        Returns:
            DataFrame: The filtered DataFrame.
        """
        return input_df.filter("age >= 18")

    def sink(self, env: TaskEnvironment, output_df: DataFrame) -> None:
        """
        Displays the filtered DataFrame.

        Args:
            env (TaskEnvironment): The task environment.
            output_df (DataFrame): The transformed DataFrame.
        """
        output_df.show()
