import unittest
from collections import deque
from unittest import mock

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import IntegerType, StringType, StructField, StructType

from db_zpark_pyf.workflow_subtask import SubtaskContext, WorkflowSubtask
from db_zpark_pyf.workflow_task import TaskEnvironment


class MockSparkSession:
    """A simple wrapper around a real SparkSession for testing."""

    def __init__(self):
        self.spark = (
            SparkSession.builder.appName("WorkflowSubtaskTest")
            .master("local[1]")
            .getOrCreate()
        )

    def createDataFrame(self, data, schema=None):
        return self.spark.createDataFrame(data, schema)

    def stop(self):
        self.spark.stop()


class MockTaskEnvironment(TaskEnvironment):
    """A mock implementation of TaskEnvironment for testing."""

    def __init__(self, spark_session=None, app_name="test-app"):
        self._spark = spark_session or MockSparkSession()
        self._app_name = app_name

    @property
    def spark_session(self):
        return self._spark

    @property
    def app_name(self):
        return self._app_name


class SimpleSubtask(WorkflowSubtask):
    """Basic implementation of WorkflowSubtask for testing."""

    def __init__(self, name="test-subtask", group_id=1, should_ignore_failures=False):
        self._context = SubtaskContext(name=name, group_id=group_id)
        self.ignore_and_log_failures = should_ignore_failures
        self.execution_order = deque()
        self.env = MockTaskEnvironment(spark_session=MockSparkSession())

    @property
    def context(self) -> SubtaskContext:
        return self._context

    def _get_env(self) -> TaskEnvironment:
        return self.env

    def pre_process(self, env: TaskEnvironment) -> None:
        self.execution_order.append("pre_process")

    def read_source(self, env: TaskEnvironment) -> DataFrame:
        self.execution_order.append("read_source")
        # Create a real DataFrame
        data = [("Alice", 1), ("Bob", 2), ("Charlie", 3)]
        return env.spark_session.createDataFrame(data, ["name", "value"])

    def transformer(self, env: TaskEnvironment, input_df: DataFrame) -> DataFrame:
        self.execution_order.append("transformer")
        # Transform the real DataFrame
        return input_df.selectExpr("name", "value * 2 as doubled_value")

    def sink(self, env: TaskEnvironment, output_df: DataFrame) -> None:
        self.execution_order.append("sink")
        # Just collect the data to validate it worked
        output_df.collect()

    def post_process(self, env: TaskEnvironment) -> None:
        self.execution_order.append("post_process")


class FailingSubtask(WorkflowSubtask):
    """Implementation that fails at a specific stage."""

    def __init__(
        self,
        name="failing-subtask",
        group_id=1,
        fail_at="read_source",
        should_ignore_failures=False,
    ):
        self._context = SubtaskContext(name=name, group_id=group_id)
        self.ignore_and_log_failures = should_ignore_failures
        self.fail_at = fail_at
        self.execution_order = deque()
        self.env = MockTaskEnvironment(spark_session=MockSparkSession())

    @property
    def context(self) -> SubtaskContext:
        return self._context

    def _get_env(self) -> TaskEnvironment:
        return self.env

    def pre_process(self, env: TaskEnvironment) -> None:
        self.execution_order.append("pre_process")
        if self.fail_at == "pre_process":
            raise ValueError(f"Simulated failure in {self.fail_at}")

    def read_source(self, env: TaskEnvironment) -> DataFrame:
        self.execution_order.append("read_source")
        if self.fail_at == "read_source":
            raise ValueError(f"Simulated failure in {self.fail_at}")

        # Create a real DataFrame
        data = [("Alice", 1), ("Bob", 2), ("Charlie", 3)]
        return env.spark_session.createDataFrame(data, ["name", "value"])

    def transformer(self, env: TaskEnvironment, input_df: DataFrame) -> DataFrame:
        self.execution_order.append("transformer")
        if self.fail_at == "transformer":
            raise ValueError(f"Simulated failure in {self.fail_at}")

        # Apply a real transformation
        return input_df.selectExpr("name", "value * 2 as doubled_value")

    def sink(self, env: TaskEnvironment, output_df: DataFrame) -> None:
        self.execution_order.append("sink")
        if self.fail_at == "sink":
            raise ValueError(f"Simulated failure in {self.fail_at}")

        # Actually execute the DataFrame operation
        output_df.collect()

    def post_process(self, env: TaskEnvironment) -> None:
        self.execution_order.append("post_process")
        if self.fail_at == "post_process":
            raise ValueError(f"Simulated failure in {self.fail_at}")


class TestWorkflowSubtask(unittest.TestCase):
    """Tests for the WorkflowSubtask implementation."""

    def setUp(self):
        """Set up for tests with log capture."""
        # Create a list for log capture
        self.log_messages = []

        # Mock the logger to capture logs
        self.patcher = mock.patch("pyfecto.runtime.LOGGER.info")
        self.mock_logger_info = self.patcher.start()
        self.mock_logger_info.side_effect = (
            lambda msg, **kwargs: self.log_messages.append(msg)
        )

        self.error_patcher = mock.patch("pyfecto.runtime.LOGGER.error")
        self.mock_logger_error = self.error_patcher.start()
        self.mock_logger_error.side_effect = (
            lambda msg, **kwargs: self.log_messages.append(msg)
        )

        # Configure Spark to be less verbose during tests
        import logging

        logging.getLogger("py4j").setLevel(logging.ERROR)

    def tearDown(self):
        """Clean up after tests."""
        self.patcher.stop()
        self.error_patcher.stop()

        # Try to clean up any Spark sessions
        try:
            SparkSession.builder.getOrCreate().stop()
        except Exception:
            pass

    def test_successful_execution(self):
        """Test that a subtask executes all stages in the correct order."""
        subtask = SimpleSubtask()
        result = subtask.run().run()

        # Verify result
        self.assertIsNone(result, "Subtask should complete successfully")

        # Verify execution order
        expected_order = [
            "pre_process",
            "read_source",
            "transformer",
            "sink",
            "post_process",
        ]
        actual_order = list(subtask.execution_order)
        self.assertEqual(
            expected_order, actual_order, "Steps should execute in the correct order"
        )

        # Verify logging
        self.assertIn(f"Starting subtask {subtask.context.name}", self.log_messages)
        self.assertIn("Finished pre-processing", self.log_messages)
        self.assertIn("Finished sink", self.log_messages)
        self.assertIn(f"Finished subtask {subtask.context.name}", self.log_messages)

    def test_failure_without_ignore(self):
        """Test that a failing subtask returns the error if ignore_and_log_failures is False."""
        subtask = FailingSubtask(fail_at="transformer", should_ignore_failures=False)
        result = subtask.run().run()

        # Verify result is an error
        self.assertIsInstance(result, ValueError)
        self.assertEqual(str(result), "Simulated failure in transformer")

        # Verify execution order (should stop at failure)
        expected_order = ["pre_process", "read_source", "transformer"]
        actual_order = list(subtask.execution_order)
        self.assertEqual(
            expected_order, actual_order, "Execution should stop at failure point"
        )

        # Verify error logging
        self.assertIn(
            f"Subtask {subtask.context.name} failed: Simulated failure in transformer",
            self.log_messages,
        )

    def test_failure_with_ignore(self):
        """Test that a failing subtask continues if ignore_and_log_failures is True."""
        subtask = FailingSubtask(fail_at="transformer", should_ignore_failures=True)
        result = subtask.run().run()

        # Verify result (should be None since we're ignoring failures)
        self.assertIsNone(
            result,
            "Subtask should complete despite error when ignore_and_log_failures=True",
        )

        # Verify execution order (should still have executed up to failure point)
        expected_order = ["pre_process", "read_source", "transformer"]
        actual_order = list(subtask.execution_order)
        self.assertEqual(
            expected_order, actual_order, "Execution should record steps up to failure"
        )

        # Verify error logging
        self.assertIn(
            f"Subtask {subtask.context.name} failed: Simulated failure in transformer",
            self.log_messages,
        )

    def test_early_stage_failure(self):
        """Test failure at an early stage (pre-process)."""
        subtask = FailingSubtask(fail_at="pre_process")
        result = subtask.run().run()

        # Verify result is an error
        self.assertIsInstance(result, ValueError)

        # Verify execution order
        expected_order = ["pre_process"]
        actual_order = list(subtask.execution_order)
        self.assertEqual(expected_order, actual_order)

    def test_late_stage_failure(self):
        """Test failure at a late stage (post-process)."""
        subtask = FailingSubtask(fail_at="post_process")
        result = subtask.run().run()

        # Verify result is an error
        self.assertIsInstance(result, ValueError)

        # Verify execution order
        expected_order = [
            "pre_process",
            "read_source",
            "transformer",
            "sink",
            "post_process",
        ]
        actual_order = list(subtask.execution_order)
        self.assertEqual(expected_order, actual_order)

    def test_custom_context(self):
        """Test subtask with custom context values."""
        custom_name = "custom-subtask"
        custom_group = 42
        subtask = SimpleSubtask(name=custom_name, group_id=custom_group)

        # Verify context properties
        self.assertEqual(subtask.context.name, custom_name)
        self.assertEqual(subtask.context.group_id, custom_group)

        # Run to verify context is used in logging
        subtask.run().run()
        self.assertIn(f"Starting subtask {custom_name}", self.log_messages)

    def test_env_not_implemented(self):
        """Test that a bare subtask without _get_env implementation fails properly."""

        # Create a minimal subtask class that inherits the base implementation of _get_env
        # which should raise NotImplementedError
        class MinimalSubtask(WorkflowSubtask):
            @property
            def context(self):
                return SubtaskContext("minimal", 1)

            def read_source(self, env):
                data = [("Test", 1)]
                schema = StructType(
                    [
                        StructField("name", StringType(), False),
                        StructField("value", IntegerType(), False),
                    ]
                )
                return env.spark_session.createDataFrame(data, schema)

            def transformer(self, env, input_df):
                return input_df

            def sink(self, env, output_df):
                pass

        subtask = MinimalSubtask()

        # Execute the subtask and verify it fails
        result = subtask.run().run()

        # The result should be a NotImplementedError
        self.assertIsInstance(result, NotImplementedError)
        self.assertIn("TaskEnvironment must be provided", str(result))
