import unittest
from unittest import mock

from pyfecto.pyio import PYIO
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import IntegerType, StringType, StructField, StructType

from db_zpark_pyf.workflow_subtask import SubtaskContext, WorkflowSubtask
from db_zpark_pyf.workflow_subtasks_runner import (SequentialRunner,
                                                   WorkflowSubtasksRunner)
from db_zpark_pyf.workflow_task import TaskEnvironment


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


class MockSparkSession:
    """A simple wrapper around a real SparkSession for testing."""

    def __init__(self):
        self.spark = (
            SparkSession.builder.appName("WorkflowSubtasksRunnerTest")
            .master("local[1]")
            .getOrCreate()
        )

    def createDataFrame(self, data, schema=None):
        return self.spark.createDataFrame(data, schema)

    def stop(self):
        self.spark.stop()


class TestSubtask(WorkflowSubtask):
    """Basic subtask implementation for testing."""

    def __init__(self, name, success=True, env=None):
        self._context = SubtaskContext(name=name, group_id=1)
        self.success = success
        self.executed = False
        self.env = env or MockTaskEnvironment(spark_session=MockSparkSession())

    @property
    def context(self) -> SubtaskContext:
        return self._context

    def _get_env(self) -> TaskEnvironment:
        return self.env

    def pre_process(self, env: TaskEnvironment) -> None:
        print(f"Pre-processing {self.context.name}")
        pass

    def read_source(self, env: TaskEnvironment) -> DataFrame:
        print(f"Reading source for {self.context.name}")
        data = [(self.context.name, 1)]
        schema = StructType(
            [
                StructField("name", StringType(), False),
                StructField("value", IntegerType(), False),
            ]
        )
        return env.spark_session.createDataFrame(data, schema)

    def transformer(self, env: TaskEnvironment, input_df: DataFrame) -> DataFrame:
        print(f"Transforming data for {self.context.name}")
        self.executed = True  # Mark as executed here
        if not self.success:
            raise ValueError(f"Simulated failure in {self.context.name}")
        return input_df

    def sink(self, env: TaskEnvironment, output_df: DataFrame) -> None:
        print(f"Sinking data for {self.context.name}")
        # Just collect to verify DataFrame operations
        output_df.collect()

    def post_process(self, env: TaskEnvironment) -> None:
        print(f"Post-processing {self.context.name}")
        pass


class TestWorkflowSubtasksRunner(unittest.TestCase):
    """Tests for the WorkflowSubtasksRunner implementation."""

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
            # Log or handle specific cleanup failures if needed
            pass

    def test_sequential_runner_empty(self):
        """Test running with an empty list of subtasks."""
        runner = SequentialRunner([])
        result = runner.run().run()

        # Should complete successfully with no tasks
        self.assertIsNone(result)

    def test_sequential_runner_success(self):
        """Test running multiple successful subtasks in sequence."""
        # Create several test subtasks and store their IDs
        subtasks = [
            TestSubtask("task1", success=True),
            TestSubtask("task2", success=True),
            TestSubtask("task3", success=True),
        ]

        # Print initial object IDs and executed status
        print("\n=== BEFORE EXECUTION ===")
        for i, subtask in enumerate(subtasks):
            print(
                f"Subtask {i} ({subtask.context.name}) - ID: {id(subtask)}, executed: {subtask.executed}"
            )

        # Run the tasks sequentially
        runner = SequentialRunner(subtasks)
        print(f"Runner ID: {id(runner)}, Subtasks list ID: {id(runner.subtasks)}")

        # Execute and capture result
        result = runner.run().run()

        # Print final object IDs and executed status
        print("\n=== AFTER EXECUTION ===")
        for i, subtask in enumerate(subtasks):
            print(
                f"Subtask {i} ({subtask.context.name}) - ID: {id(subtask)}, executed: {subtask.executed}"
            )

        # Debug runner's subtasks too
        print("\n=== RUNNER'S SUBTASKS ===")
        for i, subtask in enumerate(runner.subtasks):
            print(
                f"Runner's subtask {i} ({subtask.context.name}) - ID: {id(subtask)}, executed: {subtask.executed}"
            )

        # Should complete successfully
        self.assertIsNone(result)

        # All tasks should have been executed
        for i, subtask in enumerate(subtasks):
            self.assertTrue(
                subtask.executed,
                f"Subtask {subtask.context.name} should have been executed",
            )

    def test_sequential_runner_failure(self):
        """Test that a failure in one subtask stops execution and returns the error."""
        # Create a mix of successful and failing subtasks
        subtasks = [
            TestSubtask("task1", success=True),
            TestSubtask("task2", success=False),  # This one will fail
            TestSubtask("task3", success=True),  # This one should not run
        ]

        # Run the tasks sequentially
        runner = SequentialRunner(subtasks)
        result = runner.run().run()

        # Should have failed with the error from task2
        self.assertIsInstance(result, ValueError)
        self.assertEqual(str(result), "Simulated failure in task2")

        # First two tasks should have been executed, but not the third
        self.assertTrue(subtasks[0].executed)
        self.assertTrue(subtasks[1].executed)
        self.assertFalse(subtasks[2].executed)

        # Should have logged the error
        error_logs = [msg for msg in self.log_messages if "failed" in msg]
        self.assertGreaterEqual(len(error_logs), 1)
        self.assertIn("task2 failed", error_logs[0])

    def test_runner_with_shared_environment(self):
        """Test running subtasks with a shared environment."""
        # Create a shared environment
        shared_env = MockTaskEnvironment(spark_session=MockSparkSession())

        # Create subtasks that use the shared environment
        subtasks = [
            TestSubtask("task1", env=shared_env),
            TestSubtask("task2", env=shared_env),
            TestSubtask("task3", env=shared_env),
        ]

        # Run the tasks sequentially
        runner = SequentialRunner(subtasks)
        result = runner.run().run()

        # Should complete successfully
        self.assertIsNone(result)

        # All tasks should have been executed
        for subtask in subtasks:
            self.assertTrue(subtask.executed)

            # Verify they all used the same environment
            self.assertIs(subtask.env, shared_env)

    def test_custom_runner_class(self):
        """Test creating a custom runner class with additional functionality."""

        # Define a custom runner that adds logging
        class LoggingRunner(WorkflowSubtasksRunner):
            def run(self):
                return (
                    PYIO.log_info(f"Running {len(self.subtasks)} tasks")
                    .then(super().run())
                    .then(PYIO.log_info("All tasks completed"))
                )

        # Create some test subtasks
        subtasks = [TestSubtask("task1"), TestSubtask("task2")]

        # Run with the custom runner
        runner = LoggingRunner(subtasks)
        result = runner.run().run()

        # Should complete successfully
        self.assertIsNone(result)

        # Should have logged the custom messages
        self.assertIn("Running 2 tasks", self.log_messages)
        self.assertIn("All tasks completed", self.log_messages)
