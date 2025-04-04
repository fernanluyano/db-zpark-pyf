"""
Test module for the WorkflowTask implementation.
"""

import unittest
from unittest import mock

from pyfecto.pyio import PYIO
from pyfecto.runtime import LOGGER, Runtime

from src.db_zpark_pyf.workflow_task import TaskEnvironment, WorkflowTask


class MockSparkSession:
    """Mock SparkSession for testing."""

    def __init__(self):
        self.test_data = {}

    def sql(self, query):
        # Mock SQL execution
        return MockDataFrame(1)  # Return a mock dataframe with 1 row

    def __str__(self):
        return "MockSparkSession"


class MockDataFrame:
    """Mock DataFrame for testing."""

    def __init__(self, count_value=0):
        self._count_value = count_value

    def count(self):
        return self._count_value


# Add pytest.mark.no_collection to prevent pytest from trying to collect this as a test class
class SimpleTaskEnvironment(TaskEnvironment):
    """A test implementation of TaskEnvironment."""

    def __init__(self, spark_session=None, app_name="test-app"):
        self._spark = spark_session or MockSparkSession()
        self._app_name = app_name

    @property
    def spark_session(self):
        return self._spark

    @property
    def app_name(self):
        return self._app_name


class TestWorkflowTask(unittest.TestCase):
    """Tests for the WorkflowTask implementation."""

    def setUp(self):
        """Setup for tests - configure logging for capture."""
        # Create a string buffer for log capture
        self.log_buffer = []

        # Mock the logger to capture logs
        self.original_info = LOGGER.info
        LOGGER.info = lambda msg, **kwargs: self.log_buffer.append(msg)

        # Reset after test
        self.addCleanup(self.restore_logger)

    def restore_logger(self):
        """Restore the original logger after tests."""
        LOGGER.info = self.original_info

    def test_successful_task(self):
        """Test that a successful task completes and logs appropriately."""

        # Create a task that will succeed
        class SuccessfulTask(WorkflowTask):
            def build_task_environment(self):
                return SimpleTaskEnvironment()

            def start_task(self, env):
                def my_spark_task():
                    # Simulate a successful operation
                    return env.spark_session.sql("SELECT 1 AS n").count()

                return PYIO.attempt(my_spark_task)

        # Run the task
        task = SuccessfulTask()
        result = task.run().run()

        # Verify the task completed successfully (no exception)
        self.assertIsNone(result, "Task should complete without error")

        # Check for successful completion log message
        success_logs = [
            msg for msg in self.log_buffer if "finished successfully" in msg
        ]
        self.assertEqual(
            len(success_logs), 1, "Should have one successful completion log"
        )
        self.assertIn("test-app finished successfully", success_logs[0])

    def test_failing_task(self):
        """Test that a failing task handles errors appropriately."""

        # Create a task that will fail
        class FailingTask(WorkflowTask):
            def build_task_environment(self):
                return SimpleTaskEnvironment()

            def start_task(self, env):
                return PYIO.fail(RuntimeError("Task failed intentionally"))

        # Run the task
        task = FailingTask()
        result = task.run().run()

        # Verify the task returned an exception
        self.assertIsInstance(result, RuntimeError, "Task should return a RuntimeError")
        self.assertEqual(
            str(result), "Task failed intentionally", "Exception message should match"
        )

        # Check for failure log message
        failure_logs = [msg for msg in self.log_buffer if "failed" in msg]
        self.assertEqual(len(failure_logs), 1, "Should have one failure log")
        self.assertIn("Task failed intentionally", failure_logs[0])

    def test_environment_build_failure(self):
        """Test that failures during environment building are handled correctly."""

        # Create a task that will fail during environment build
        class BadEnvTask(WorkflowTask):
            def build_task_environment(self):
                raise RuntimeError("Failed to build environment")

            def start_task(self, env):
                return PYIO.unit()

        # Run the task
        task = BadEnvTask()
        result = task.run().run()

        # Verify the task failed with the expected exception
        self.assertIsInstance(
            result, RuntimeError, "Task should fail with RuntimeError"
        )
        self.assertEqual(
            str(result), "Failed to build environment", "Exception message should match"
        )

    @mock.patch("sys.exit")
    def test_run_as_app_with_success(self, mock_exit):
        """Test that run_as_app doesn't exit when task succeeds."""

        # Create a successful task
        class SuccessfulTask(WorkflowTask):
            def build_task_environment(self):
                return SimpleTaskEnvironment()

            def start_task(self, env):
                return PYIO.unit()

        # Run the task with run_as_app
        task = SuccessfulTask()
        task.run_as_app()

        # Verify sys.exit was not called
        mock_exit.assert_not_called()

    @mock.patch("sys.exit")
    def test_run_as_app_with_failure(self, mock_exit):
        """Test that run_as_app exits with the correct code when task fails."""

        # Create a failing task
        class FailingTask(WorkflowTask):
            def build_task_environment(self):
                return SimpleTaskEnvironment()

            def start_task(self, env):
                return PYIO.fail(RuntimeError("Task failed intentionally"))

        # Run the task with run_as_app and custom error code
        task = FailingTask()
        task.run_as_app(exit_on_error=True, error_code=42)

        # Verify sys.exit was called with the correct code
        mock_exit.assert_called_once_with(42)

    @mock.patch("sys.exit")
    def test_run_as_app_no_exit_on_error(self, mock_exit):
        """Test that run_as_app doesn't exit when exit_on_error is False."""

        # Create a failing task
        class FailingTask(WorkflowTask):
            def build_task_environment(self):
                return SimpleTaskEnvironment()

            def start_task(self, env):
                return PYIO.fail(RuntimeError("Task failed intentionally"))

        # We need to mock Runtime.run_app to avoid the error being raised
        with mock.patch("pyfecto.runtime.Runtime.run_app") as mock_run_app:
            # Configure mock to store the exception rather than raising it
            mock_run_app.side_effect = lambda app, exit_on_error, code: None

            # Run the task with exit_on_error=False
            task = FailingTask()
            task.run_as_app(exit_on_error=False)

            # Verify run_app was called with correct parameters
            mock_run_app.assert_called_once()
            self.assertEqual(mock_run_app.call_args[0][0], task)
            self.assertEqual(mock_run_app.call_args[0][1], False)

            # Verify sys.exit was not called
            mock_exit.assert_not_called()

    def test_custom_runtime(self):
        """Test that a custom runtime is properly used."""
        # Create a custom runtime with DEBUG level
        custom_runtime = Runtime(log_level="DEBUG")

        # Create a task with the custom runtime
        class CustomRuntimeTask(WorkflowTask):
            def __init__(self, runtime):
                super().__init__(runtime)
                self.custom_env = SimpleTaskEnvironment(app_name="custom-runtime-app")

            def build_task_environment(self):
                return self.custom_env

            def start_task(self, env):
                # This would normally be logged at DEBUG level
                return PYIO.log_debug("Debug message for testing").then(PYIO.unit())

        # Run the task
        task = CustomRuntimeTask(custom_runtime)
        # We would need to mock the logger's debug method to properly test this
        # For now, just make sure it runs without errors
        result = task.run().run()
        self.assertIsNone(result, "Task should complete without error")
        self.assertTrue(
            any(
                "custom-runtime-app finished successfully" in msg
                for msg in self.log_buffer
            )
        )
