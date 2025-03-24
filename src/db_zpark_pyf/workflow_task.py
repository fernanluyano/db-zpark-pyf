import abc
import time
from typing import Optional

from pyfecto.app import PyfectoApp
from pyfecto.pyio import PYIO
from pyfecto.runtime import Runtime
from pyspark.sql import SparkSession


class TaskEnvironment:
    """
    Dependencies for the WorkflowTask.
    Since it's a Spark application, at least a SparkSession and application name should be provided.
    """

    @property
    @abc.abstractmethod
    def spark_session(self) -> SparkSession:
        """
        Clients will provide a custom-built SparkSession.

        Returns:
            SparkSession: The Spark session for this task
        """
        pass

    @property
    @abc.abstractmethod
    def app_name(self) -> str:
        """
        The Spark application name.

        Returns:
            str: The name of the application
        """
        pass


class WorkflowTask(PyfectoApp[Exception], abc.ABC):
    """
    The interface for defining a Databricks workflow task using pyfecto.
    Handles environment setup, execution, and error management.
    """

    def __init__(self, runtime: Optional[Runtime] = None):
        """
        Initialize a WorkflowTask with an optional runtime configuration.

        Args:
            runtime: Optional Runtime instance. If not provided, will create a default runtime.
        """
        super().__init__(runtime)

    def run_as_app(self, exit_on_error: bool = True, error_code: int = 1) -> None:
        Runtime.run_app(self, exit_on_error, error_code)

    def run(self) -> PYIO[Exception, None]:
        """
        Runs the workflow task, initializing the environment and executing the task.

        Returns:
            PYIO effect containing the task execution
        """
        start_time_ns = time.time_ns()

        def task_flow(env: TaskEnvironment) -> PYIO[Optional[Exception], None]:
            return (
                PYIO.log_info(f"Starting task: {env.app_name}")
                .then(self.start_task(env))
                .match_pyio(
                    success=lambda _: self._happy_path(start_time_ns, env),
                    failure=lambda e: self._sad_path(start_time_ns, env, e),
                )
            )

        return PYIO.attempt(self.build_task_environment).flat_map(task_flow)

    def _happy_path(
        self, start_time_ns: int, env: TaskEnvironment
    ) -> PYIO[Optional[Exception], None]:
        """
        Terminates the task successfully with timing information.

        Args:
            start_time_ns: The start time in nanoseconds
            env: The task environment

        Returns:
            PYIO effect for successful completion
        """
        elapsed_seconds = self._get_elapsed_seconds(start_time_ns)
        return PYIO.log_info(
            f"Task {env.app_name} finished successfully in {elapsed_seconds:.2f} seconds"
        )

    def _sad_path(
        self, start_time_ns: int, env: TaskEnvironment, cause: Exception
    ) -> PYIO[Exception, None]:
        """
        Logs the failure message and returns a failed PYIO effect.

        Args:
            start_time_ns: The start time in nanoseconds
            env: The task environment
            cause: The exception that caused the failure

        Returns:
            PYIO effect for task failure
        """
        elapsed_seconds = self._get_elapsed_seconds(start_time_ns)
        message = f"Task {env.app_name} failed in {elapsed_seconds:.2f} seconds due to: {cause}"

        return PYIO.log_info(message).then(PYIO.fail(cause))

    @staticmethod
    def _get_elapsed_seconds(start_time_ns: int) -> float:
        """
        Calculates the elapsed time in seconds.

        Returns:
            float: Elapsed time in seconds
        """
        return (time.time_ns() - start_time_ns) / 1_000_000_000

    @abc.abstractmethod
    def build_task_environment(self) -> TaskEnvironment:
        """
        Builds the task execution environment: external dependencies, basic info about the task, etc.

        Returns:
            TaskEnvironment: The environment for this task execution
        """
        pass

    @abc.abstractmethod
    def start_task(self, env: TaskEnvironment) -> PYIO[Exception, None]:
        """
        Defines the main logic of the task.

        Args:
            env: The task environment with dependencies

        Returns:
            PYIO effect representing the task's main logic
        """
        pass
