from abc import ABC, abstractmethod
from dataclasses import dataclass

from pyfecto.pyio import PYIO
from pyspark.sql import DataFrame

from db_zpark_pyf.workflow_task import TaskEnvironment


@dataclass
class SubtaskContext:
    """
    Metadata about the subtask.
    """

    name: str
    group_id: int


class WorkflowSubtask(ABC):
    """
    A composable unit of work within a workflow that processes data through defined pipeline stages.

    This class implements a standardized sequence of operations:
    - Pre-processing setup
    - Data source reading
    - Data transformation
    - Writing to a sink
    - Post-processing and/or cleanup

    Each stage is tracked with appropriate logging for monitoring and diagnostics.
    """

    # Configure whether to ignore failures and continue
    ignore_and_log_failures: bool = False

    @property
    @abstractmethod
    def context(self) -> SubtaskContext:
        """
        Get metadata about the subtask.

        Returns:
            SubtaskContext: The context object with subtask metadata
        """
        pass

    def run(self) -> PYIO[Exception, None]:
        """
        Executes the subtask with logging and timing.

        Returns:
            PYIO effect that runs the subtask in a TaskEnvironment
        """
        return (
            PYIO.log_info(f"Starting subtask {self.context.name}")
            .then(PYIO.attempt(lambda: self._get_env()))
            .flat_map(lambda env: self._execute_with_logging(env))
        )

    def _execute_with_logging(self, env: TaskEnvironment) -> PYIO[Exception, None]:
        """
        Runs the subtask with timing and logging.

        Args:
            env: The task environment

        Returns:
            A PYIO effect with timing and logging
        """
        return PYIO.log_span(
            name=f"subtask-{self.context.name}",
            log_msg=f"Executing subtask {self.context.name}",
            operation=self._run_subtask(env),
        )

    def _get_env(self) -> TaskEnvironment:
        """
        Helper method to get the task environment from the current context.
        In a real implementation, this would likely come from somewhere else.
        """
        # This is a placeholder - in a real implementation, the environment
        # would be passed in or retrieved from a context
        raise NotImplementedError("TaskEnvironment must be provided")

    def _run_subtask(self, env: TaskEnvironment) -> PYIO[Exception | None, None]:
        """
        Executes all stages of the subtask in sequence.

        Args:
            env: The task environment containing Spark session and other dependencies

        Returns:
            A PYIO effect representing the subtask execution
        """

        def _process_transformed_data(
            transformed: DataFrame,
        ) -> PYIO[Exception | None, None]:
            """Write data to sink and run post-processing"""
            return (
                PYIO.attempt(lambda: self.sink(env, transformed))
                .then(PYIO.log_info("Finished sink"))
                .then(PYIO.attempt(lambda: self.post_process(env)))
                .then(PYIO.log_info(f"Finished subtask {self.context.name}"))
            )

        flow = (
            PYIO.attempt(lambda: self.pre_process(env))
            .then(PYIO.log_info("Finished pre-processing"))
            .then(PYIO.attempt(lambda: self.read_source(env)))
            .flat_map(lambda src: PYIO.attempt(lambda: self.transformer(env, src)))
            .flat_map(_process_transformed_data)
        )

        return flow.match_pyio(
            success=lambda _: PYIO.unit(),
            failure=lambda e: (
                PYIO.log_error(f"Subtask {self.context.name} failed: {str(e)}").then(
                    PYIO.unit() if self.ignore_and_log_failures else PYIO.fail(e)
                )
            ),
        )

    def pre_process(self, env: TaskEnvironment) -> None:
        """
        Optional pre-processing step executed before reading data.

        Args:
            env: The task environment
        """
        pass

    @abstractmethod
    def read_source(self, env: TaskEnvironment) -> DataFrame:
        """
        Reads data from a source.

        Args:
            env: The task environment

        Returns:
            A DataFrame containing the source data
        """
        pass

    @abstractmethod
    def transformer(self, env: TaskEnvironment, input_df: DataFrame) -> DataFrame:
        """
        Transforms the input dataset.

        Args:
            env: The task environment
            input_df: The input dataset to transform

        Returns:
            A transformed dataset
        """
        pass

    @abstractmethod
    def sink(self, env: TaskEnvironment, output_df: DataFrame) -> None:
        """
        Writes the transformed data to a destination.

        Args:
            env: The task environment
            output_df: The dataset to write
        """
        pass

    def post_process(self, env: TaskEnvironment) -> None:
        """
        Optional post-processing step executed after all other steps complete.

        Args:
            env: The task environment
        """
        pass
