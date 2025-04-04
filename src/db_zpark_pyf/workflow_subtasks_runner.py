from abc import ABC
from typing import Any, Callable, List

from pyfecto.pyio import PYIO

from db_zpark_pyf.workflow_subtask import WorkflowSubtask


class WorkflowSubtasksRunner(ABC):
    """
    Base class for executing a sequence of workflow subtasks.
    Provides foundational execution behavior with error logging.
    """

    def __init__(self, subtasks: List[WorkflowSubtask]):
        """
        Initialize the runner with a list of subtasks.

        Args:
            subtasks: The collection of subtasks to be executed
        """
        self.subtasks = subtasks

    def run(self) -> PYIO[Exception | None, None]:
        """
        Executes all subtasks in the configured sequence.
        The default implementation runs tasks sequentially.

        Returns:
            A PYIO effect that completes when all subtasks have been processed
        """
        if not self.subtasks:
            return PYIO.log_warning("No subtasks to run")

        seq_runs = []
        for subtask in self.subtasks:

            def make_runner(
                s: WorkflowSubtask = subtask,
            ) -> Callable[[Any], PYIO[Exception | None, None]]:
                return lambda prev: self._run_one(s)

            seq_runs.append(make_runner())

        return PYIO.pipeline(*seq_runs)

    @staticmethod
    def _run_one(subtask: WorkflowSubtask) -> PYIO[Exception | None, None]:
        """
        Executes a single subtask with error logging.

        Args:
            subtask: The subtask to execute

        Returns:
            A PYIO effect representing the execution of the subtask
        """

        def handle_failure(error: Exception) -> PYIO[Exception, None]:
            """Handle subtask failure with logging."""
            return PYIO.log_error(
                f"Subtask {subtask.context.name} failed: {str(error)}"
            ).then(PYIO.fail(error))

        return subtask.run().match_pyio(
            success=lambda _: PYIO.unit(), failure=handle_failure
        )


class SequentialRunner(WorkflowSubtasksRunner):
    """
    A runner that executes subtasks sequentially in the order provided.
    Fails fast if any subtask fails.
    """

    def __init__(self, subtasks: List[WorkflowSubtask]):
        """
        Creates a new sequential runner.

        Args:
            subtasks: The subtasks to execute in sequence
        """
        super().__init__(subtasks)
