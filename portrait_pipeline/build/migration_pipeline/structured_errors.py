class PipelineError(RuntimeError):
    """A user-correctable pipeline contract or workflow error."""


class ValidationFailure(PipelineError):
    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("; ".join(errors))
