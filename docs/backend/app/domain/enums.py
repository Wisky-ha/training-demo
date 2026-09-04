"""Closed sets of values used by the model-training domain."""

from enum import Enum


class ModelType(str, Enum):
    """The model families supported by the platform."""

    ELECTRIC_LOAD = "electric_load"
    HEATING_COOLING_LOAD = "heating_cooling_load"
    INTEGRATED_ENERGY = "integrated_energy"


class ScriptType(str, Enum):
    """Kinds of Python scripts that can be selected by a training job."""

    PREPROCESSOR = "preprocessor"
    TRAINER = "trainer"


class ScriptStatus(str, Enum):
    """Availability of a script in the global script library."""

    ENABLED = "ENABLED"
    DISABLED = "DISABLED"


class SplitStrategy(str, Enum):
    """Dataset split strategy persisted with a training job/version."""

    TIME_ORDERED = "time_ordered"


class TrainingJobStatus(str, Enum):
    """Lifecycle and execution stages of a training job."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    PREPROCESSING = "PREPROCESSING"
    SPLITTING = "SPLITTING"
    TRAINING = "TRAINING"
    EVALUATING = "EVALUATING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class ModelVersionStatus(str, Enum):
    """Lifecycle status of a saved model version."""

    DRAFT = "DRAFT"
    TRAINING = "TRAINING"
    READY = "READY"
    PUBLISHED = "PUBLISHED"
    RETIRED = "RETIRED"
    ABNORMAL = "ABNORMAL"
    FAILED = "FAILED"


class AlertStatus(str, Enum):
    """Status of a model-type anomaly alert.

    ``RESOLVED`` is also the normal/no-open-alert value for the model type
    record.  An alert is only resolved by a successful model publication.
    """

    ACTIVE = "ACTIVE"
    RESOLVED = "RESOLVED"


class RollbackStatus(str, Enum):
    """Execution status of an automatic or manual rollback record."""

    PENDING = "PENDING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
