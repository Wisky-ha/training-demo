"""Closed sets of values used by the model-training domain."""

from enum import Enum


class ModelType(str, Enum):
    """The model families supported by the platform."""

    ELECTRIC_LOAD = "electric_load"
    HEATING_COOLING_LOAD = "heating_cooling_load"
    INTEGRATED_ENERGY = "integrated_energy"


# Canonical model_type values used by every resource link and API filter.
MODEL_TYPE_CODES = tuple(item.value for item in ModelType)


class ScriptType(str, Enum):
    """Kinds of Python scripts that can be selected by a training job."""

    PREPROCESSOR = "preprocessor"
    TRAINER = "trainer"


class ScriptStatus(str, Enum):
    """Availability of a script in the global script library."""

    ENABLED = "ENABLED"
    DISABLED = "DISABLED"


class DatasetStatus(str, Enum):
    """Parsing state of an uploaded dataset record."""

    UPLOADED = "uploaded"
    PARSED = "parsed"
    FAILED = "failed"


class HealthStatus(str, Enum):
    """Operational health, independent of a model version's lifecycle.

    ``UNKNOWN`` is intentional: an absent health check is not evidence that a
    model is healthy and must not pass a health-gated operation.
    """

    HEALTHY = "HEALTHY"
    ABNORMAL = "ABNORMAL"
    UNKNOWN = "UNKNOWN"


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


class PreprocessingTaskStatus(str, Enum):
    """Outcome of the independently observable preprocessing step."""

    WAITING = "WAITING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    SKIPPED = "SKIPPED"
    FAILED = "FAILED"


class PreprocessingStage(str, Enum):
    """Stages exposed by the preprocessing progress contract."""

    WAITING = "waiting"
    DATA_READING = "data_reading"
    PREPROCESSING = "preprocessing"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"


class ModelLifecycleStatus(str, Enum):
    """Public lifecycle values for a saved model version.

    Health is deliberately not represented here.  A version can therefore be
    ``READY`` and ``UNKNOWN`` without conflating lifecycle with health.
    """

    READY = "READY"
    PUBLISHED = "PUBLISHED"
    RETIRED = "RETIRED"
    FAILED = "FAILED"


class ModelVersionStatus(str, Enum):
    """Persisted model-version status, including internal compatibility values.

    The public lifecycle contract is :class:`ModelLifecycleStatus`.  ``DRAFT``
    and ``TRAINING`` remain for the current training executor, while
    ``ABNORMAL`` is read-only compatibility for pre-contract rows.  New code
    must record abnormality in ``health_status`` instead of this field.
    """

    DRAFT = "DRAFT"
    TRAINING = "TRAINING"
    READY = "READY"
    PUBLISHED = "PUBLISHED"
    RETIRED = "RETIRED"
    FAILED = "FAILED"
    ABNORMAL = "ABNORMAL"  # legacy rows only; never assign for new writes


# Keeping these sets next to the enum makes the boundary explicit for API and
# UI code without breaking reads of historical DRAFT/TRAINING/ABNORMAL rows.
MODEL_LIFECYCLE_STATUSES = frozenset(
    {
        ModelVersionStatus.READY,
        ModelVersionStatus.PUBLISHED,
        ModelVersionStatus.RETIRED,
        ModelVersionStatus.FAILED,
    }
)


class AlertStatus(str, Enum):
    """Alert state; acknowledgement does not resolve an alert."""

    ACTIVE = "ACTIVE"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    RESOLVED = "RESOLVED"


class RollbackStatus(str, Enum):
    """Execution status of an automatic or manual rollback record."""

    PENDING = "PENDING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
