from enum import Enum, auto

class TaskType(Enum):
    BINARY = auto()
    MULTICLASS = auto()
    MULTILABEL = auto()