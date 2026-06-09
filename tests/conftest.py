# Pre-import modules that PySide6's shibokensupport otherwise intercepts
# on this Python 3.12 / PySide6 environment, causing an AttributeError.
import six  # noqa: F401
import dateutil  # noqa: F401
import pandas  # noqa: F401
