"""Service layer.

This package keeps thin compatibility wrappers at the top level while the
implementation code is grouped by domain under subpackages:

- ``app.services.evaluation``
- ``app.services.kb``
- ``app.services.retrieval``

This keeps existing imports stable during the ongoing cleanup.
"""
