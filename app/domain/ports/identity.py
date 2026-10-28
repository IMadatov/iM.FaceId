from typing import Protocol, runtime_checkable


@runtime_checkable
class IdentityFactory(Protocol):
    def new_id(self) -> str: ...
