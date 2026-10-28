import uuid


class Uuid4IdentityFactory:
    """IdentityFactory adapter. Replaces the inline uuid.uuid4() in the old routes."""

    def new_id(self) -> str:
        return str(uuid.uuid4())
