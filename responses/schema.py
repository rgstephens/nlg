from typing import Optional, Iterable, List, Dict, Text, Any
from .models import Response
# from django.contrib.auth.models import User
from ninja import ModelSchema, Schema

class ResponseSchema(ModelSchema):
    class Config:
        model = Response
        model_fields = "__all__"

class SlotSchema(Schema):
    name: str
    value: str

class ChannelSchema(Schema):
    name: str

class NLGSchema(Schema):
    text: str
    slots: Dict[str, Any]
    # slots: Optional[Iterable[SlotSchema]]
    channel: Optional[ChannelSchema]

        # sender_id: Text,

