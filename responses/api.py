from ninja import Router
from .models import Response
from .schema import NLGSchema
import django.utils

router = Router()

@router.get('/')
def list_responses(request):
    return [
        {"id": d.response_id, "language": d.language}
        for d in Response.objects.all()
    ]

@router.get('/{response_id}')
def response_details(request, response_id: int):
    doc = Response.objects.get(id=response_id)
    return {"language": doc.language, "response_text": doc.response_text, "active": True}

@router.post('/nlg/')
def lookup_response(request, payload: NLGSchema):
    docs = Response.objects.filter(response_id=payload.text, language=payload.slots.get('language'))
    doc = docs[0]
    return {"language": doc.language, "response_text": doc.response_text, "active": str(doc.active)}
