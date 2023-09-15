from ninja import Router
from .models import Response
from .schema import NLGSchema
import logging
import django.utils

logger = logging.getLogger(__name__)

router = Router()

@router.get('/')
def list_responses(request):
    logger.debug("listing all responses")
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
    language = payload.slots.get('language')
    if not language:
        language = 'en-US'
    logger.debug(f"NLG lookup text: {payload.text}, language payload: {payload.slots.get('language')}, language set to: {language}")
    docs = Response.objects.filter(response_id=payload.text, language=language)
    if docs.count() <= 0:
        return {"text": "response not found in database"}
    else:
        doc = docs[0]
        return {"text": doc.response_text}

@router.post('/nlg_bank/')
def lookup_response(request, payload: NLGSchema):
    logger.debug(f"NLG lookup text: {payload.text}, language: {payload.slots.get('language')}")
    docs = Response.objects.filter(response_id=payload.text, language=payload.slots.get('language'))
    if docs.count() <= 0:
        return {}
    else:
        doc = docs[0]
        return {"language": doc.language, "response_text": doc.response_text, "active": str(doc.active)}
