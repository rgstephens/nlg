from ninja import NinjaAPI
# from .authentication import GlobalAuhentication
from responses.api import router as responses_router

api = NinjaAPI()
# api = NinjaAPI(auth=GlobalAuhentication())

api.add_router("/responses/", responses_router, tags=["Responses"])
