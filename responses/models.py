from django.db import models
from django.utils import timezone

# settings: language slot, default language, valid languages = "en-US", "fr-FR", "es-ES"
#  country/language - https://www.fincher.org/Utilities/CountryLanguageList.shtml
# buttons: button_id

class Response(models.Model):
    response_id = models.CharField('Id', max_length=120, blank=True, null=True)
    language = models.CharField('Language', max_length=6, blank=True, null=True)  # Fill with system default language by default, prompt with list of valid languages
    question_text = models.CharField('Question', max_length=120, blank=True, null=True)
    response_text = models.CharField('Response', max_length=120, blank=True, null=True)
    active = models.BooleanField('Active', default=True)
    date_created = models.DateTimeField('Date created', auto_now_add=True, blank=True, null=True)
    date_modified = models.DateTimeField('Date modified', auto_now_add=True, blank=True, null=True)

    def __str__(self):
        return self.response_id
