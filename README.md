## Setup Steps

- https://django-ninja.rest-framework.com/tutorial/

```sh
django-admin startproject nlg
cd nlg
mkdir responses  # responses table
```

## Responses

- Copy existing app folder (`courses`)
- Replace all names (`course` with `response`)
- Customize `models.py`
- Update `api.py` with field names
- Update `nlg/api.py` with new router
- Add to `nlg/settings.py` 
- Run:

```sh
python manage.py makemigrations responses
python manage.py migrate
```

## Startup

```sh
python manage.py createsuperuser
python manage.py runserver
```
