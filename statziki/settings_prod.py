# settings_prod.py
from .settings import *

DEBUG = False

ALLOWED_HOSTS = [host.strip() for host in os.environ.get('ALLOWED_HOSTS', '').split(',') if host.strip()]

SECRET_KEY = os.environ.get('DJANGO_SECRET_KEY')