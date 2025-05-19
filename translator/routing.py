from django.urls import re_path
from . import consumers

websocket_urlpatterns = [
    re_path(r'ws/translator/(?P<session_id>\w+)/$', consumers.TranslatorConsumer.as_asgi()),
]
