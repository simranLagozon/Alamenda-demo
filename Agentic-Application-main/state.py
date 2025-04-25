# state.py
from asyncio import Lock

# Shared session state and lock
session_state = {}

session_lock = Lock()