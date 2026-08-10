"""
app.py — "Chat with documents" using the Claude API + Streamlit.

Settings come from config.py (which reads .env). No secrets in this file.

Run:
    streamlit run app.py
"""

import datetime

import streamlit as st
import anthropic

