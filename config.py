"""
config.py — the central configuration point for the application.
 
All settings are read here, in one place. Secrets (such as the API key) are retrieved from
environment variables or a .env file and are NEVER stored in the code.
 
Loading order:
  1) ANTHROPIC_API_KEY environment variable (if already set in the system);
  2) .env file in the project root (see .env.example).
"""