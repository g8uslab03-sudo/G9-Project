"""GCP WSGI entrypoint for Cloud Run."""

from gcp.app import create_app

application = create_app()
