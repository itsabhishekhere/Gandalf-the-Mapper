"""
Gandalf the Mapper - Main application module.
"""
import logging
from flask import Flask
from gandalf.api.routes import api
from gandalf.config.settings import get_config
from gandalf.core.taxonomy import initialize_matcher

def create_app() -> Flask:
    """
    Create and configure the Flask application.
    
    Returns:
        Configured Flask application
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create Flask app
    app = Flask(__name__)
    
    # Load configuration
    config = get_config()
    
    # Configure app
    app.config.update(
        TEMPLATES_AUTO_RELOAD=True,
        MAX_CONTENT_LENGTH=16 * 1024 * 1024  # 16MB max file size
    )
    
    # Initialize taxonomy matcher
    initialize_matcher(str(config.TAXONOMY_FILE))
    
    # Register blueprints
    app.register_blueprint(api)
    
    return app 