"""
Entry point script for running the application.
"""
from gandalf import create_app
from gandalf.config.settings import get_config

def main():
    """Run the application."""
    config = get_config()
    app = create_app()
    app.run(
        host=config.HOST,
        port=config.PORT,
        debug=config.DEBUG
    )

if __name__ == '__main__':
    main() 