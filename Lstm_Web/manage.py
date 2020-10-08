#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys
import nltk
import warnings
def main():
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'net.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    warnings.filterwarnings(action='ignore',category=UserWarning,module='gensim')
    warnings.filterwarnings(action='ignore',category=Warning,module='gensim')
    warnings.filterwarnings(action='ignore', category=FutureWarning, module='gensim')
    nltk.data.path.append("./net/nltk_data")
    main()
