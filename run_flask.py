#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 13:54:55 2024

@author: noshitha
"""
import os

# Ensure the current working directory is the project directory
os.chdir('/Users/noshitha/Desktop/Github/Audio_analytics')

# Set the environment variables
os.environ['FLASK_APP'] = 'app.py'
os.environ['FLASK_DEBUG'] = '1'

# Import and run Flask CLI
from flask.cli import main
main(args=['run'])
