import json

notebook = {
    "cells": [
        {"cell_type": "markdown", "metadata": {}, "source": ["# NHS Demo - Run Shift+Enter"]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["import requests; print(requests.get('http://localhost:8000/health').json())"]},
        {"cell_type": "markdown", "metadata": {}, "source": ["---", "## P001 HIGH"]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["import requests; f=[5,3,1,1,1,1,1,18,15,4,3,0.68,0,0]; r=requests.post('http://localhost:8000/predict',json={'features':f}); result=r.json(); print(f"P001: {result['probability']*10:.2f} {result['confidence']}")"]},
        {"cell_type": "markdown", "metadata": {}, "source": ["---", "## P002 LOW"]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["import requests; f=[1,1,0,0,0,0,0,3,2,0,0,0.10,0,0]; r=requests.post('http://localhost:8000/predict',json={'features':f}); result=r.json(); print(f"P002: {result['probability']*10:.2f} {result['confidence']}")"]},
        {"cell_type": "markdown", "metadata": {}, "source": ["---", "## P003 HIGH"]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["import requests; f=[3,4,1,1,1,1,2,20,18,4,3,0.79,0,0]; r=requests.post('http://localhost:8000/predict',json={'features':f}); result=r.json(); print(f"P003: {result['probability']*10:.2f} {result['confidence']}")"]}
    ],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
    "nbformat": 4,
    "nbformat_minor": 4
}

with open('demo.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f)

print("Done!")
