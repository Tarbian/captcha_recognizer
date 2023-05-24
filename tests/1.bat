@echo off

set IMAGE_PATH="samples\2nf26.png"

curl -X POST -F "image=@%IMAGE_PATH%" http://localhost:8000/predict