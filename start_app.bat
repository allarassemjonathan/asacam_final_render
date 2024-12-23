@echo off

echo "Installing requirements..."
python -m pip install -r %USERPROFILE%\Downloads\asacam_deployment\requirements.txt
pip install --upgrade numpy
pip install --upgrade opencv-python

pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

python -m pip install -r %USERPROFILE%\Downloads\asacam_deployment\requirements.txt

echo "Starting Flask app..."

start http://127.0.0.1:5000/

python -m flask --app %USERPROFILE%\Downloads\asacam_deployment\tools\dist\demo_UI.py run & 

echo "Productx app launched."

pause