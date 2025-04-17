import subprocess

# Run the Streamlit app
subprocess.run(["python", "-m", "streamlit", "run", "OEIS_Classification_Tool/OEIS_Streamlit_App.py"])

# Whenever wanting to package a new version of the app, run the following in your bash terminal - requires you to "pip install pyinstaller" first
# python -m PyInstaller --noconfirm --onefile --clean --console --name OEIS_SequenceTool RunApp.py