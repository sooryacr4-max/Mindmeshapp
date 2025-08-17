# Mindmeshapp
An ML model trained on user prompts to generate simple mesh files in seconds for flate plate and BFS geometries of any dimension and bias values

##Getting Started
###1. Clone the repository
git clone https://github.com/sooryacr4-max/Mindmeshapp.git
###2. install requirements
pip install -r requirements.txt
###3.Generate the model, use the prompt dataset csv file to train the model
python train_classifier.py
###4. Run the streamlit app
streamlit run mind_meshapp.py


## Files
- `mind_meshapp.py` – Streamlit web application
- `train_classifier.py` – Script to build and save the ML model
- `requirements.txt` – List of required Python libraries
- `README.md` – Project information

## Contributing
Pull requests are welcome! 




