# Project-Disaster-Response-Pipeline
Udacity DSND Project in conjunction with Figure Eight

## Overview
This project is part of Udacity's Data Science Nanodegree. The aim of the project is to demostrate knowlege of Data Engineering skills by preparing ETL and ML pipelines, and then using the trained model in a web application that could be used for future predictions in the field.

### Repository Contents
The repository contains the following folders and files:

   - Inside the data folder is the work related to the ETL pipeline, including the original EDA Jupyter notebook, the csv
    files containing the messages to be used in training the model, the finalised ETL Script in process_data.py and the
    DisasterResponse database created by the scipt.
   
   - Inside the models folder is the work related to the Machine Learning pipeline, and includes the original ML Jupyter
    notebook, the python script derived from the notebook in train_classifier.py. The final trained model output saved as a
    pickle file called classifier.pkl is not included in the repository due to size restrictions.
    
   - Inside the app folder is the web app with the python script to run the app called run.py and then a template folder
   containing the two front-end html files master.html and go.html.
   
### Dependencies

   - Python 3.6
   - pandas 0.23.3
   - sqlalchemy 1.1.13
   - numpy 1.12.1
   - re 2.2.1
   - nltk 3.2.5
   - sklearn 0.19.1
   - pickle
   - Flask
   - plotly
   - json

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database. 
        `python data/process_data.py data/messages.csv data/categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves. 
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
