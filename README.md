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
    notebook, the python script derived from the notebook in train_classifier.py and the final trained model saved as a
    pickle file in classifier.pkl.
    
   - Inside the app folder

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database. 
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves. 
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
