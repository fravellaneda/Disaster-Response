# Disaster Response 
Analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages.

### File Description

```
|-- app
|   |-- run.py                                  # Main flask app file
|   |-- adicionalFeatures.py                    # TextLengthExtractor tranformer class for sklearn pipeline
|   |-- templates
|   |   |-- go.html                             # Classify message result page template
|   |   |-- master.html                         # Main page template
|-- data
|   |-- disaster_categories.csv                 # Categories data
|   |-- disaster_messages.csv                   # Messages data
|   |-- DisasterResponse.db                     # Sqlite db for save pre-processed data
|   `-- process_data.py                         # Data pre process script
|-- models
|   |-- classifier.pkl                          # Saved model file
|   |-- train_classifier.py                     # Model training script
|   |-- adicionalFeatures.py                    # TextLengthExtractor tranformer class for sklearn pipeline
|-- README.md                                   
|-- requirments.txt                             # Used by pip to install required python packages
```

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.
    - Install required python packages
        `pip install -r requirements.txt`
    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://127.0.0.1:5000/
