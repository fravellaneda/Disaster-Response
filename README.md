# Disaster Response 
Analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages. The main idea of doing that is to create a model to categorize the messages. In order to that I developed a web application to categorize new messages based on the trainning data before described. The web applications looks like the following image

![web application image](https://github.com/fravellaneda/Disaster-Response/blob/main/capture.PNG)

Also you will find some graphics very important for understand the data training.

This work and this heartfelt motivation would not have been possible without the request of the monitors who reviewed my work at Udacity. To him special thanks.

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
|   |-- count_words.csv                         # csv with information of the mosr common words without stopwords
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
