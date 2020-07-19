# DisasterResponseProject
The project aims at developing a message classifier embedded into a web application which allows emergency workers to detect appropriately disasters and put in place the type of response needed.

An ETL and ML pipelines are developed to train a classifier on a series of labelled messages.
Once the model has been trained, the input message given via the web application passes through the pre-processing NLP pipeline and is served to the model that classifies it into the different disaster response classes.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database

        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves the model

        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.

    `python run.py`

3. Go to http://0.0.0.0:3001/

### Description:

1. [jupyter notebooks](jupyter-notebooks) contains the notebooks where exploration and first analysis has been done on the message data used to train the model as well as prototyping the ETL and ML pipelines.

2. [data](data) contains the ETL pipeline and database storing the messages.

3. [models](models) contains the ML pipeline used to train the NLP model.

4. [app](app) contains the web application structure and leverages the files in data and models.
