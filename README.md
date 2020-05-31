# Sentiment-Analysis

You will find here : 
* A Pytorch BiLSTM Model trained on IMDB Dataset (50k movie reviews) for 3 Epochs
* Scripts for training the model from scratch 
* Finally, you can deploy the model via API with Flask 

**Have Fun !** 

## Model training 
The architecture is basic, BiLSTM with LayerNormalization. 

The model has been trained for 3 epochs with Adam optimizer and Cyclical LR Schedueler

### run training 

```shell
python main.py --data_path='../Data/IMDB Dataset.csv' 
               --texts_col='review' --labels_col='sentiment' 
                --n_classes=2 --batch_size=16 --batch_size_eval=64
                --n_epochs=3 --cuda=1
```

## APP 

In order to use the API code, you will have to deploy the flask server and send a request via terminal. 

The APP code is available inside the app repo 

```python 
# Under the app directory 
python app.py
``` 
### API 

```shell
curl -d '{"text": "Very good movie but too long"}' -X POST http://127.0.0.1:5000/predict
```


## TO DO 

1. Review the code base and make it more generic 
2. Provide a docker image of the project
3. Add extra features : multilangual models, csv as input ... 
4. Unit Tests 

