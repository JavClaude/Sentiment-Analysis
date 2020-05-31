# Sentiment-Analysis

### run training 

```shell
python .\main.py --data_path='../Data/IMDB Dataset.csv' --texts_col='review' --labels_col='sentiment' --n_classes=2 --batch_size=16 --batch_size_eval=64 --n_epochs=2 --cuda=1
```
