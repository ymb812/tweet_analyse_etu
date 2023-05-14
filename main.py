import re
import time
from tqdm import tqdm
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords as nltk_stopwords
from pymystem3 import Mystem
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *


# загрузим список стоп-слов
stopwords = set(nltk_stopwords.words('russian'))
np.array(stopwords)


# Оставим в тексте только кириллические символы
def clear_text(text):
    clear_text = re.sub(r'[^А-яЁё]+', ' ', text).lower()
    return " ".join(clear_text.split())


# напишем функцию удаляющую стоп-слова
def clean_stop_words(text, stopwords):
    text = [word for word in text.split() if word not in stopwords]
    return " ".join(text)


def lemmatize(df,
              text_column,
              n_samples: int,
              break_str='br',
              ) -> pd.Series:
    result = []
    m = Mystem()

    for i in tqdm(range((df.shape[0] // n_samples) + 1)):
        start = i * n_samples
        stop = start + n_samples

        sample = break_str.join(df[text_column][start: stop].values)

        lemmas = m.lemmatize(sample)
        lemm_sample = ''.join(lemmas).split(break_str)

        result += lemm_sample

    return pd.Series(result, index=df.index)


# функция для загрузки отчищенных и лемматизированных твитов в csv для дальнейшей быстрой работы
# достаточно вызвать один раз отдельно от основного кода, чтобы сгенерировать файл
def create_data_frame():
    positive = pd.read_csv(
        'positive.csv',
        sep=';',
        header=None
    )

    negative = pd.read_csv(
        'negative.csv',
        sep=';',
        header=None
    )

    positive_text = pd.DataFrame(positive.iloc[:, 3])
    negative_text = pd.DataFrame(negative.iloc[:, 3])

    positive_text['label'] = [1] * positive_text.shape[0]
    negative_text['label'] = [0] * negative_text.shape[0]

    labeled_tweets = pd.concat([positive_text, negative_text])
    labeled_tweets.index = range(labeled_tweets.shape[0])
    labeled_tweets.columns = ['text', 'label']

    # test + clear
    start_clean = time.time()
    labeled_tweets['text_clear'] = labeled_tweets['text'].apply(
        lambda x: clean_stop_words(clear_text(str(x)), stopwords))
    print('Обработка текстов заняла: ' + str(round(time.time() - start_clean, 2)) + ' секунд')

    # lemmatize
    labeled_tweets['lemm_clean_text'] = lemmatize(
        df=labeled_tweets,
        text_column='text_clear',
        n_samples=1000,
        break_str='br',
    )

    labeled_tweets.to_csv('lemmatize_text.csv', sep=';', encoding='utf-8')

    return labeled_tweets


def create_metric(vectorizer, lemmatize_text, train, test):
    vectorizer.fit(lemmatize_text['text'])
    tf_idf_train_base_1 = vectorizer.transform(train['text'])
    tf_idf_test_base_1 = vectorizer.transform(test['text'])

    model_lr_base_1 = LogisticRegression(solver='lbfgs',
                                         random_state=12345,
                                         max_iter=10000,
                                         n_jobs=-1)

    model_lr_base_1.fit(tf_idf_train_base_1, train['label'])  # обучаем
    predict_lr_base_proba = model_lr_base_1.predict(tf_idf_test_base_1)  # предиктим на основе обучения

    target_names = ['positive', 'negative']
    return classification_report(list(test['label']), predict_lr_base_proba, target_names=target_names)


def main():
    lemmatize_text = pd.read_csv(
        'lemmatize_text.csv',
        sep=';'
    )

    train, test = train_test_split(
        lemmatize_text,
        test_size=0.2,
        random_state=12348,
    )

    for i in range(1, 4):
        if i == 1:
            gramm = 'Униграмы'
        elif i == 2:
            gramm = 'Биграммы'
        elif i == 3:
            gramm = 'Триграммы'

        print(f'\n{gramm}, используя CountVectorizer и TfidfVectorizer соответственно:')
        print(create_metric(TfidfVectorizer(ngram_range=(i, i)), lemmatize_text, train, test))
        print(create_metric(CountVectorizer(ngram_range=(i, i)), lemmatize_text, train, test))


if __name__ == '__main__':
    main()
