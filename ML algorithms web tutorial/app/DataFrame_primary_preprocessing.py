import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None  # default='warn'

# https://www.kaggle.com/datasets/yasserh/song-popularity-dataset/code
# https://habr.com/ru/post/460313/
# https://ahsieh53632.github.io/music-attributes-and-popularity/

"""первоначальная подготовка датасета:
- загрузка
- переименование колонок и drop избыточных
- рандомное создание nan
"""

df_load = pd.read_csv('https://github.com/a18091986/MachineLearning/blob/main/Datasets/song_data.csv?raw=true')
df_origin = df_load[:3000]


for i in range(10000):
    col = np.random.choice(['song_duration_ms', 'acousticness',
                                    'danceability', 'energy',
                                    'instrumentalness', 'liveness',
                                    'loudness', 'tempo'])
    num = np.random.randint(0, df_origin.shape[0]-1)
    df_origin.loc[num, col] = np.nan

columns_names = ['Длительность, мс', 'Акустичность', 'Танцевальность', 'Активность',
                 'Инструментальность', 'Живое исполнение', 'Громкость', 'Темп']

df = df_origin.drop(
    columns=['song_name', 'song_popularity', 'key', 'audio_mode', 'speechiness', 'time_signature', 'audio_valence'])

df.rename(columns=dict(zip(df.columns.to_list(), columns_names)), inplace=True)

df = df.round(5)

