import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df=pd.read_csv("charts.csv")

frequency_Artists = df['Artists'].value_counts()
print(frequency_Artists)


frequency_Song = df['Name'].value_counts()
print(frequency_Song)


# Genre=pd.DataFrame(Temp1)
# GenreList = ','.join(Genre['Genre']).split(',')
# frequency_Genre = pd.Series(GenreList).value_counts()
# print(frequency_Genre[0:6])

# import pandas as pd

# df = pd.read_csv("charts.csv")

# frequency_song = df['Name'].value_counts()
# for song_name, count in frequency_song.items():
#     artist = df.loc[df['Name'] == song_name, 'Artists'].iloc[0]
#     print(f"Song: {song_name}, Artists: {artist}, Frequency: {count}")