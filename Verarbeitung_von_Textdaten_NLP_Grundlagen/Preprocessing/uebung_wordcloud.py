from wordcloud import WordCloud
import requests

# ToDo: laden und Preprocessing des Textes
# ...





# Erzeugen einer Wordcloud, ersetzen Sie 'wort1, wort2' durch Ihre eigenen Wörter
wordcloud = WordCloud(background_color = 'white',
                       width = 512,
                       height = 384).generate('wort1, wort2')
# Speichern des Wordcloud-Bildes
wordcloud.to_file('wordcloud.png')
