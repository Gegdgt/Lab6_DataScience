import pandas as pd
import re
import string
from sklearn.feature_extraction.text import CountVectorizer

# Ruta del archivo
file_path = r'C:\Users\manue\OneDrive\Escritorio\Data_Science\Lab6\GrammarandProductReviews.csv'

# Leer el archivo CSV
reviews_data = pd.read_csv(file_path)

# Limpiar el texto de las reseñas (convertir a minúsculas, eliminar puntuación, etc.)
def clean_text_v2(text):
    text = text.lower()  # Convertir a minúsculas
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Eliminar URLs
    text = re.sub(r'@\w+|#\w+', '', text)  # Eliminar hashtags y menciones
    text = text.translate(str.maketrans('', '', string.punctuation))  # Eliminar puntuación
    return text

# Aplicar la limpieza a la columna 'reviews.text'
reviews_data['reviews.text'] = reviews_data['reviews.text'].astype(str)
reviews_data['cleaned_text'] = reviews_data['reviews.text'].apply(clean_text_v2)

# Clasificar las reseñas como positivas, negativas o neutrales según la puntuación
def classify_sentiment(rating):
    if rating >= 4:
        return 'Positive'
    elif rating <= 2:
        return 'Negative'
    else:
        return 'Neutral'

reviews_data['sentiment'] = reviews_data['reviews.rating'].apply(lambda x: classify_sentiment(float(x)))

# Contar las reseñas positivas y negativas por producto
product_sentiment = reviews_data.groupby(['name', 'brand', 'sentiment']).size().unstack(fill_value=0)

# Obtener los 10 productos mejor valorados
top_positive_products = product_sentiment.sort_values(by='Positive', ascending=False).head(10)

# Obtener los 10 productos peor valorados
top_negative_products = product_sentiment.sort_values(by='Negative', ascending=False).head(10)

# Mostrar los resultados
print("Top 10 productos mejor valorados:")
print(top_positive_products)

print("\nTop 10 productos peor valorados:")
print(top_negative_products)
