import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from datetime import datetime


# Load the existing Excel file
input_path = 'qa_inaxa_groundtruth_version08-05-2024_1.xlsx'
df_sheet1 = pd.read_excel(input_path, sheet_name='ERPI')
df_sheet2 = pd.read_excel(input_path, sheet_name='ERPI_v2')

date = datetime.today()
# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')


# Preprocess the text data
def preprocess(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize
    words = word_tokenize(text)
    # Remove stop words
    stop_words = set(stopwords.words('french'))
    words = [word for word in words if word not in stop_words]
    return words

df_sheet1['Process_Réponse'] = df_sheet1['Réponse'].apply(preprocess)
df_sheet1['Process_Réponse smartInAxa'] = df_sheet1['Réponse smartInAxa'].apply(preprocess)
df_sheet2['Process_Réponse'] = df_sheet2['Réponse'].apply(preprocess)
df_sheet2['Process_Réponse smartInAxa'] = df_sheet2['Réponse smartInAxa'].apply(preprocess)


# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to get BERT embeddings
def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True) #512 tokens
    outputs = model(**inputs)
    # We take the mean of the token embeddings from the last layer
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings # averaged embedding vector.

# Function to calculate cosine similarity percentage using BERT
def cosine_similarity_percentage_bert(embedding1, embedding2):
    cosine_sim = cosine_similarity(embedding1.detach().numpy(), embedding2.detach().numpy())
    return f'{cosine_sim[0][0] * 100:.2f}'

# Calculate BERT similarities for sheet1
similarity_percentages_sheet1 = []
for _, row in df_sheet1.iterrows():
    embedding1 = get_bert_embedding(row["Process_Réponse"])
    embedding2 = get_bert_embedding(row["Process_Réponse smartInAxa"])
    similarity = cosine_similarity_percentage_bert(embedding1, embedding2)
    similarity_percentages_sheet1.append(similarity)

df_sheet1['bert_similarity_percentage'] = similarity_percentages_sheet1
df_bert_sheet1 = df_sheet1[['Question','Réponse','Lien_source', 'Réponse smartInAxa', 'Lien smartInAxa', 'Perimetre', 'bert_similarity_percentage']]
df_bert_sheet1.columns = ['Question','Réponse','Lien_source', 'Réponse smartInAxa', 'Lien smartInAxa', 'Perimetre', 'BERT Similarity Percentage']

# Calculate BERT similarities for sheet2
similarity_percentages_sheet2 = []
for _, row in df_sheet2.iterrows():
    embedding1 = get_bert_embedding(row["Process_Réponse"])
    embedding2 = get_bert_embedding(row["Process_Réponse smartInAxa"])
    similarity = cosine_similarity_percentage_bert(embedding1, embedding2)
    similarity_percentages_sheet2.append(similarity)

df_sheet2['bert_similarity_percentage'] = similarity_percentages_sheet2
df_bert_sheet2 = df_sheet2[['Question','Réponse','Lien_source', 'Réponse smartInAxa', 'Lien smartInAxa', 'Perimetre', 'bert_similarity_percentage']]
df_bert_sheet2.columns = ['Question','Réponse','Lien_source', 'Réponse smartInAxa', 'Lien smartInAxa', 'Perimetre', 'BERT Similarity Percentage']

# Save the results to the same Excel file with different sheets for the results
output_path = f'qa_inaxa_groundtruth_version08-05-2024_1.xlsx'
with pd.ExcelWriter(output_path, engine='openpyxl', mode='a') as writer:
  df_bert_sheet1.to_excel(writer, sheet_name='PR-P_ERPI11', index=False)
  df_bert_sheet2.to_excel(writer, sheet_name='PR-P_ERPI_v2111', index=False)

print(f"Results saved to {output_path}")