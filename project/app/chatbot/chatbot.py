import dash
from dash import dcc, html, Input, Output, callback, State
import pandas as pd
import plotly.express as px
from dash.dependencies import Input, Output
import openai
import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import random
from project.config import open_ai_key

# Load environment variables
openai.api_key = open_ai_key

# Menggunakan OpenAI untuk chatbot (menggunakan GPT-3 atau GPT-4)
# 1. Load and preprocess data
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

chatbot_table_file = os.path.join(project_root, "data/query_result/chatbot_table_data.pkl")
data = pd.read_pickle(chatbot_table_file)

# Select relevant columns for embedding
data['combined_text'] = data['Category'] + ' ' + data['Brand'] + ' ' + data['Seasonality']

# 2. Generate embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')  # Load a pre-trained embedding model
embeddings = model.encode(data['combined_text'].tolist())

# 3. Initialize FAISS index
embedding_dim = embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(embedding_dim)
#faiss_index.add(np.array(embeddings))
faiss_index.add(embeddings.astype('float32'))

# 4. Retrieval function
def retrieve(user_input, top_k=5):
    """Retrieve the top_k most relevant entries based on the query."""
    query_embedding = model.encode([user_input])
    #distances, indices = faiss_index.search(np.array(query_embedding), top_k)
    distances, indices = faiss_index.search(query_embedding.astype('float32'), top_k)
    results = data.iloc[indices[0]]
    return results

def chatbox():
    return html.Div([
        # Area chat dengan scroll
        html.Div([
            html.H2("Chatbot"),
        ], style={
            'backgroundColor': '#007BFF',
            'border': '1px solid #ddd', 
            'borderRadius': '10px',
            'color': 'white',
            'textAlign': 'center'}
            ),
        #dcc.Store(id="chat-history", storage_type="session"),
        dcc.Store(id="chat-history", data=[]),
        dcc.Loading(
            id='loading-chat',
            type='circle',
            children=[
                html.Div(id='chat-box', style={
                    'height': '250px', 'overflowY': 'scroll', 'border': '1px solid #ddd', 'padding': '10px', 'borderRadius': '10px',
                    'backgroundColor': '#f9f9f9', 'marginBottom': '20px'
                })
            ]
        ),

        # Input text untuk pesan
        dcc.Input(id='user-input', 
                  type='text', 
                  placeholder=random.choice([
                      'what product that i should sell to avoid overstock?',
                      'what product that i should restock based on current season?',
                      'produk apa yang saya jual untuk menghindari overstock?',
                      'produk apa yang harus saya restock berdasarkan musim yang sedang berjalan?'
                  ]), 
                  style={
            'width' : '95%', 'padding': '10px', 'borderRadius': '5px', 'border': '1px solid #ddd'
        }),

        # Tombol kirim pesan
        html.Button('Send', id='send-button', n_clicks=0, style={
            'padding': '10px 20px', 'backgroundColor': '#007bff', 'border': 'none', 'color': 'white', 'borderRadius': '5px'
        }),
    ], style={'width': '400px', 'margin': '0 auto', 'padding': '10px', 'border': '1px solid #ddd', 'borderRadius': '10px', 'fontFamily': 'Helvetica', 'marginTop': '10px',
              "boxShadow": "0px 4px 6px rgba(0,0,0,0.1)", 'backgroundColor': '#f9f9f9'})

# callback untuk chatbot
@callback(
    [Output('chat-box', 'children'), Output('chat-history', 'data'), Output('user-input', 'value'),],
    [Input('send-button', 'n_clicks')],
    [State('user-input', 'value'), State('chat-history', 'data')],
    prevent_initial_call=True
    )
def update_chatbot_output(n_clicks, user_input, chat_history):
    if not user_input:
        return dash.no_update, chat_history

    # Jika tidak ada, inisialisasi sebagai list kosong
    if chat_history is None:
        chat_history = []
    
        # Pesan dari pengguna
    user_message = {
        "role": "user",
        "content": user_input,
        "style":{
            'backgroundColor': '#007bff', 'color': 'white', 'padding': '8px', 'borderRadius': '10px', 'marginBottom': '5px',
            'alignSelf': 'flex-end', 'maxWidth': '80%', 'margin-left': 'auto', 'fontFamily': 'Helvetica', 'width' : 'fit-content', 'margin-right': '5px'
        }
    }

    df = retrieve(user_input, top_k=5)
    conversation_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])

    # Generate insights
    insights = []

    # Basic DataFrame Information
    insights.append(
        f"The DataFrame contains {len(df)} rows and {len(df.columns)} columns."
    )
    insights.append("Here are the first 5 rows of the DataFrame:\n")
    insights.append(df.head().to_string(index=False))

    # Summary Statistics
    insights.append("\nSummary Statistics:")
    insights.append(df.describe().to_string())

    # Column Information
    insights.append("\nColumn Information:")
    for col in df.columns:
        insights.append(f"- Column '{col}' has {df[col].nunique()} unique values.")

    insights_text = "\n".join(insights)

    prompt = (
        "Objective: Provide insights for inventory optimization."
        "Instructions: You will be given a dataset containing inventory information from a company, including product data, stock levels, demand rates, lead times, and other relevant data. Your task is to provide insights that can help optimize the company's inventory management. The insights should be based on the available data and aim to improve efficiency, reduce waste, and ensure optimal stock levels."
        "Possible areas to explore:"
        "1.Identify Overstocked and Understocked Products: Find products that are overstocked or understocked and recommend adjustments in ordering or stock management."
        "2.Demand Forecasting: Use historical data to predict future demand and recommend changes in procurement strategies."
        "3.Inventory Turnover Analysis: Identify products with slow turnover rates and provide suggestions to increase sales or reduce order quantities."
        "4.Lead Time Optimization: Analyze the lead time for specific products and recommend when to reorder to ensure stock availability without overstocking."
        "5.Storage Space Management: Provide suggestions on how to optimize storage space usage based on inventory data."
        "6.Cost Analysis: Identify high-cost products that are not moving or selling well, and suggest ways to reduce storage costs or find alternative suppliers."
        "Provide data-driven insights to improve inventory management, making it more efficient and cost-effective. Ensure that the recommendations align with the goal of optimization and cost savings for the company."
        "The user should be invited to ask another question at the end of the response. You can also respond based on the conversation history. You can also answer in Indonesian, depending on the user's question"
        #"You can also answer in Indonesian, depending on the user's question. The user should be invited to ask another question at the end of the response"
        #"In addition to the insights, generate visualizations that help better understand and communicate the data."
    )
    messages = [{"role": "system", "content": f"{prompt}\n\nContext:\n\n{insights_text}\n\nconversation history: {conversation_history}"}]
    messages.append({"role": "user", "content": user_input})  # Tambahkan pertanyaan baru

    prompt = f"{prompt}\n\nContext:\n\n{insights_text}"
    prompt = f"""{prompt}\n\nUser's Question: {user_input}"""
    prompt = f"{prompt}\n\nconversation history: {conversation_history}"

    response = openai.ChatCompletion.create(
    model="gpt-4o", messages=messages
    #model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}]
    )
    bot_response = response.choices[0].message.content  # Mengembalikan respons dari chatbot
    bot_message = {
        "role": "bot",
        "content": bot_response,
        "style":{
            'backgroundColor': '#f1f1f1', 'color': 'black', 'padding': '8px', 'borderRadius': '10px', 'marginBottom': '5px',
            'alignSelf': 'flex-start', 'maxWidth': '80%', 'margin-right': 'auto', 'whiteSpace': 'pre-wrap', 'fontFamily': 'Helvetica', 'textAlign': 'left', 'margin-left': '5px'
        }
    }
    # Perbarui chat_history
    chat_history.append(user_message)
    chat_history.append(bot_message)

    chat_elements = [
        html.Div(message["content"], style=message["style"])
        for message in chat_history
    ]

    return chat_elements, chat_history, ""
