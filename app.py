import streamlit as st
import yfinance as yf
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from openai import OpenAI
from transformers import AutoTokenizer, AutoModel
import torch
import os


# Load environment variables
load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=groq_api_key
)

# Initialize Pinecone client
pc = Pinecone(api_key=pinecone_api_key)

# Ensure the index exists
if "stocks-index" not in pc.list_indexes().names():
    pc.create_index(
        name="stocks-index",
        dimension=768, 
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

pinecone_index = pc.Index("stocks-index")

# Load Hugging Face model for embedding
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def get_embeddings(query):
    inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings.squeeze().numpy()

# Streamlit App Title
st.title("StockView")
st.write("A tool to find relevant stocks using Pinecone and generate AI insights.")

# Sidebar Filters
st.sidebar.header("Filters")
market_cap = st.sidebar.slider("Market Cap (in billions)", 0, 500, (0, 500))
sector = st.sidebar.selectbox("Sector", ["All", "Technology", "Healthcare", "Finance", "Energy", "Others"])

# User Query Input
query = st.text_input("Enter your query:", "What are companies that manufacture consumer hardware?")

# Search Button
if st.button("Search"):
    with st.spinner("Fetching data..."):
        # Generate embeddings for the query
        raw_query_embedding = get_embeddings(query)

        # Build Filters for Pinecone Query
        filters = {}
        if sector != "All":
            filters["sector"] = sector  # Add sector filter if it's not "All"

        # Query Pinecone for Stocks in the Sector
        results = pinecone_index.query(
            vector=raw_query_embedding.tolist(),
            top_k=50, 
            include_metadata=True,
            namespace="stocks",
            filter=filters 
        )

        # Fetch Market Cap for Each Stock
        filtered_stocks = []
        for match in results["matches"]:
            metadata = match["metadata"]
            stock_ticker = metadata.get("Ticker") 
            try:
                # Fetch stock data using yfinance
                stock = yf.Ticker(stock_ticker)
                stock_info = stock.info
                market_cap = stock_info.get("marketCap", None)

                # Check if Market Cap is within the selected range
                if market_cap and market_cap >= market_cap[0] * 1e9 and market_cap <= market_cap[1] * 1e9:
                    filtered_stocks.append({
                        "name": stock_ticker,
                        "sector": metadata.get("sector", "Unknown"),
                        "market_cap": market_cap,
                        "business_summary": metadata.get("Business Summary", "No summary available")
                    })
            except Exception as e:
                st.write(f"Error fetching data for {stock_ticker}: {e}")
                continue

        # Display Filtered Results
        st.subheader("Top Matches")
        if filtered_stocks:
            for stock in filtered_stocks:
                st.write(f"**Company Name:** {stock['name']}")
                st.write(f"**Sector:** {stock['sector']}")
                st.write(f"**Market Cap:** {stock['market_cap'] / 1e9:.2f}B")
                st.write(f"**Business Summary:** {stock['business_summary']}")
                st.write("---")
        else:
            st.write("No stocks found matching the criteria.")

        response = client.chat.completions.create(
            model="llama-3.2-3b-preview",
            messages=[
                {"role": "system", "content": "You are a financial analysis assistant."},
                {"role": "user", "content": f"Generate a financial report for companies related to: {query}"}
            ]
        )

        # Display AI-generated report
        st.subheader("AI-Generated Report")
        st.write(response.choices[0].message.content)