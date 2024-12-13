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


# Load Hugging Face model for embedding
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")


def get_embeddings(query):
    inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings.squeeze().numpy()

# Ensure the index exists
if "stocks" not in pc.list_indexes().names():
    pc.create_index(
        name="stocks",
        dimension=768, 
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

pinecone_index = pc.Index("stocks")

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
            filters["Sector"] = sector  # Use "Sector" exactly as it appears in your metadata

        # Query Pinecone for Stocks in the Sector
        results = pinecone_index.query(
            vector=raw_query_embedding.tolist(),
            top_k=50,  # Fetch more results to apply additional filtering
            include_metadata=True,
            namespace="stock-descriptions",  # Update based on your namespace
            filter=filters  # Apply sector filter
        )
        st.write(f"Filters applied: {filters}")
        st.write(f"Number of matches from Pinecone: {len(results['matches'])}")
        # st.write(results)  # This will display the raw response for debugging




        # Ensure no duplicates in filtered_stocks
        filtered_stocks = []
        unique_tickers = set()
        slider_min, slider_max = market_cap

        for match in results["matches"]:
            metadata = match["metadata"]
            stock_ticker = metadata.get("Ticker")
            if not stock_ticker or stock_ticker in unique_tickers:
                continue  # Skip duplicates or entries without tickers

            try:
                stock = yf.Ticker(stock_ticker)
                stock_info = stock.info
                market_cap = stock_info.get("marketCap")

                # Apply Market Cap filter
                if market_cap and slider_min * 1e9 <= market_cap <= slider_max * 1e9:
                    filtered_stocks.append({
                        "name": metadata.get("Name", "Unknown"),
                        "sector": metadata.get("Sector", "Unknown"),
                        "market_cap": market_cap,
                        "ticker": stock_ticker,
                        "business_summary": metadata.get("Business Summary", "No summary available"),
                    })
                    unique_tickers.add(stock_ticker)  # Track seen tickers

            except Exception as e:
                continue

        # Display Filtered Results
        st.subheader("Top Matches")
        if filtered_stocks:
            st.write(f"Found {len(filtered_stocks)} matching stocks within the market cap range {slider_min}B - {slider_max}B.")

            # Display a compact summary of filtered stocks
            for stock in filtered_stocks:
                st.write(f"- **{stock['name']}** ({stock['sector']}) | Ticker: {stock['ticker']} | Market Cap: {stock['market_cap'] / 1e9:.2f}B")

            # Add collapsible details for each stock
            for stock in filtered_stocks:
                with st.expander(f"Details for {stock['name']}"):
                    st.write(f"**Sector:** {stock['sector']}")
                    st.write(f"**Ticker:** {stock['ticker']}")
                    st.write(f"**Market Cap:** {stock['market_cap'] / 1e9:.2f}B")
                    st.write(f"**Business Summary:** {stock['business_summary']}")
        
            # Pass Filtered Stocks to OpenAI for Enhanced AI Report
            pinecone_stock_list = ", ".join([f"{stock['name']} (Ticker: {stock['ticker']}, Sector: {stock['sector']}, Market Cap: {stock['market_cap'] / 1e9:.2f}B)" for stock in filtered_stocks])
            ai_query = f"Generate a financial report for the following companies: {pinecone_stock_list}. Include their sectors, financial performance, and any relevant recommendations."

            try:
                # Generate AI Report using OpenAI
                response = client.chat.completions.create(
                    model="llama-3.2-3b-preview",
                    messages=[
                        {"role": "system", "content": "You are a financial analysis assistant."},
                        {"role": "user", "content": ai_query}
                    ]
                )

                # Display AI-generated report
                st.subheader("AI-Generated Report")
                st.write(response.choices[0].message.content)

            except Exception as e:
                st.error(f"Error generating AI report: {e}")
        else:
            st.write(f"No stocks found matching the criteria within the market cap range {slider_min}B - {slider_max}B.")

