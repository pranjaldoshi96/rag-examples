import streamlit as st

from chains import (
    generate_response,
)  # Import the function to generate responses from the AI model
from ingest import (  # Import functions for managing document ingestion
    get_vectorstore,
    ingest_document,
)

# Initialize session state variables if they do not already exist
# These are used to store URLs and chat history between user interactions
if "urls" not in st.session_state:
    st.session_state.urls = []  # List to keep track of ingested URLs
if "messages" not in st.session_state:
    st.session_state.messages = []  # List to keep track of chat messages

# Sidebar section for URL ingestion
st.sidebar.header("URL Ingestion")  # Set the header for the sidebar section
url_input = st.sidebar.text_input(
    "Enter a website URL:"
)  # Text input for entering URLs

# Button to trigger URL ingestion
if st.sidebar.button("Ingest URL"):
    if (
        url_input and url_input not in st.session_state.urls
    ):  # Check if URL is valid and not already ingested
        ingest_document(url_input)  # Ingest the document associated with the URL
        st.session_state.urls.append(
            url_input
        )  # Add the URL to the list of ingested URLs
        st.sidebar.success(
            f"URL '{url_input}' ingested successfully!"
        )  # Show a success message

# Display the list of ingested URLs in the sidebar
st.sidebar.subheader("Ingested URLs")  # Subheader for the ingested URLs section
st.sidebar.write(
    st.session_state.urls
)  # Write the list of ingested URLs to the sidebar

# Main content area for chatting with the AI
st.subheader("Chat with Olympic Data!")  # Set the subheader for the chat section

# Display the chat history
for message in st.session_state.messages:
    # Display each message in the chat history with appropriate role (user or assistant)
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input box for the user to type their message
user_input = st.chat_input("Ask Question here?")  # Text input for user questions

# Check if the user has entered a message
if user_input:
    # Append the user's message to the chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    # Display the user's message in the chat
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate a response from the AI model based on the user's input
    full_response = generate_response(user_input)
    # Append the assistant's response to the chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    # Display the assistant's response in the chat
    with st.chat_message("assistant"):
        st.markdown(full_response)
