### Youtube and Online Article Summarizer

A Streamlit app that uses the LangChain library to generate summaries of user input.
The app has two tabs:

    * Chat: This tab allows the user to chat with the language model and generate summaries of their input.
    * Usage Chart: This tab shows a bar chart of the total OpenAI API usage cost over time.
Attributes:

    llm_chain (summarizer_agent.summary_agent): The LangChain agent that is used to generate summaries.
    messages (list): A list of dictionaries, where each dictionary contains a user message and a bot message.

