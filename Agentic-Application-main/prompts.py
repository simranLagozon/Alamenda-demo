from examples import get_example_selector
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder,FewShotChatMessagePromptTemplate,PromptTemplate # type: ignore
import os
from dotenv import load_dotenv # type: ignore

load_dotenv()  # Load environment variables from .env file

# Get the static part of the prompt
static_prompt = os.getenv("FINAL_PROMPT")
example_prompt = ChatPromptTemplate.from_messages(
    [
        # ("human", "{input}\nSQLQuery:"),
         ("human", "{input}"),
        ("ai", "{query}"),
    ]
)
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    example_selector=get_example_selector(),
    input_variables=["input","top_k"],
)

final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", static_prompt.format(table_info="{table_info}")),
        few_shot_prompt,
        MessagesPlaceholder(variable_name="messages"),
        ("human", "{input}"),
    ]
)
# print ("This is final prompt ...." , final_prompt , " .. and this is few shot prompt.." , few_shot_prompt)
answer_prompt = PromptTemplate.from_template(
    """Given the user question, corresponding SQL query, and SQL result, answer the user question.
     Start with SQL query as the first line of your answer, then follow it with your answer in a new line.
     Respond without modifying any of the nouns or numerical values.
     DO NOT modify any of the nouns or numerical values received in the SQL result.
     
     After the answer, generate three relevant follow-up questions that the user might ask next.

Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer: {answer}

Here are three relevant follow-up questions:
1. {follow_up_1}
2. {follow_up_2}
3. {follow_up_3}
"""
)
