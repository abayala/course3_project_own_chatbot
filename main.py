import pandas as pd
import openai
from openai.embeddings_utils import get_embedding, distances_from_embeddings
import tiktoken

openai.api_base = "https://openai.vocareum.com/v1"
openai.api_key = ""


def prepare_dataset(dataset_filepath: str = "2023_fashion_trends.csv"):
    fashion_dataframe = pd.read_csv(dataset_filepath)
    out_dataframe = fashion_dataframe[["Trends"]].rename(columns={"Trends": "text"})
    out_dataframe = add_embeddings(out_dataframe)
    return out_dataframe


def add_embeddings(input_df, embeddings_model_name="text-embedding-ada-002"):
    batch_size = 100
    embeddings = []
    for i in range(0, len(input_df), batch_size):
        # Send text data to OpenAI model to get embeddings
        response = openai.Embedding.create(
            input=input_df.iloc[i:i + batch_size]["text"].tolist(),
            engine=embeddings_model_name
        )
        # Add embeddings to list
        embeddings.extend([data["embedding"] for data in response["data"]])

    # Add embeddings list to dataframe
    input_df["embeddings"] = embeddings
    return input_df


def get_rows_sorted_by_relevance(in_question, df, embeddings_model_name="text-embedding-ada-002"):
    """
    Function that takes in a question string and a dataframe containing
    rows of text and associated embeddings, and returns that dataframe
    sorted from least to most relevant for that question
    """

    # Get embeddings for the question text
    question_embeddings = get_embedding(in_question, engine=embeddings_model_name)

    # Make a copy of the dataframe and add a "distances" column containing
    # the cosine distances between each row's embeddings and the
    # embeddings of the question
    df_copy = df.copy()
    df_copy["distances"] = distances_from_embeddings(
        question_embeddings,
        df_copy["embeddings"].values,
        distance_metric="cosine"
    )

    # Sort the copied dataframe by the distances and return it
    # (shorter distance = more relevant so we sort in ascending order)
    df_copy.sort_values("distances", ascending=True, inplace=True)
    return df_copy


def create_prompt(in_question, df, max_token_count):
    """
    Given a question and a dataframe containing rows of text and their
    embeddings, return a text prompt to send to a Completion model
    """
    # Create a tokenizer that is designed to align with our embeddings
    tokenizer = tiktoken.get_encoding("cl100k_base")

    # Count the number of tokens in the prompt template and question
    prompt_template = """
        You are a expert fashion agent. Answer the question based on the context below, and if the question
        can't be answered based on the context, say "I don't know". Add details of how you reached to that answer.
        
        Context: 
        
        {}
        
        ---
        
        Question: {}
        Answer:"""

    current_token_count = len(tokenizer.encode(prompt_template)) + \
                          len(tokenizer.encode(in_question))

    context = []
    for text in get_rows_sorted_by_relevance(in_question, df)["text"].values:

        # Increase the counter based on the number of tokens in this row
        text_token_count = len(tokenizer.encode(text))
        current_token_count += text_token_count

        # Add the row of text to the list if we haven't exceeded the max
        if current_token_count <= max_token_count:
            context.append(text)
        else:
            break

    return prompt_template.format("\n\n###\n\n".join(context), in_question)


def answer_question(
        input_question, df, max_prompt_tokens=1800, max_answer_tokens=150, input_comp_modelname="gpt-3.5-turbo"
):
    """
    Given a question, a dataframe containing rows of text, and a maximum
    number of desired tokens in the prompt and response, return the
    answer to the question according to an OpenAI Completion model

    If the model produces an error, return an empty string
    """

    prompt = create_prompt(input_question, df, max_prompt_tokens)

    try:
        response = openai.Completion.create(
            model=input_comp_modelname,
            prompt=prompt,
            max_tokens=max_answer_tokens
        )
        return response["choices"][0]["text"].strip()
    except Exception as e:
        print(e)
        return ""


if __name__ == '__main__':
    completion_model_name = "gpt-3.5-turbo-instruct"
    fashion_dataset = prepare_dataset()
    questions =["What are the fashion trends in 2023?",
                "Which type of men pants you recommend for 2023"]

    simple_prompt_template = """
    Question: "{}?"
    Answer:
    """
    for question in questions:
        current_simple_prompt = simple_prompt_template.format(question)
        not_custom_answer = openai.Completion.create(
                model=completion_model_name,
                prompt=current_simple_prompt,
                max_tokens=150
            )["choices"][0]["text"].strip()
        custom_answer = answer_question(question, fashion_dataset, input_comp_modelname=completion_model_name)
        print(f"\n\nQuestion: {question}")
        print(f"Answer without context and customization:\n {not_custom_answer}")
        print(f"Answer with context and customization:\n {custom_answer}")
