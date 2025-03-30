# Project: Build Your Own Custom Chatbot
In this project you will change the dataset to build your own custom chatbot.
## Usage Instructions

* Install in your virtual environment the packages in [requirements.txt](requirements.txt)
* Modify questions list in the main section of [main.py](main.py)
* Set your OPENAPI key
* Run main.py
## Project Description

### Dataset Selection
I want to create a chatbot that gives recommendations or answers questions about fashion topics in 2023.
For this reason I selected the dataset [2023_fashion_trends.csv](2023_fashion_trends.csv) included in the course material, which contains reviews and comments about fashion topics.

### Helper Function Descriptions
Here are some explanations of the functions implemented in [main.py](main.py):
* The function ```prepare_dataset()``` reads the dataset csv file, extracts only the "Trends" column renaming it to "text", computes each row text embeddings and adds them to a column "embeddings" for later use in context search.
* The function ```get_rows_sorted_by_relevance()``` receives a questions and uses the embeddings to perform to semantic similarity search in the database
* The function ```create_prompt()``` creates a personalized prompt by adding the context extracted with ```get_rows_sorted_by_relevance()``` and adds the most rows possible that fits in the context window. The template prompt looks as follows: 
  *     ```You are an expert fashion agent. Answer the question based on the context below, and if the question
         can't be answered based on the context, say "I don't know". Add details of how you reached to that answer.
        
         Context: 
        
         {}
        
         ---
        
         Question: {}
         Answer:```
* The function ```answer_question()```receives an input question and a pandas dataframe, builds a prompt extracting the relevant context from the dataframe and uses OpenAI api to get a response for the prompt.

### Project Outline
The workflow of  [main.py](main.py) is as follows:
* The dataset containing fashion information of 2023 and embeddings is extracted
* A simple prompt template is defined to ask the llm questions without prompt customization
* A defined the next questions relevant to the fashion chatbot:
  * Which are the fashion trends in 2023?
  * Which type of men pants you recommend for 2023?
* The previous questions are input to completion OpenAI api through openai.Completion.create, one time without context and prompt customization, and one time with prompt customization and context

### Result Analysis
I got the next outputs:
```
Question: Which are the fashion trends in 2023?
Answer without context and customization:
 Unfortunately, as someone who lacks the ability to predict the future, I am unable to accurately answer this question. Fashion trends are constantly changing and evolving, so it is difficult to predict what will be popular in 2023. However, some current trends that may continue in the future are sustainability, comfort-focused fashion, and diversity and inclusivity in the industry.
Answer with context and customization:
 There are several fashion trends mentioned in the context, including sheer clothing, red, shine for daytime, cobalt blue, cargo pants, elevated basics, maxi skirts, denim reimagined, slouchy-fit trousers, perfectly cut trousers, green, "indie sleaze," pinstripe tailoring, and simplicity/everyday dressing.


Question: Which type of men pants you recommend for 2023?
Answer without context and customization:
 It's difficult to say what type of men's pants will be popular in 2023, as fashion trends are always changing. However, some classic styles that are likely to remain popular for years to come include straight leg jeans, chinos, and tailored dress pants. Comfortable and versatile options such as joggers and cargo pants may also continue to be a popular choice. Ultimately, it's important to choose pants that fit well and make you feel confident and stylish.
Answer with context and customization:
 Slouchy-fit trousers or cargo pants. Based on the context provided, the women's fashion experts mentioned multiple times that relaxed silhouettes and utilitarian wear are on trend for 2023. Slouchy-fit trousers and cargo pants fit into these categories and were mentioned by multiple sources, indicating their popularity and relevance in the fashion industry for the upcoming year. Additionally, cargo pants were specifically mentioned as a top trend for 2023 in one of the contexts.
```
As it can be observed the replies obtained with the prompt customization that provides context are more accurate and rich. Without context the LLM mentions it does not know specifics of that year. 