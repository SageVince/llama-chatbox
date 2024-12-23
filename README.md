# llama-chatbox
a python app that has your local llm train on csv pdf and images for data and download them into datasets



The app is built using Flask, a Python web framework. The app has three main functions:

Chatbot UI: The app has a simple chatbot UI that allows users to interact with the model by typing a message.
File conversion: The app has three file conversion functions that convert CSV, image, and PDF files into a format that the model understands. The conversion functions are:
convert_csv_to_dataset: reads a CSV file and converts it into a list of dictionaries, where each dictionary represents a single data point.
convert_image_to_dataset: reads an image file and converts it into a list of dictionaries, where each dictionary represents a single data point.
convert_pdf_to_dataset: reads a PDF file and converts it into a list of dictionaries, where each dictionary represents a single data point.
Training: The app has a training function that takes a dataset as input and trains the model on it. The training function uses the LLaMAForSequenceClassification model and the LLaMAModel processor.
Download: The app has a download function that allows users to download the trained dataset in JSON format.
Error codes:

The app returns the following error codes:

Error: unsupported file format: returned when the app receives a file that is not supported (e.g. not CSV, image, or PDF).
Error: training failed: returned when the training function fails for some reason (e.g. invalid dataset).
