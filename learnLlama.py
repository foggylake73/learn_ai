import os
from llama_api_client import LlamaAPIClient
from dotenv import load_dotenv

load_dotenv() # loads .env file



# or Llama-4-Maverick-17B-128E-Instruct-FP8
def llama4(prompt, image_urls = [], model = "Llama-4-Scout-17B-16E-Instruct-FP8"):
  image_urls_content = []
  for url in image_urls:
    image_urls_content.append(
        {"type": "image_url", "image_url": {"url": url}})

  content = [{"type": "text", "text": prompt}]
  content.extend(image_urls_content)

  client = LlamaAPIClient(api_key = os.environ.get("LLAMA_API_KEY"))

  response = client.chat.completions.create(
    model = model,
    messages = [{
        "role": "user",
        "content": content
    }],
    temperature = 0
  )
  return response.completion_message.content.text


# question = """how many languages do you understand? answer in all the languages you can speak."""
# print("\n" + llama4(question) + "\n")



# -----------------------------

# image stuff
import requests # send http requests using python, used to get images from urls
from PIL import Image # python imaging library
from io import BytesIO
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg

def display_image_from_url(image_url):
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def display_image_from_file(img):
    plt.imshow(mpimg.imread(img))
    plt.axis('off')
    plt.show()


img_url = "https://raw.githubusercontent.com/meta-llama/llama-models/refs/heads/main/Llama_Repo.jpeg"
img_url2 = "https://raw.githubusercontent.com/meta-llama/PurpleLlama/refs/heads/main/logo.png"
# display_image_from_url(img_url)
# display_image_from_url(img_url2)
# print("\n" + llama4("Compare these two images.", [img_url, img_url2]) + "\n")



# -----------------------------

# image grounding
import base64


def encode_image_to_base64(image_path): # encode local image file to base64 string
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


# image_file = "data/testImage.png"
# display_image_from_file(image_file)
# prompt = """Which tools in the image can be used for measuring length? Provide bounding boxes for every recognized item."""
# base64_tools = encode_image_to_base64(image_file)
# print("\n" + llama4(prompt, [f"data:image/jpeg;base64, {base64_tools}"]) + "\n")



# -----------------------------

# analyze table from pdf
from pathlib import Path
from pypdf import PdfReader # for pdf stuff

def pdf_to_text(file : str):
  text = ''
  with Path(file).open("rb") as f:
    reader = PdfReader(f)
    text = "\n\n".join([page.extract_text() for page in reader.pages])

  return text


# pdf to text
# meta_q4_2024_txt = pdf_to_text("data/testPdf.pdf")
# start = meta_q4_2024_txt.find("Fourth Quarter and Full Year 2024 Financial Highlights")
# print("\n" + meta_q4_2024_txt[start:start+1000] + "\n")
# prompt = f"""How much is 2024 operating margin based on Meta's financial quarter report below: {meta_q4_2024_txt}"""
# print("\n" + llama4(prompt) + "\n")

# print("\n-----\n")

# image of pdf
# display_image_from_file("data/testPdfImage.png")
# base64_meta_q4_2024_image = encode_image_to_base64("data/testPdfImage.png")
# prompt = """How much is 2024 operating margin based on Meta's financial report?"""
# print("\n" + llama4(prompt, [f"data:image/jpeg;base64,{base64_meta_q4_2024_image}"]) + "\n")



# -----------------------------

# analyze image / code from image
# image_screenshot = "data/testImageScreenshot.png"
# display_image_from_file(image_screenshot)
# base64_image = encode_image_to_base64(image_screenshot)
# prompt = """What is the current tempertature? If I want to change the temperature on the image, where should I click? 
# Return the bounding box for the location."""
# output = llama4(prompt, [f"data:image/jpeg;base64,{base64_image}"])
# print("\n" + output + "\n")

# print("\n-----\n")

# use maverick for better code understanding, maverick for text/code? scout for images?
# prompt = """"Write a python script that uses Gradio to implement the chatbot UI in the image."""
# output = llama4(prompt,[f"data:image/jpeg;base64,{base64_image}"], model="Llama-4-Maverick-17B-128E-Instruct-FP8")
# print("\n" + output + "\n")



# -----------------------------

# solving math problems from image
image_math = "data/testImageMath.png"
base64_math = encode_image_to_base64(image_math)
prompt = "Answer the question in the image."
print("\n" +llama4(prompt, [f"data:image/png;base64,{base64_math}"]) + "\n")



# ------------------------------


# llama message format
# messages = [
#     {
#         "role": "user",
#         "content": [
#             {
#                 "type": "text",
#                 "text": "Describe the image below.",
#             },
#             {
#                 "type": "image", 
#                 "url": url
#             },
#         ],
#     },
# ]
