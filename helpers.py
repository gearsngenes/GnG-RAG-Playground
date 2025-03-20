from openai import OpenAI
from dotenv import load_dotenv
import os
import base64
from PyPDF2 import PdfReader
from docx import Document
from pptx import Presentation

load_dotenv()

#===OpenAI===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)
#===Local Files Root directory===
UPLOAD_FOLDER = os.getenv("UPLOAD_ROOT")

#===File-Processing Helper Methods===
DOC_EXTENSIONS = ['.pdf', '.docx', '.pptx', '.txt']
IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png']
def extract_text(file_path, chunk_size=500):
    ext = file_path.lower()
    full_text = ""
    if ext.endswith(".pdf"):
        reader = PdfReader(file_path)
        full_text = "\n".join([page.extract_text() or "" for page in reader.pages])
    elif ext.endswith(".docx"):
        doc = Document(file_path)
        full_text = "\n".join([para.text.strip() for para in doc.paragraphs if para.text.strip()])
    elif ext.endswith(".pptx"):
        prs = Presentation(file_path)
        full_text = "\n".join([" ".join([shape.text for shape in slide.shapes if hasattr(shape, "text")]) for slide in prs.slides])
    elif ext.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            full_text = f.read()
    else:
        raise ValueError("Unsupported file format.")
    return chunk_text(full_text, chunk_size)

def extract_images_from_pdf(file_path, images_dir):
    reader = PdfReader(file_path)
    images = []
    for i, page in enumerate(reader.pages):
        for img_index, image in enumerate(page.images):
            img_data = image.data
            img_filename = f"page-{i}-{img_index}.jpg"
            img_path = os.path.join(images_dir, img_filename)
            with open(img_path, "wb") as f:
                f.write(img_data)
            images.append(img_path)
    return images

def extract_images_from_docx(file_path, images_dir):
    doc = Document(file_path)
    images = []
    for img_index, rel in enumerate(doc.part.rels):
        if "image" in doc.part.rels[rel].target_ref:
            img_data = doc.part.rels[rel].target_part.blob
            img_filename = f"image-{img_index}.jpg"
            img_path = os.path.join(images_dir, img_filename)
            with open(img_path, "wb") as f:
                f.write(img_data)
            images.append(img_path)
    return images

def extract_images_from_pptx(file_path, images_dir):
    prs = Presentation(file_path)
    images = []
    for slide_index, slide in enumerate(prs.slides):
        for img_index, shape in enumerate(slide.shapes):
            if hasattr(shape, "image"):
                img_data = shape.image.blob
                img_filename = f"slide-{slide_index}-{img_index}.jpg"
                img_path = os.path.join(images_dir, img_filename)
                with open(img_path, "wb") as f:
                    f.write(img_data)
                images.append(img_path)
    return images

#===RAG Helper Methods===
def get_embedding(text):
    """Generates an embedding for the given text."""
    response = client.embeddings.create(input=text, model="text-embedding-ada-002")
    return response.data[0].embedding

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def generate_gpt4_description(image_path):
    image_data = encode_image(image_path)
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "Describe the image in great detail, including objects, people, actions, and background elements."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in full detail."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                ],
            },
        ],
    )
    return response.choices[0].message.content

def chunk_text(text, chunk_size=500, overlap=250):
    tokens = text.split()
    if chunk_size <= overlap:
        overlap = chunk_size // 2
    step_size = max(1, chunk_size - overlap)
    chunks = [" ".join(tokens[i:i+chunk_size]) for i in range(0, len(tokens), step_size)]
    return chunks