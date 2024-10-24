# import google.generativeai as genai        
# genai.configure(api_key='AIzaSyC-DUwcu0XTsd-jEafCYEmlqscBsjV8DSI',transport='rest')    
# model = genai.GenerativeModel('gemini-1.5-flash')    
# response = model.generate_content("write a poem about the moon")    
# print(response.text)export GENAI_API_KEY=YOUR_NEW_API_KEY

import os
import pandas as pd
from PIL import Image
import google.generativeai as genai
from tqdm import tqdm
import time

def main():
    # Retrieve API key from environment variable
    api_key = os.getenv('GENAI_API_KEY')

    if not api_key:
        raise ValueError("API key not found. Please set the 'GENAI_API_KEY' environment variable.")

    # Configure Google Generative AI
    genai.configure(api_key=api_key, transport='rest')
    model = genai.GenerativeModel('gemini-1.5-flash')

    # Define the directory containing images
    image_dir = r'C:\AA\data\ImageDatabase'  # 使用原始字符串来避免转义字符的问题

    # Initialize list to store results
    results = []

    # Define the prompt
    prompt = (
        "You are an expert in image quality assessment. "
        "Please help me analyze the quality of the following image comprehensively. "
        "Identify the key aspects that determine the image's quality and provide a detailed assessment for each aspect, including clear explanations to ensure the analysis is interpretable."
    )

    # Total number of images
    total_images = 590

    # Iterate over each image with progress bar
    for i in tqdm(range(1, total_images + 1), desc="Processing Images"):
        image_name = f'DatabaseImage{str(i).zfill(4)}.jpg'  # Adjust extension if necessary
        image_path = os.path.join(image_dir, image_name)
        print(f"Checking image: {image_path}")  # 打印出正在检查的图像路径

        if os.path.exists(image_path):
            try:
                # Open and prepare the image
                with Image.open(image_path) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')

                    # Initialize retry mechanism
                    max_retries = 3
                    for attempt in range(1, max_retries + 1):
                        try:
                            # Generate content using the model
                            response = model.generate_content([prompt, img])
                            assessment = response.text
                            break  # Exit retry loop on success
                        except Exception as e:
                            assessment = f"Error processing image on attempt {attempt}: {e}"
                            if attempt < max_retries:
                                time.sleep(2)  # Wait before retrying
                            else:
                                print(f"Failed to process {image_name} after {max_retries} attempts.")
            except Exception as e:
                assessment = f"Error opening image: {e}"
        else:
            assessment = 'Image not found.'
            print(f"Warning: {image_name} does not exist.")

        # Append the result to the list
        results.append({'Image': image_name, 'Assessment': assessment})

    # Convert results to a DataFrame
    df = pd.DataFrame(results)

    # Save the DataFrame to an Excel file
    output_excel = 'image_assessments.xlsx'
    try:
        df.to_excel(output_excel, index=False)
        print(f"\nAssessment completed. Results saved to '{output_excel}'.")
    except Exception as e:
        print(f"\nFailed to save results to Excel: {e}")

if __name__ == "__main__":
    main()
