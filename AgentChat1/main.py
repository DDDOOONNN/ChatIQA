#Now I also need to set up a role to determine whether the questioner and the answerer are going off topic. If they are going off topic (e.g., they start talking about photography techniques, image processing techniques, etc.), they need to remind them and ask them to ask or answer again. The code below is modified by me according to the code above. Help me improve the code below. Please think step by step.


import os
import pandas as pd
from tqdm import tqdm
from PIL import Image
import google.generativeai as genai
import time
import base64
import argparse
import logging
import re
import logging
from image_encoder import encode_image
from chat_handler import initialize_chat, send_message_with_retry
from score_extractor import extract_final_score
from args_parser import parse_arguments


logging.basicConfig(
    filename='image_analysis.log',
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)



def main():
    args = parse_arguments()
    image_dir = args.image_dir
    comparison_img_name = args.comparison_img
    output_excel = args.output_excel
    total_images = args.total_images
    num_cycles = args.num_cycles

    # Retrieve API key from environment variables
    api_key = os.getenv('GENAI_API_KEY')

    if not api_key:
        logging.critical("API key not found. Please set the 'GENAI_API_KEY' environment variable.")
        raise ValueError("API key not found. Please set the 'GENAI_API_KEY' environment variable.")

    # Configure Google Generative AI
    try:
        genai.configure(api_key=api_key, transport='rest')
        model = genai.GenerativeModel('gemini-1.5-flash')
    except Exception as e:
        logging.critical(f"Failed to configure Google Generative AI: {e}")
        raise e

    # Define the ComparisonIMG path
    comparison_img_path = os.path.join(image_dir, comparison_img_name)

    if not os.path.exists(comparison_img_path):
        logging.critical(f"Comparison image '{comparison_img_name}' not found in '{image_dir}'.")
        raise FileNotFoundError(f"Comparison image '{comparison_img_name}' not found in '{image_dir}'.")

    # Encode ComparisonIMG
    try:
        comparison_img_data_uri = encode_image(comparison_img_path)
    except Exception as e:
        logging.critical(f"Error encoding comparison image: {e}")
        raise e

    # Initialize the results list
    results = []

    # Define system messages for both roles
    responder_system_message = (
        "You are an Image Analysis Expert. Your task is to assess image quality based on various technical and aesthetic factors. "
        "Always focus your answer on the quality of the image and what really affects the quality of the image."
        " Don't go off topic and talk about the photography techniques and image optimization techniques of the image and other irrelevant topics."
        "Provide comprehensive and objective evaluations."
    )

    asker_system_message = (
        "You are an Inquisitive Analyst. Your task is to ask insightful and valuable questions to the Image Analysis Expert to ensure a thorough and comprehensive evaluation of image quality."
        "Ask useful questions to guide the Image Analysis Expert to think about what really affects image quality. Be careful not to digress."
    )

    judge_system_message = (
        ""
    )

    # Iterate through each image with a progress bar
    for i in tqdm(range(1, total_images + 1), desc="Processing Images"):
        image_num = i
        image_name = f'DatabaseImage{str(i).zfill(4)}.jpg'  # Naming as DatabaseImage0001.jpg, etc.
        image_path = os.path.join(image_dir, image_name)
        print(f"\nProcessing image: {image_path}")  # Print the image path being processed

        if not os.path.exists(image_path):
            logging.warning(f"Image {image_name} does not exist.")
            results.append({
                'Image': image_name,
                'Responder_Assessment_ComparisonIMG': None,
                'Responder_Assessment_CurrentImage': 'Image not found.',
                'Asker_Question_1': None,
                'Responder_Response_1': None,
                'Asker_Question_2': None,
                'Responder_Response_2': None,
                'Asker_Question_3': None,
                'Responder_Response_3': None,
                'Asker_Question_4': None,
                'Responder_Response_4': None,
                'Asker_Question_5': None,
                'Responder_Response_5': None,
                'Final_Score': None
            })
            continue

        try:
            # Open and prepare the current image
            with Image.open(image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                current_img_data_uri = encode_image(image_path)
        except Exception as e:
            logging.error(f"Error opening image {image_name}: {e}")
            results.append({
                'Image': image_name,
                'Responder_Assessment_ComparisonIMG': None,
                'Responder_Assessment_CurrentImage': f"Error opening image: {e}",
                'Asker_Question_1': None,
                'Responder_Response_1': None,
                'Asker_Question_2': None,
                'Responder_Response_2': None,
                'Asker_Question_3': None,
                'Responder_Response_3': None,
                'Asker_Question_4': None,
                'Responder_Response_4': None,
                'Asker_Question_5': None,
                'Responder_Response_5': None,
                'Final_Score': None
            })
            continue

        # Initialize chat sessions for Responder and Asker
        try:
            responder_chat = initialize_chat(model, responder_system_message, role='model')
            asker_chat = initialize_chat(model, asker_system_message, role='model')
        except Exception as e:
            logging.error(f"Error initializing chat sessions for image {image_name}: {e}")
            results.append({
                'Image': image_name,
                'Responder_Assessment_ComparisonIMG': None,
                'Responder_Assessment_CurrentImage': f"Error initializing chat sessions: {e}",
                'Asker_Question_1': None,
                'Responder_Response_1': None,
                'Asker_Question_2': None,
                'Responder_Response_2': None,
                'Asker_Question_3': None,
                'Responder_Response_3': None,
                'Asker_Question_4': None,
                'Responder_Response_4': None,
                'Asker_Question_5': None,
                'Responder_Response_5': None,
                'Final_Score': None
            })
            continue

        # Initialize dictionaries to hold interactions
        interaction = {
            'Image': image_name,
            'Responder_Assessment_ComparisonIMG': None,
            'Responder_Assessment_CurrentImage': None,
            'Asker_Question_1': None,
            'Responder_Response_1': None,
            'Asker_Question_2': None,
            'Responder_Response_2': None,
            'Asker_Question_3': None,
            'Responder_Response_3': None,
            'Asker_Question_4': None,
            'Responder_Response_4': None,
            'Asker_Question_5': None,
            'Responder_Response_5': None,
            'Final_Score': None
        }

        # ------------------------------
        # Cycle 1: Initial Analysis
        # ------------------------------
        # Responder analyzes ComparisonIMG
        comparison_analysis_prompt = (
            f"Please assess the quality of the following image comprehensively. "
            f"This image is named ComparisonIMG, and it's score is 57"
            f"Identify the key aspects that determine the image's quality and provide a detailed assessment for each aspect."
        )
        inline_image_ComparisonIMG = {
            "mime_type": "image/jpeg",
            "data": comparison_img_data_uri.split(",")[1]  # Remove the data URI prefix
        }
        responder_assessment_comparison = send_message_with_retry(
            responder_chat, 
            comparison_analysis_prompt, 
            role='user',
            inline_image=inline_image_ComparisonIMG
        )
        interaction['Responder_Assessment_ComparisonIMG'] = responder_assessment_comparison
        print(f"\nResponder's Assessment for {comparison_img_name}:\n{responder_assessment_comparison}")
        logging.info(f"Responder's Assessment for {comparison_img_name}: {responder_assessment_comparison}")

        # Responder analyzes the Current Image
        current_image_prompt = (
            f"Please assess the quality of the following image comprehensively, and based on the ComparisonIMG's score, give this image a score between 0 and 100. "
            f"This image is named {image_name}. When evaluating, consider both objective technical factors "
            f"(such as sharpness, contrast, color accuracy, etc.) and subjective user experience factors "
            f"(such as aesthetic appeal, emotional impact, etc.). "
            f"Identify the key aspects that determine the image's quality and provide a detailed assessment for each aspect, "
            f"ensuring that acceptable quality images are not unduly penalized."
        )
        # Prepare inline image data
        inline_image = {
            "mime_type": "image/jpeg",
            "data": current_img_data_uri.split(",")[1]  # Remove the data URI prefix
        }
        responder_assessment_current = send_message_with_retry(
            responder_chat, 
            current_image_prompt, 
            role='user', 
            inline_image=inline_image
        )
        interaction['Responder_Assessment_CurrentImage'] = responder_assessment_current
        print(f"\nResponder's Assessment for {image_name}:\n{responder_assessment_current}")
        logging.info(f"Responder's Assessment for {image_name}: {responder_assessment_current}")

        # ------------------------------
        # Cycles 2-6: Question and Answer Rounds
        # ------------------------------
        num_cycles = num_cycles  # Number of interaction cycles
        previous_response = responder_assessment_current  # Initialize with the current image assessment

        for cycle in range(1, num_cycles + 1):
            # Asker generates a question based on the Responder's previous answer
            asker_question_prompt = (
                f"Based on the following response from the Image Analysis Expert, generate a thoughtful and insightful question to further analyze the image quality:\n\n"
                f"Responder's Response:\n{previous_response}"
            )
            asker_question = send_message_with_retry(
                asker_chat, 
                asker_question_prompt, 
                role='user',
                inline_image=inline_image
            )
            interaction[f'Asker_Question_{cycle}'] = asker_question
            print(f"\nAsker's Question {cycle}:\n{asker_question}")
            logging.info(f"Asker's Question {cycle}: {asker_question}")

            # Responder answers the Asker's question
            responder_response = send_message_with_retry(
                responder_chat, 
                asker_question, 
                role='user',
                inline_image=inline_image,
            )
            interaction[f'Responder_Response_{cycle}'] = responder_response
            print(f"\nResponder's Response {cycle}:\n{responder_response}")
            logging.info(f"Responder's Response {cycle}: {responder_response}")

            # Update the previous_response for the next cycle
            previous_response = responder_response

        # Extract Final Score from the last response
        # Give a final total assessment
        final_assessment_prompt = (
            f"Based on the above chat history, summarize the image quality and provide a final score between 0 and 100."
        )
        final_score = send_message_with_retry(
                responder_chat, 
                final_assessment_prompt, 
                role='user',
                inline_image=inline_image,
            )
        interaction['Final_Score'] = final_score
        logging.info(f"Final Score for {image_name}: {final_score}")

        # Append the interaction to the results
        results.append(interaction)

        time.sleep(20)

    # Convert results to a DataFrame
    df = pd.DataFrame(results)

    # Save the DataFrame to an Excel file
    try:
        df.to_excel(output_excel, index=False)
        print(f"\nAssessment completed. Results saved to '{output_excel}'.")
        logging.info(f"Assessment completed. Results saved to '{output_excel}'.")
    except Exception as e:
        print(f"\nFailed to save results to Excel: {e}")
        logging.error(f"Failed to save results to Excel: {e}")

if __name__ == "__main__":
    main()