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

from image_encoder import encode_image
from chat_handler import initialize_chat, send_message_with_retry
from score_extractor import extract_final_score
from args_parser import parse_arguments

# Configure logging
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

    # Define system messages for Responder, Asker, and Judge
    responder_system_message = (
        "You are an Image Analysis Expert specializing in evaluating the quality of images based on their specific types, such as portraits and landscapes."
        "Your task is to identify, analysis and assess the key factors that genuinely impact the quality of each image."
        "For example, in portrait images, prioritize factors like the clarity of the subject and background aesthetics, while in landscape images, focus on the overall composition and detail emphasis. "
        "Avoid discussing photography techniques, image optimization processes, or any unrelated technical aspects. "
        "Provide comprehensive, objective, and type-specific evaluations that highlight the most significant quality determinants for each image."
    )

    asker_system_message = (
        "You are an Inquisitive Analyst focused on understanding the true factors that affect the quality of different types of images, such as portraits and landscapes. "
        "Your task is to ask insightful and targeted questions to the Image Analysis Expert to uncover the key quality determinants specific to each image type being evaluated. "
        "For instance, inquire about aspects like subject focus in portraits or composition intricacies in landscapes. "
        "Ensure that your questions remain relevant to assessing image quality and avoid straying into areas like photography techniques or image processing methods."
    )

    judge_system_message = (
        "You are a Conversation Moderator.Your task is to monitor the dialogue between the Inquisitive Analyst and the Image Analysis Expert."
        "Individually evaluate the Inquisitive Analyst's questions and the Image Analysis Expert's answers to ensure they strictly relate to assessing image quality based on this image's quality factors."  
        "If a question from the Inquisitive Analyst is off-topic, politely remind them to focus on image quality assessment and prompt them to regenerate the question. "
        "Similarly, if an answer from the Image Analysis Expert is off-topic, politely remind them to focus on image quality assessment and prompt them to regenerate the answer."
        "Your remind should start with \"Remind that:\""
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

        # Initialize chat sessions for Responder, Asker, and Judge
        try:
            responder_chat = initialize_chat(model, responder_system_message, role='model')
            asker_chat = initialize_chat(model, asker_system_message, role='model')
            judge_chat = initialize_chat(model, judge_system_message, role='model')
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
            f"This image is named {comparison_img_name}, and its score is 54. "
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
            f"Please assess the quality of the following image comprehensively, and based on the ComparisonIMG's score, "
            f"give this image a score between 0 and 100. "
            f"This image is named {image_name}. When evaluating, consider both objective factors "
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
        # Cycles 2-6: Question and Answer Rounds with Judge Evaluations
        # ------------------------------
        previous_response = responder_assessment_current  # Initialize with the current image assessment

        for cycle in range(1, num_cycles + 1):
            # ------------------------------
            # Step 1: Asker Generates a Question
            # ------------------------------
            asker_question_prompt = (
                f"Based on the following response from the Image Analysis Expert,  generate a thoughtful and insightful question to help the Image Analysis Expert correctly explore what factors really affect the quality of this image\n\n"
                f"Responder's Response:\n{previous_response}"
            )
            asker_question = send_message_with_retry(
                asker_chat, 
                asker_question_prompt, 
                role='user',
                inline_image=inline_image  # As questions don't include images
            )
            interaction[f'Asker_Question_{cycle}'] = asker_question
            print(f"\nAsker's Question {cycle}:\n{asker_question}")
            logging.info(f"Asker's Question {cycle}: {asker_question}")

            # ------------------------------
            # Step 2: Judge Evaluates Asker's Question
            # ------------------------------
            judge_evaluate_asker_prompt = (
                f"Evaluate the following question to determine if it is strictly on topic regarding image quality assessment. "
                f"Respond with 'on-topic' or 'off-topic'. If 'off-topic', provide a gentle reminder to the Inquisitive Analyst to focus on image quality assessment.Your remind should start with \"Remind that:\"\n\n"
                f"Asker's Question:\n{asker_question}"
            )
            judge_feedback_asker = send_message_with_retry(
                judge_chat, 
                judge_evaluate_asker_prompt, 
                role='user',
                inline_image=inline_image
            )
            print(f"\nJudge's Feedback on Asker's Question {cycle}:\n{judge_feedback_asker}")
            logging.info(f"Judge's Feedback on Asker's Question {cycle}:\n{judge_feedback_asker}")

            # Check Judge's feedback for Asker's question
            if re.search(r'\boff-topic\b', judge_feedback_asker, re.IGNORECASE):
                # Asker's question is off-topic; prompt regeneration
                # Extract the reminder from Judge's feedback
                reminder_match = re.search(r'Remind\s+that[^\n]*', judge_feedback_asker, re.IGNORECASE)
                reminder = reminder_match.group(0) if reminder_match else "Please focus on image quality assessment."

                print(f"\nJudge detected that the Asker went off topic. Prompting Asker to regenerate Question {cycle}.")
                logging.info(f"Judge detected Asker off-topic in Cycle {cycle}. Prompting regeneration.")

                # Prompt Asker to regenerate the question
                asker_regenerate_prompt = (
                    f"{reminder}\n\n"
                    f"Based on the following response from the Image Analysis Expert, regenerate the question to stay on topic:\n\n"
                    f"Responder's Response:\n{previous_response}\n\n"
                )
                regenerated_asker_question = send_message_with_retry(
                    asker_chat, 
                    asker_regenerate_prompt, 
                    role='user',
                    inline_image=inline_image
                )
                interaction[f'Asker_Question_{cycle}'] = regenerated_asker_question
                print(f"\nAsker's Regenerated Question {cycle}:\n{regenerated_asker_question}")
                logging.info(f"Asker's Regenerated Question {cycle}: {regenerated_asker_question}")

                # Re-evaluate the regenerated question
                judge_feedback_regenerated_asker = send_message_with_retry(
                    judge_chat, 
                    judge_evaluate_asker_prompt.replace(asker_question, regenerated_asker_question), 
                    role='user',
                    inline_image=inline_image
                )
                print(f"\nJudge's Feedback on Regenerated Asker's Question {cycle}:\n{judge_feedback_regenerated_asker}")
                logging.info(f"Judge's Feedback on Regenerated Asker's Question {cycle}:\n{judge_feedback_regenerated_asker}")

                if re.search(r'\boff-topic\b', judge_feedback_regenerated_asker, re.IGNORECASE):
                    # Still off-topic after regeneration; skip to next cycle or handle accordingly
                    print(f"\nAsker's regenerated question is still off-topic. Skipping Cycle {cycle}.")
                    logging.warning(f"Asker's regenerated question still off-topic in Cycle {cycle}. Skipping.")
                    continue  # Or implement further handling
                else:
                    # Regenerated question is on-topic; proceed
                    asker_question = regenerated_asker_question
            else:
                # Asker's question is on-topic; proceed
                logging.info(f"Asker's Question {cycle} is on-topic.")

            # ------------------------------
            # Step 3: Responder Answers the Question
            # ------------------------------

            asker_question += f"\n\nAfter answer the quastion, please state what you now think are the key factors that really affects the quality of this image {image_name}, and give a brief summary."

            responder_response = send_message_with_retry(
                responder_chat, 
                asker_question, 
                role='user',
                inline_image=inline_image  # Assuming answers don't include images
            )
            interaction[f'Responder_Response_{cycle}'] = responder_response
            print(f"\nResponder's Response {cycle}:\n{responder_response}")
            logging.info(f"Responder's Response {cycle}: {responder_response}")

            # ------------------------------
            # Step 4: Judge Evaluates Responder's Answer
            # ------------------------------
            judge_evaluate_responder_prompt = (
                f"Evaluate the following answer to determine if it is strictly on topic regarding image quality assessment. "
                f"Respond with 'on-topic' or 'off-topic'. If 'off-topic', provide a gentle reminder to the Image Analysis Expert to focus on image quality assessment.Your remind should start with \"Remind that:\"\n\n"
                f"Responder's Response:\n{responder_response}"
            )
            judge_feedback_responder = send_message_with_retry(
                judge_chat, 
                judge_evaluate_responder_prompt, 
                role='user',
                inline_image=inline_image
            )
            print(f"\nJudge's Feedback on Responder's Response {cycle}:\n{judge_feedback_responder}")
            logging.info(f"Judge's Feedback on Responder's Response {cycle}:\n{judge_feedback_responder}")

            # Check Judge's feedback for Responder's answer
            if re.search(r'\boff-topic\b', judge_feedback_responder, re.IGNORECASE):
                # Responder's answer is off-topic; prompt regeneration
                # Extract the reminder from Judge's feedback
                reminder_match = re.search(r'Remind\s+that[^\n]*', judge_feedback_responder, re.IGNORECASE)
                reminder = reminder_match.group(0) if reminder_match else "Please focus on image quality assessment."

                print(f"\nJudge detected that the Responder went off topic. Prompting Responder to regenerate Response {cycle}.")
                logging.info(f"Judge detected Responder off-topic in Cycle {cycle}. Prompting regeneration.")

                # Prompt Responder to regenerate the answer
                responder_regenerate_prompt = (
                    f"{reminder}\n\n"
                    f"Based on the following question, regenerate your answer to focus specifically on the key factors affecting the quality of this image:\n\n"
                    f"Asker's Question {cycle}:\n{asker_question}\n\n"
                    f"Responder's Original Response:\n{responder_response}"
                )
                regenerated_responder_response = send_message_with_retry(
                    responder_chat, 
                    responder_regenerate_prompt, 
                    role='user',
                    inline_image=inline_image
                )
                interaction[f'Responder_Response_{cycle}'] = regenerated_responder_response
                print(f"\nResponder's Regenerated Response {cycle}:\n{regenerated_responder_response}")
                logging.info(f"Responder's Regenerated Response {cycle}: {regenerated_responder_response}")

                # Re-evaluate the regenerated answer
                judge_feedback_regenerated_responder = send_message_with_retry(
                    judge_chat, 
                    judge_evaluate_responder_prompt.replace(responder_response, regenerated_responder_response), 
                    role='user',
                    inline_image=inline_image
                )
                print(f"\nJudge's Feedback on Regenerated Responder's Response {cycle}:\n{judge_feedback_regenerated_responder}")
                logging.info(f"Judge's Feedback on Regenerated Responder's Response {cycle}:\n{judge_feedback_regenerated_responder}")

                if re.search(r'\boff-topic\b', judge_feedback_regenerated_responder, re.IGNORECASE):
                    # Still off-topic after regeneration; skip to next cycle or handle accordingly
                    print(f"\nResponder's regenerated answer is still off-topic. Skipping Cycle {cycle}.")
                    logging.warning(f"Responder's regenerated answer still off-topic in Cycle {cycle}. Skipping.")
                    continue  # Or implement further handling
                else:
                    # Regenerated answer is on-topic; proceed
                    responder_response = regenerated_responder_response
            else:
                # Responder's answer is on-topic; proceed
                logging.info(f"Responder's Response {cycle} is on-topic.")

            # ------------------------------
            # Update previous_response for next cycle
            # ------------------------------
            previous_response = responder_response

            # Optional: Sleep to respect API rate limits
            time.sleep(2)  # Adjust as necessary

        # ------------------------------
        # Final Assessment by Judge
        # ------------------------------
        final_assessment_prompt = (
            f"Conversation History:\n"
        )

        # Compile the conversation history
        for cycle in range(1, num_cycles + 1):
            final_assessment_prompt += (
                f"\nAsker's Question {cycle}:\n{interaction.get(f'Asker_Question_{cycle}', '')}\n\n"
                f"Responder's Response {cycle}:\n{interaction.get(f'Responder_Response_{cycle}', '')}\n\n"
            )
        
        final_assessment_prompt += "\nCombine with your previous chat, provide a concise summary of the image quality with the main factors that really affect the quality of this image and then assign a final score and provide a final score between 0 and 100."

        final_score_response = send_message_with_retry(
            responder_chat,
            final_assessment_prompt,
            role='user',
            inline_image=inline_image
        )
        interaction['Final_Score'] = final_score_response
        print(f"\nJudge's Final Assessment and Score for {image_name}:\n{final_score_response}")
        logging.info(f"Judge's Final Assessment and Score for {image_name}:\n{final_score_response}")

        # Append the interaction to the results
        results.append(interaction)

        # Optional: Sleep to respect API rate limits before processing the next image
        time.sleep(20)  # Adjust as necessary based on API usage policies

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