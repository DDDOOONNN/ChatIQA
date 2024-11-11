# chat_handler.py

import logging
import time
import os
from openai import OpenAI

client = OpenAI(
    base_url='https://xiaoai.plus/v1',
    # sk-xxx替换为自己的key
    api_key='sk-m5aPEydbaoJIY700x9dLJgaMZ07fCmsrcqSLYmRvQlhBGzIK'
)

def initialize_chat(system_message, role='system'):
    """
    Initialize a chat session with the given system message.
    
    Parameters:
    - system_message: The system message to initialize the chat.
    - role: The role of the message sender ('system', 'user', etc.).
    
    Returns:
    - chat_session: A list representing the chat history.
    """
    try:
        history = [
            {
                "role": role,
                "content": system_message
            }
        ]
        return history
    except Exception as e:
        logging.error(f"Failed to initialize chat session: {e}")
        raise e

def send_message(chat_session, message, role='user', inline_image=None):
    """
    Send a message to the chat session and return the response.
    
    Parameters:
    - chat_session: The current chat history.
    - message: The text message to send.
    - role: The role of the message sender ('user' or 'assistant').
    - inline_image: (Optional) A dictionary containing 'mime_type' and 'data' for images.
    
    Returns:
    - response_text: The text response from the model.
    """
    try:
        if inline_image:
            # If there's an image, include it in 'inline_data'
            user_message = {
                "role": role,
                "content": [
                    {"type": "text", "text": message},
                    {
                        "type": "image_url", 
                        "image_url": {
                        "url": inline_image,
                       }
                    }
                ]

            }
        else:
            # Only text
            user_message = {
                "role": role,
                "content": message
            }
        
        # Append user message to history
        chat_session.append(user_message)
        
        # Create chat completion using OpenAI's API
        chat_completion = client.chat.completions.create(
            model="gpt-4o", 
            messages=chat_session
        )
        
        # Extract the assistant's response
        response = chat_completion.choices[0].message.content.strip()
        
        # Append assistant's response to history
        chat_session.append({
            "role": "assistant",
            "content": response
        })
        
        return response
    except Exception as e:
        logging.error(f"Error sending message: {e}")
        return f"Error: {e}"

def send_message_with_retry(chat_session, message, role='user', inline_image=None, retries=3, delay=5):
    """
    Send a message with retry logic.
    
    Parameters:
    - chat_session: The current chat history.
    - message: The text message to send.
    - role: The role of the message sender ('user' or 'assistant').
    - inline_image: (Optional) A dictionary containing 'mime_type' and 'data' for images.
    - retries: Number of retry attempts.
    - delay: Delay in seconds between retries.
    
    Returns:
    - response_text: The text response from the model or an error message.
    """
    for attempt in range(1, retries + 1):
        response = send_message(chat_session, message, role, inline_image)
        if not response.startswith("Error:"):
            return response
        else:
            logging.warning(f"Attempt {attempt} failed with error: {response}")
            if attempt < retries:
                logging.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                logging.error("All retry attempts failed.")
                return response
    
    time.sleep(2)