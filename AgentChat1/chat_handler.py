import logging
import time

def initialize_chat(model, system_message, role='model'):
    """
    Initialize a chat session with the given system message.
    
    Parameters:
    - model: The generative AI model instance.
    - system_message: The system message to initialize the chat.
    - role: The role of the message sender ('user', or 'model').
    
    Returns:
    - chat_session: The initialized chat session.
    """
    try:
        history = [
            {
                "role": role,
                "parts": [{"text": system_message}]
            }
        ]
        return model.start_chat(history=history)
    except Exception as e:
        logging.error(f"Failed to initialize chat session: {e}")
        raise e

def send_message(chat_session, message, role='user', inline_image=None):
    """
    Send a message to the chat session and return the response.
    
    Parameters:
    - chat_session: The initialized chat session.
    - message: The text message to send.
    - role: The role of the message sender ('user' or 'model').
    - inline_image: (Optional) A dictionary containing 'mime_type' and 'data' for images.
    
    Returns:
    - response_text: The text response from the model.
    """
    try:
        if inline_image:
            # If there's an image, include it in 'inline_data'
            parts = [
                {"text": message},
                {"inline_data": {
                    "mime_type": inline_image['mime_type'],
                    "data": inline_image['data']
                }}
            ]
            user_message = {"role": role, "parts": parts}
        else:
            # Only text
            user_message = {"role": role, "parts": [{"text": message}]}
        
        response = chat_session.send_message(user_message)
        return response.text.strip()
    except Exception as e:
        logging.error(f"Error sending message: {e}")
        return f"Error: {e}"

def send_message_with_retry(chat_session, message, role='user', inline_image=None, retries=3, delay=5):
    """
    Send a message with retry logic.
    
    Parameters:
    - chat_session: The initialized chat session.
    - message: The text message to send.
    - role: The role of the message sender ('user' or 'model').
    - inline_image: (Optional) A dictionary containing 'mime_type' and 'data' for images.
    - retries: Number of retry attempts.
    - delay: Delay in seconds between retries.
    
    Returns:
    - response_text: The text response from the model or error message.
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
