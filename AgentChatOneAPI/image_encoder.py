import base64
import logging

def encode_image(image_path):
    """
    Encode an image file to Base64 data URI format.
    """
    try:
        with open(image_path, "rb") as img_file:
            img_bytes = img_file.read()
        img_b64 = base64.b64encode(img_bytes).decode('utf-8')
        img_data_uri = f"data:image/jpeg;base64,{img_b64}"
        return img_data_uri
    except Exception as e:
        logging.error(f"Failed to encode image {image_path}: {e}")
        raise e