import chat_handler
from args_parser import parse_arguments
from image_encoder import encode_image
import argparse
import os


args = parse_arguments()
image_dir = args.image_dir
comparison_img_name = args.comparison_img

comparison_img_path = os.path.join(image_dir, comparison_img_name)
comparison_img_data_uri = encode_image(comparison_img_path)



initialize_with_image = chat_handler.initialize_chat("hi", role = "system")

reponse = chat_handler.send_message_with_retry(initialize_with_image, "Do you see this image?", role='user', inline_image=comparison_img_data_uri)
print(reponse)



