import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Dynamic Image Analysis with Responder and Asker Roles.")
    parser.add_argument('--image_dir', type=str, default=r"C:\AA\data\ImageDatabase", help='Directory containing images.')
    parser.add_argument('--comparison_img', type=str, default=r"ComparisonIMG2.jpg", help='Name of the comparison image.')
    parser.add_argument('--output_excel', type=str, default='Prompt_BID.xlsx', help='Output Excel file name.')
    parser.add_argument('--total_images', type=int, default=1, help='Total number of images to process.')
    parser.add_argument('--num_cycles', type=int, default=1, help='Number of question-answer cycles.')
    args = parser.parse_args()
    return args