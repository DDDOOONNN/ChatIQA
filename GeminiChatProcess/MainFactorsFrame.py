import os
import pandas as pd
from PIL import Image
import google.generativeai as genai
from tqdm import tqdm
import time
import base64
from io import BytesIO

def encode_image(image_path):
    """
    Encode an image file to Base64 data URI format.
    """
    with open(image_path, "rb") as img_file:
        img_bytes = img_file.read()
    img_b64 = base64.b64encode(img_bytes).decode('utf-8')
    img_data_uri = f"data:image/jpeg;base64,{img_b64}"
    return img_data_uri

def main():
    # 获取环境变量中的API密钥
    api_key = os.getenv('GENAI_API_KEY')

    if not api_key:
        raise ValueError("API key not found. Please set the 'GENAI_API_KEY' environment variable.")

    # 配置Google Generative AI
    genai.configure(api_key=api_key, transport='rest')
    model = genai.GenerativeModel('gemini-1.5-flash')

    # 定义包含图像的目录
    image_dir = r'C:\AA\data\ImageDatabase'  # 使用raw string避免转义字符问题

    # 初始化结果列表
    results = []

    # 定义图像的总数
    total_images = 20

    # 迭代处理每张图像，带有进度条
    for i in tqdm(range(1, total_images + 1), desc="Processing Images"):
        image_name = f'DatabaseImage{str(i).zfill(4)}.jpg'  # 根据需要调整扩展名
        image_path = os.path.join(image_dir, image_name)
        print(f"\nProcessing image: {image_path}")  # 打印正在处理的图像路径

        if not os.path.exists(image_path):
            assessment = 'Image not found.'
            print(f"Warning: {image_name} does not exist.")
            results.append({'Image': image_name, 'Assessment': assessment})
            continue

        try:
            # 打开并准备图像
            with Image.open(image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                img_data_uri = encode_image(image_path)

        except Exception as e:
            assessment = f"Error opening image: {e}"
            results.append({'Image': image_name, 'Assessment': assessment})
            continue

        # 初始化重试机制
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                # 每次循环开始时初始化对话内容
                initial_content = {
                    "parts": [
                        {
                            "text": (
                                "You are an expert in image quality assessment. "
                                "Your task is to identify and outline the primary image quality concerns that are generally applicable to a wide range of images. "
                                "After determining these key factors, ensure that your assessments consistently reference the same set of factors in every new conversation, providing a comprehensive and objective evaluation."
                            )
                        },
                        {
                            "text": (
                                "Understood. I will focus on the following primary image quality concerns that are commonly applicable to a wide range of images:\n\n"
                                "Sharpness: This refers to the clarity and detail in an image. A sharp image has well-defined edges and visible fine details.\n"
                                "Contrast: This is the difference between the darkest and lightest parts of an image. High contrast images have a wide range of tones, while low contrast images appear flat or washed out.\n"
                                "Color Accuracy: This refers to how accurately the colors in an image represent the real-world colors. A color-accurate image should have natural-looking colors that are consistent with the original scene.\n"
                                "Noise: This is unwanted random variations in pixel intensity. Noise can cause an image to appear grainy or speckled.\n"
                                "Artifacts: These are unnatural or distorted elements that appear in an image due to processing or compression. Artifacts can take many forms, such as blockiness, ringing, or color banding.\n"
                                "Exposure: This refers to the overall brightness or darkness of an image. An overexposed image is too bright, while an underexposed image is too dark.\n"
                                "Focus: This refers to the sharpness of the subject in relation to the background. A properly focused image has a sharp subject and a blurred background.\n\n"
                                "I will use these factors as a consistent framework for evaluating image quality in this conversation."
                            )
                        }
                    ]
                }

                # 创建带有结构化内容的用户部分
                user_content = {
                    "parts": [
                        {
                            "text": (
                                "Please assess the quality of the following image comprehensively. "
                                "Identify the key aspects that determine the image's quality and provide a detailed assessment for each aspect."
                            )
                        },
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": img_data_uri.split(",")[1]
                            }
                        }
                    ]
                }

                # 合并初始内容和用户内容
                temp_content = {
                    "parts": initial_content["parts"] + user_content["parts"]
                }

                # 调用API生成内容
                response = model.generate_content(temp_content)
                assistant_response = response.text.strip()

                # 保存评估结果
                assessment = assistant_response
                break  # 成功后退出重试循环

            except Exception as e:
                assessment = f"Error processing image on attempt {attempt}: {e}"
                if attempt < max_retries:
                    print(f"Attempt {attempt} failed for {image_name}. Retrying in 2 seconds...")
                    time.sleep(2)  # 重试前等待
                else:
                    print(f"Failed to process {image_name} after {max_retries} attempts.")

        # 将结果添加到结果列表
        results.append({'Image': image_name, 'Assessment': assessment})

    # 将结果转换为DataFrame
    df = pd.DataFrame(results)

    # 保存DataFrame到Excel文件
    output_excel = 'MainFactors1.xlsx'
    try:
        df.to_excel(output_excel, index=False)
        print(f"\nAssessment completed. Results saved to '{output_excel}'.")
    except Exception as e:
        print(f"\nFailed to save results to Excel: {e}")

if __name__ == "__main__":
    main()