import os
import pandas as pd
from PIL import Image
import google.generativeai as genai
from tqdm import tqdm
import time
import base64

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
    image_dir = r'C:\AA\data\ImageDatabase'  # 使用原始字符串避免转义字符问题

    # 定义 ComparisonIMG 的路径
    comparison_img_name = 'ComparisonIMG.jpg'  # 根据实际文件名调整
    comparison_img_path = os.path.join(image_dir, comparison_img_name)

    if not os.path.exists(comparison_img_path):
        raise FileNotFoundError(f"Comparison image '{comparison_img_name}' not found in '{image_dir}'.")

    # 编码 ComparisonIMG
    try:
        comparison_img_data_uri = encode_image(comparison_img_path)
    except Exception as e:
        raise Exception(f"Error encoding comparison image: {e}")

    # 初始化结果列表
    results = []

    # 定义图像的总数
    total_images = 1

    # 定义初始对话内容
    initial_system_instruction = (
        "You are an expert in image quality assessment. "
        "Your task is to identify and outline the primary image quality concerns that are generally applicable to a wide range of images. "
        "After determining these key factors, ensure that your assessments consistently reference the same set of factors in every new conversation, providing a comprehensive and objective evaluation."
    )

    initial_assistant_response = (
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

    # 迭代处理每张图像，带有进度条
    for i in tqdm(range(1, total_images + 1), desc="Processing Images"):
        image_num = i
        image_name = f'DatabaseImage{str(i).zfill(4)}.jpg' # 图片命名为 DataImage1.jpg, Image2.jpg, ...
        image_path = os.path.join(image_dir, image_name)
        print(f"\nProcessing image: {image_path}")  # 打印正在处理的图像路径

        if not os.path.exists(image_path):
            assessment_comparison = None
            assessment_current = 'Image not found.'
            print(f"Warning: {image_name} does not exist.")
            results.append({
                'Image': image_name,
                'Assessment_ComparisonIMG': assessment_comparison,
                'Assessment_CurrentImage': assessment_current
            })
            continue

        try:
            # 打开并准备当前图像
            with Image.open(image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                current_img_data_uri = encode_image(image_path)
        except Exception as e:
            assessment_comparison = None
            assessment_current = f"Error opening image: {e}"
            results.append({
                'Image': image_name,
                'Assessment_ComparisonIMG': assessment_comparison,
                'Assessment_CurrentImage': assessment_current
            })
            continue

        # 初始化重试机制
        max_retries = 3
        assessment_comparison = None
        assessment_current = None

        # ------------------------------------
        # 创建新的 ChatSession
        # ------------------------------------
        try:
            # 初始化对话历史，包括系统指令和助手初始响应
            history = [
                {
                    "parts": [{"text": initial_system_instruction}],
                    "role": "user"
                },
                {
                    "parts": [{"text": initial_assistant_response}],
                    "role": "model"
                }
            ]

            chat = model.start_chat(history=history)
        except Exception as e:
            assessment_comparison = None
            assessment_current = f"Error initializing chat session: {e}"
            results.append({
                'Image': image_name,
                'Assessment_ComparisonIMG': assessment_comparison,
                'Assessment_CurrentImage': assessment_current
            })
            continue

        # ------------------------------------
        # 第一轮对话：发送 ComparisonIMG
        # ------------------------------------
        for attempt in range(1, max_retries + 1):
            try:
                # 构建用户消息，包含文本和 ComparisonIMG 图像
                user_message_comparison = {
                    "parts": [
                        {
                            "text": (
                                f"Please assess the quality of the following image comprehensively. "
                                f"This image is named ComparisonIMG. Identify the key aspects that determine the image's quality and provide a detailed assessment for each aspect."
                            )
                        },
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": comparison_img_data_uri.split(",")[1]
                            }
                        }
                    ]
                }

                # 发送消息并接收响应
                response_comparison = chat.send_message(user_message_comparison)
                assistant_response_comparison = response_comparison.text.strip()

                # 打印和记录评估结果
                assessment_comparison = assistant_response_comparison
                print(f"Assessment for ComparisonIMG: {assessment_comparison}")
                break  # 成功后退出重试循环

            except Exception as e:
                assessment_comparison = f"Error processing ComparisonIMG on attempt {attempt}: {e}"
                if attempt < max_retries:
                    print(f"Attempt {attempt} failed for ComparisonIMG. Retrying in 2 seconds...")
                    time.sleep(2)  # 重试前等待
                else:
                    print(f"Failed to process ComparisonIMG after {max_retries} attempts.")

        # ------------------------------------
        # 第二轮对话：发送当前图片
        # ------------------------------------
        for attempt in range(1, max_retries + 1):
            try:
                # 构建用户消息，包含文本和当前图片
                user_message_current = {
                    "parts": [
                        {
                            "text": (
                                f"Please assess the quality of the following image comprehensively. "
                                f"This image is named {image_name}. Identify the key aspects that determine the image's quality and provide a detailed assessment for each aspect."
                            )
                        },
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": current_img_data_uri.split(",")[1]
                            }
                        }
                    ]
                }

                # 发送消息并接收响应
                response_current = chat.send_message(user_message_current)
                assistant_response_current = response_current.text.strip()

                # 打印和记录评估结果
                assessment_current = assistant_response_current
                print(f"Assessment for {image_name}: {assessment_current}")
                break  # 成功后退出重试循环

            except Exception as e:
                assessment_current = f"Error processing {image_name} on attempt {attempt}: {e}"
                if attempt < max_retries:
                    print(f"Attempt {attempt} failed for {image_name}. Retrying in 2 seconds...")
                    time.sleep(2)  # 重试前等待
                else:
                    print(f"Failed to process {image_name} after {max_retries} attempts.")

        # 将结果添加到结果列表
        results.append({
            'Image': image_name,
            'Assessment_ComparisonIMG': assessment_comparison,
            'Assessment_CurrentImage': assessment_current
        })

    # 将结果转换为DataFrame
    df = pd.DataFrame(results)

    # 保存DataFrame到Excel文件
    output_excel = 'ChatFactors2.xlsx'
    try:
        df.to_excel(output_excel, index=False)
        print(f"\nAssessment completed. Results saved to '{output_excel}'.")
    except Exception as e:
        print(f"\nFailed to save results to Excel: {e}")

if __name__ == "__main__":
    main()
