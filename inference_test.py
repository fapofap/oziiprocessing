from inference_sdk import InferenceHTTPClient
your_image = "test\istock.jpg"  # Replace with your image path
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="API_KEY"
)

result = CLIENT.infer(your_image, model_id="helmet-atofc/2")