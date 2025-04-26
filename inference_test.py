from inference_sdk import InferenceHTTPClient
your_image = r"test\istock.jpg"  # Replace with your image path
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="rf_Jnfc9ahHilQtL7iLHYu2ObiyOoB2"
)

result = CLIENT.infer(your_image, model_id="helmet-atofc/2")
