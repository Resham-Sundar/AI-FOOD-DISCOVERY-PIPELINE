import streamlit as st
from PIL import Image
from io import BytesIO
import base64
import redis
import json
import time

# Connect to Redis
redis_conn = redis.Redis(host="localhost", port=6379, db=0)

# Streamlit UI
st.title("Where is this delicacy?")
st.write("Upload the image along with a query.")

# Image and Query Input
uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
user_query = st.text_area("Enter your query:", "")

# Submit Button
if st.button("Submit"):
    if uploaded_image is not None and user_query.strip():
        # Convert image to Base64
        image = Image.open(uploaded_image).convert("RGB")
        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        # Prepare Data
        alarm_id = f"alarm_{int(time.time())}"
        message_id = f"msg_{int(time.time())}"
        data = {
            "AlarmId": alarm_id,
            "MessageId": message_id,
            "Base64": image_base64,
            "Prompt": user_query,
        }
        
        # Push to Redis image queue
        redis_conn.lpush("image_queue", json.dumps(data))
        # st.success(f"Data enqueued with AlarmId: {alarm_id}")

        # Poll for Results
        st.write("Processing your request. Please wait...")
        while True:
            print(redis.ConnectionError)
            result_data = redis_conn.rpop("result_queue")
            print(result_data)
            if result_data:
                result_data = json.loads(result_data)
                if result_data["id"] == alarm_id:
                    st.subheader("Response:")
                    st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)
                    st.text_area("Generated Answer:", result_data["result"])
                    break
            time.sleep(1)  # Small delay to avoid busy waiting
    else:
        st.error("Please upload an image and enter a query.")


