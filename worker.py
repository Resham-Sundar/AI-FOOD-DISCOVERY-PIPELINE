import redis
from blipworker import Blip
from PIL import Image
from io import BytesIO
import base64
import json
from datetime import datetime
import lovely_logger as log
from datetime import datetime

log.init("worker.log", max_kb=10024)

redis_conn = redis.Redis(host="localhost", port=6379, db=0)
blip = Blip()

knowledge_embeddings = blip.create_knowledge_base()
blip.store_embeddings(knowledge_embeddings)

def process_images():
    while True:
        message_alarm_data = {}
        data = redis_conn.rpop("image_queue")
        try:
            data = json.loads(data)
        except:
            continue

        image = Image.open(BytesIO(base64.b64decode(data["Base64"])))

        # Resizing image for inference
        user_prompt = "USER: <image>\n {} \nASSISTANT:".format(data["Prompt"])
        message_alarm_data.update({data["AlarmId"]:data["MessageId"]})
        # batch.append((data["AlarmId"], image, user_prompt))
        log.info(
            "[INFO] Processing Id {}".format(data["AlarmId"])
        )
        image_caption,top_results = blip.process_query(image, user_prompt)
        # rag_response = blip.get_final_result(image_caption, user_prompt, top_results)
        final_result = f"""
        Here are the top 2 similar places:
        1. {top_results[0]['name']}: {top_results[0]['description']}
        Address: {top_results[0]['address']}
        Rating: {top_results[0]['metadata']['rating']}
        2. {top_results[1]['name']}: {top_results[1]['description']}
        Address: {top_results[1]['address']}
        Rating: {top_results[1]['metadata']['rating']}
        """

        log.info(
            "[INFO] {} | Id: {} Result: {}".format(
                datetime.now(), data["AlarmId"], final_result
            )
        )
        answer = final_result
        result = {
            "id": data["AlarmId"],
            "result": answer,
            "MessageId": message_alarm_data[data["AlarmId"]]
        }
        print("PUSHING INTO QUEUE")
        try:
            # Ensure the response is not empty
            if json.dumps(result):
                redis_conn.lpush('result_queue', json.dumps(result))
                print(f"Successfully pushed response to Redis")
            else:
                print("Final response is empty, skipping push to Redis.")
        except redis.ConnectionError as e:
            log.error(f"Failed to connect to Redis: {e}")
        except Exception as e:
            log.error(f"Error pushing to Redis: {e}")


if __name__ == "__main__":
    process_images()
