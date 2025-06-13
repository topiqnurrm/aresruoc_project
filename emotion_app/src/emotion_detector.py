from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_watson.natural_language_understanding_v1 import Features, EmotionOptions

# Ganti 'your-api-key' dan 'your-instance-url' dengan kredensial dari IBM Watson
api_key = 'your-api-key'
url = 'your-instance-url'

authenticator = IAMAuthenticator(api_key)
nlu = NaturalLanguageUnderstandingV1(
    version='2021-08-01',
    authenticator=authenticator
)
nlu.set_service_url(url)

# def emotion_predictor(text):
#     if not text.strip():
#         return {"error": "Input text is empty"}, 400
#     response = nlu.analyze(
#         text=text,
#         features=Features(emotion=EmotionOptions())
#     ).get_result()
#     emotions = response['emotion']['document']['emotion']
#     return emotions

def emotion_predictor(text):
    if not text.strip():
        return {"error": "Input text is empty"}, 400
    response = nlu.analyze(
        text=text,
        features=Features(emotion=EmotionOptions())
    ).get_result()
    emotions = response['emotion']['document']['emotion']
    return {
        "status": "success",
        "data": emotions
    }

def emotion_predictor(text):
    if not text.strip():
        return {"error": "Input text is empty"}, 400
    try:
        response = nlu.analyze(
            text=text,
            features=Features(emotion=EmotionOptions())
        ).get_result()
        emotions = response['emotion']['document']['emotion']
        return {
            "status": "success",
            "data": emotions
        }
    except Exception as e:
        return {"error": str(e)}, 500
