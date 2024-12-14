import requests
import json

def test_produce_sentences(url: str):

    # Test input
    input_data = {
        "ids": [
            0,
            1,
            10,
            2,
            3,
            6,
            4,
            5,
            7,
            8
        ],
        "sentences": [
            "The rapid advancement of artificial intelligence (AI) is profoundly transforming our lives, work, and the way society operates.",
            "As a multidisciplinary technology, AI encompasses fields such as machine learning, natural language processing, and computer vision, with applications spanning a wide array of domains.",
            "In conclusion, artificial intelligence is a pivotal force shaping the future of technology and society.",
            "From virtual assistants to autonomous vehicles, AI has become an integral part of our daily routines.",
            "In education, AI offers personalized learning solutions tailored to individual progress and needs, significantly enhancing learning efficiency.",
            "However, the development of AI also brings challenges.",
            "In healthcare, AI technologies can analyze vast amounts of data, assisting doctors in diagnosing diseases and devising treatment plans, potentially saving countless lives.",
            "Furthermore, in industrial production, AI-driven automation has drastically improved productivity while reducing labor costs.",
            "For instance, the widespread adoption of automation might lead to the displacement of certain traditional jobs, reshaping employment structures.",
            "Additionally, concerns about the transparency and security of AI systems remain critical issues that demand attention."
        ],
        }

    # Send POST request
    response = requests.post(url, json=input_data)

    # Validate response
    if response.status_code == 200:
        print("Test Passed!")
        print("Response JSON:")
        print(json.dumps(response.json(), indent=4))
    else:
        print("Test Failed!")
        print(f"Status Code: {response.status_code}")
        print(f"Error: {response.text}")

# Run the test
if __name__ == "__main__":
    url = "http://127.0.0.1:5001//presumm"
    test_produce_sentences(url)
