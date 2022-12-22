# GoogleVisionAIInterface
Interface for Google Vision AI API

## How to use

1. Follow [these instructions](https://cloud.google.com/vision/docs/detect-labels-image-client-libraries) up to step 6 (setting the environment variable).
2. Create a file in this directory called `.env` containing:
```
GOOGLE_APPLICATION_CREDENTIALS="/path/to/file.json"
```
3. Run `pip install -r requirements.txt` to install all of the dependencies.
4. Execute main.py on an image like `python3 main.py /path/to/image.jpg`.
