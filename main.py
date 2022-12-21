from dotenv import load_dotenv  # API credentials
import io  # bytes I/O for the API
from pathlib3x import Path  # file path handling
from google.cloud import vision  # Google Vision AI API
import argparse  # CLI argument parsing
import json  # saving the output as JSON

load_dotenv()
MAX_RESULTS = 5  # the maximum number of faces for the AI to detect


def init_client():
    # Init API client
    client = vision.ImageAnnotatorClient()
    return client


def analyse(image, client):
    # do the thing
    response = client.face_detection(image=image, max_results=MAX_RESULTS)

    # convert output to a format I can actually use
    faces = annotations_to_dict(response.face_annotations)

    return faces


def main():
    # Argument parsing
    parser = argparse.ArgumentParser(
        prog="Google Vision ERA",
        description="Helper for the emotion recognition part of Google Cloud's Vision AI.",
    )
    output = parser.add_mutually_exclusive_group()
    parser.add_argument("file", help="Image file to perform emotion recognition on.")
    output.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Remove output message, just save the data.",
    )
    output.add_argument(
        "-n", "--no-save", action="store_true", help="Do not save the data."
    )
    args = parser.parse_args()

    # input and output file paths
    inp = Path(args.file)
    out = inp.append_suffix(".json")

    # read the input file into an Image object
    with io.open(inp, "rb") as f:
        content = f.read()

    client = init_client()
    image = vision.Image(content=content)

    faces = analyse(image, client)

    # save to JSON
    if not args.no_save:
        with open(out, "w") as f:
            json.dump(faces, f)

    # print results of ERA
    if not args.quiet:
        for face in faces:
            print(f"Face confidence: {face['detection_confidence']}")
            print("Emotions:")
            # ah yes, the four emotions
            # I bet the Google devs never watched Inside Out
            print(f" - Joy: {face['joy_likelihood']}")
            print(f" - Sorrow: {face['sorrow_likelihood']}")
            print(f" - Anger: {face['anger_likelihood']}")
            print(f" - Surprise: {face['surprise_likelihood']}")


# Helper function to convert the API output to a usable format (python dict)
def annotations_to_dict(faces):
    out = []
    for face in faces:
        out_elem = {}
        nest_stack = [out_elem]
        lines = face.__str__().split(
            "\n"
        )  # only way to serialise the data that actually preserves most of it
        for line in lines:
            if line.strip() == "":
                continue
            elif line[-1] == "{":
                key = line[:-1].strip()  # take off the { and strip any spaces
                new_obj = {}

                if key in nest_stack[-1]:
                    if type(nest_stack[-1][key]) is dict:
                        nest_stack[-1][key] = [nest_stack[-1][key], new_obj]
                    else:
                        nest_stack[-1][key].append(new_obj)
                else:
                    nest_stack[-1][key] = new_obj
                nest_stack.append(new_obj)
            elif line[-1] == "}":
                nest_stack.pop()
            else:
                k, v = line.split(":")
                k = k.strip()
                v = v.strip()
                if v[-1].isdigit():
                    v = float(v)
                nest_stack[-1][k] = v
        out.append(out_elem)
    return out


if __name__ == "__main__":
    main()
