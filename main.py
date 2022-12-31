from dotenv import load_dotenv  # API credentials
import io  # bytes I/O for the API
from pathlib3x import Path  # file path handling
from google.cloud import vision  # Google Vision AI API
import argparse  # CLI argument parsing
import json  # saving the output as JSON

import PIL.Image, PIL.ImageDraw

load_dotenv()


def init_client():
    """Initialise the API client

    :return: initialised API connection
    :rtype: class:`google.cloud.vision.ImageAnnotatorClient`
    """
    client = vision.ImageAnnotatorClient()
    return client


def highlight_faces(image, faces):
    """Draws a polygon around the faces. From https://github.com/GoogleCloudPlatform/python-docs-samples/blob/main/vision/snippets/face_detection/faces.py

    :param image: a file pointer to the image
    :type image: BinaryIO
    :param faces: a list of faces found in the file. This should be in the format returned by the Vision API.
    :type faces: class:`google.cloud.vision.FaceAnnotation`
    :return: an image with the faces highlighted.
    :rtype: class:`PIL.Image`
    """
    im = PIL.Image.open(image)
    draw = PIL.ImageDraw.Draw(im)
    # Sepecify the font-family and the font-size
    for face in faces:
        box = [(vertex.x, vertex.y)
               for vertex in face.bounding_poly.vertices]
        draw.line(box + [box[0]], width=5, fill='#00ff00')
        # Place the confidence value/score of the detected faces above the
        # detection box in the output image
        draw.text(((face.bounding_poly.vertices)[0].x,
                   (face.bounding_poly.vertices)[0].y - 30),
                  str(format(face.detection_confidence, '.3f')) + '%',
                  fill='#FF0000')
    return im

def analyse(image_file, client, max_results=5):
    """Analyse a given image using the API client

    :param image: a file pointer to the image to analyse
    :type image: BinaryIO
    :param client: the API client to use
    :type client: class:`google.cloud.vision.ImageAnnotatorClient`
    :return: The results of the analysis
    :rtype: dict
    """
    image = vision.Image(content=image_file)

    # do the thing
    response = client.face_detection(image=image, max_results=max_results)

    return response.face_annotations


def main():
    """Main function; parses command-line arguments, reads in an image, and calls the API to analyse it."""
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
        "-s", "--no-save", action="store_true", help="Do not save the data."
    )
    parser.add_argument("-n", "--number", default=5, type=int, help="Maximum number of faces to detect.")
    args = parser.parse_args()

    # input and output file paths
    inp = Path(args.file)
    out = inp.append_suffix(".json")
    out_img = inp.with_suffix('.hl.jpg')

    # read the input file
    with io.open(inp, "rb") as f:
        content = f.read()

        # initiliase the client and use it
        client = init_client()
        res = analyse(content, client, max_results=args.number)
        faces = annotations_to_dict(res)
        f.seek(0)
        highlighted = highlight_faces(f, res)

    # save to JSON, and save the highlighted image
    if not args.no_save:
        with open(out, "w") as f:
            json.dump(faces, f)
        with open(out_img, "wb") as f:
            highlighted.save(f)

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


def annotations_to_dict(faces):
    """Converts the results of facial analysis into a Python dictionary

    :param faces: The results to convert
    :type faces: class:`google.cloud.vision.FaceAnnotation`
    :return: The much more intuitively usable dictionary form
    :rtype: dict
    """
    out = []
    for face in faces:
        out_elem = {}
        nest_stack = [out_elem]
        # serialise the data for this face into a string
        lines = face.__str__().split("\n")
        for line in lines:
            if line.strip() == "":  # skip empty lines
                continue
            elif line[-1] == "{":  # start of a new block
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
            elif line[-1] == "}":  # end of a block
                nest_stack.pop()
            else:  # normal key:value pair
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
