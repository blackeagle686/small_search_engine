import requests


def get_img(name): 
    # URL of the image
    image_url = 'https://example.com/path/to/image.jpg'

    # Send a GET request to fetch the image
    response = requests.get(image_url)

    # Check if the request was successful
    if response.status_code == 200:
        # Open the file in binary write mode and save the image
        with open('downloaded_image.jpg', 'wb') as file:
            file.write(response.content)
        print("Image downloaded successfully.")
    else:
        print("Failed to retrieve the image. HTTP Status code:", response.status_code)
    