# File to make a GUI based image downloader

import requests
import shutil          # method in Python is used to copy the content of source file to destination file or directory
from lxml import html
from os import path
from tkinter import *
from tkinter.filedialog import askdirectory
from functools import partial

# Implementing download tool
def download(search_key: StringVar, directory: StringVar, images_number: StringVar):
    search_key = "+".join(search_key.get().split(" "))
    directory = directory.get()
    images_number = int(images_number.get())

    if len(search_key) == 0 or not path.isdir(directory):
        return
    
    # Going to be printed on the terminal
    print(f"Keyword : {search_key}, Directory : {directory}, Images number : {images_number}")

    response = requests.get(
        "https://www.google.fr/search?q=" + search_key + "&tbm=isch&ved=2ahUKEwiw_-zyzfLsAhVNYBoKHZa5CJUQ2-cCegQIABAA&oq=dog&gs_lcp=CgNpbWcQA1AAWABgoxhoAHAAeACAAQCIAQCSAQCYAQCqAQtnd3Mtd2l6LWltZw&sclient=img&ei=4ranX7CGJc3AaZbzoqgJ&bih=610&biw=1280")
    tree = html.fromstring(response.content)
    url_list = list(map(lambda x: x.get("src"), tree.xpath("//img")))
    for i in range(1, min([len(url_list), images_number + 1])):
        image_url = url_list[i]
        filename = path.join(directory, r"image_" + search_key + str(i) + ".jpg")

        r = requests.get(image_url, stream=True)

        if r.status_code == 200:
            r.raw.decode_content = True
            with open(filename, 'wb') as f:
                shutil.copyfileobj(r.raw, f)

            print('Image sucessfully Downloaded: ', filename)
        else:
            print('Image Couldn\'t be retreived')

# Function for setting up the directory, where to save the file
def ask_directory():
    directory = askdirectory()
    if len(directory) > 0:
        directory_var.set(directory)


# For creating window using Tkinter
window = Tk()
window.geometry("400x300")
window.title("Images Downloader")

search_key_var = StringVar(value="")
directory_var = StringVar(value=path.expanduser("~\\Desktop"))
images_number_var = StringVar(value="")

Label(window, text="Directory :").pack()
Entry(window, textvariable=directory_var).pack(fill=X)
Button(window, text="Select directory", command=ask_directory).pack()

Label(window, text="Number of images :").pack()
Spinbox(window, state="readonly", from_=1, to=30, textvariable=images_number_var).pack()

Label(window, text="Search keyword :").pack()
Entry(window, textvariable=search_key_var).pack(fill=X)

Button(window, text="Download", command=partial(download, search_key_var, directory_var, images_number_var)).pack()

window.mainloop()
