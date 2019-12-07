"""
This script uses a web scraper to download images from
google images. Parameters can be specified to restrict 
the search to specific keywords, sizes etc.

Here we attempt to download 100 images from each category
listed in keywords.
"""

from google_images_download import google_images_download as gid

response = gid.googleimagesdownload()

keywords = '''Photography nature,
Photography city,
Photography fruits,
Photography landscape,
Photography real estate,
Photography plants,
Photography winter,
Photography autumn,
Photography wildlife,
Photography sunset
Photography mountain,
Photography sports car,
Photography objects
'''

keywords = "".join(keywords.splitlines())

arguments = {'keywords':keywords,
        'limit':100,
        'size':'>2MP',
        'print_urls':True}
paths = response.download(arguments)
print(paths)
