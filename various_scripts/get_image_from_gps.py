
import google_streetview.api as gsv_api
import google_streetview.helpers as gsv_helpers


def get_image_from_gps(coords):
    params = {
        "size": "512x512",
        "location": coords,
        "pitch": "10",
        "radius": "1000",
        "key": "ASK MIKE OR VICTOR",
        "source": "outdoor",
        "fov": "120",
    }
    api_list = gsv_helpers.api_list(params)
    results = gsv_api.results(api_list)
    return results
    
# then: results.download_links(".") 
# change "." to another path 
# if you don't want to download images in the current working directory
