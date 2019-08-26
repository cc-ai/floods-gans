
import google_streetview.api as gsv_api
import google_streetview.helpers as gsv_helpers


def get_image_from_gps(coords: str) -> Result:
    """
    Get a result object containing the link to an street view image
    in the vicinity of the specified lat and long coordinates
    
    use as results.download_links(".") 
    change "." to another path not to download in cwd

    ex: 
    res = get_image_from_gps("45.5307147,-73.6157818")
    res.download_links("~/Downloads") 
    
    """
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
