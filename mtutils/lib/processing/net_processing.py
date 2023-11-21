

def url_validity_check(url, verbose=False):
    from urllib import request
    try:
        url = url.replace(' ', '%20')
        with request.urlopen(url) as file:
            if verbose:
                print(file.status)
                print(file.reason)
            return True
    except Exception as e:
        if verbose:
            print(e)
        return False
