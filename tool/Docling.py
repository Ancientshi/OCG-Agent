import requests

def docling_read_file(file_path):
    url = "http://0.0.0.0:6000/convert"
    json_data = {"file_path": file_path}
    try:
        response = requests.post(url, json=json_data, timeout=60)  
        content = response.json().get("content", "no response")
    except (requests.exceptions.RequestException, ValueError) as e:
        print(f"Request error: {e}")  
        content = "request error"  
    return content

if __name__ == "__main__":
    pass