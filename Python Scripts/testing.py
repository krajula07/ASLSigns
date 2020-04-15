from flask import Flask
from flask import json
from flask import request
import sys
import json
sys.path.append('../')
import lalli

app = Flask(__name__)

@app.route('/')
@app.route('/hello')
def HelloWorld():
    return 'Hello world'

@app.route('/messages', methods = ['POST'])
def api_message():
    if request.headers['Content-Type'] == 'text/plain':
        return "Text Message: " + request.data
    elif request.headers['Content-Type'] == 'application/json':
        request_join=request.get_json()
        with open('your_file.txt', 'w') as outfile:
            json.dump(request.json, outfile)
        lalli.convert('your_file.txt')
        return "CSV Message: " 
    elif request.headers['Content-Type'] == 'application/octet-stream':
        f = open('./binary', 'wb')
        f.write(request.data)
        f.close()
        return "Binary message written!"

    else:
        return "415 Unsupported Media Type ;)"

if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0',port=5000)
