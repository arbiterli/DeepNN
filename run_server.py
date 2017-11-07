import cgi
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from model import *


class cnn_handler(BaseHTTPRequestHandler):
    DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
    maybe_download_and_extract(DATA_URL)
    create_graph()
    print('server start...')

    def send_success(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

    def do_GET(self):
        request_type = self.path[1:]
        self.send_success()
        self.wfile.write(request_type.encode('utf-8'))
        return


    def do_POST(self):
        form = cgi.FieldStorage(fp=self.rfile,
            headers=self.headers,
            environ={'REQUEST_METHOD': 'POST',
                     'CONTENT_TYPE': self.headers['Content-Type'],
                     })
        self.send_success()
        for field in form:
            item = form[field]
            if item.filename:
                if item.filename.lower().endswith('.jpg') or item.filename.endswith('.jpeg'):
                    category = classify(item.file.read())
                    self.wfile.write((str(category)).encode('utf-8'))
                else:
                    self.wfile.write('not image'.encode('utf-8'))


if __name__ == '__main__':
    server = HTTPServer(('', 8000), cnn_handler)
    server.serve_forever()