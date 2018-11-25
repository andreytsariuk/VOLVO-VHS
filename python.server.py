#!/usr/bin/python
from http.server import BaseHTTPRequestHandler, HTTPServer
from tooth import *
import time
import json

HOST_NAME = 'localhost'
PORT_NUMBER = 9000
dataset_val, model, coco = prepareDatasetAndModel()
queue = [];

class MyHandler(BaseHTTPRequestHandler):

    def do_HEAD(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

    def do_POST(self):
         # <--- Gets the size of data
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length) # <--- Gets the data itself
        post_body = json.loads(post_data.decode('utf-8'))
        queue.append(post_body['image'])
        my_num = bool(0)
        while !my_num:
            my_num = bool(1) if queue[0] == post_body['image'] else  bool(0)

        print('FOOOOO POST',post_body['image'])
        saveToFile(post_body['image'], model, dataset_val)
        print('FOOOOO POST DONE')
        #live queue 
        queue.remove(post_body['image'])

        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

    def do_GET(self):
        paths = {
            '/recognize': {'status': 200},
        }

        if self.path in paths:
            self.respond(paths[self.path])
        else:
            self.respond({'status': 500})

    def handle_http(self, status_code, path):
        self.send_response(status_code)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        content = '''
        <html><head><title>Title goes here.</title></head>
        <body><p>This is a test.</p>
        <p>You accessed path: {}</p>
        </body></html>
        '''.format(path)
        return content

    def respond(self, opts):
        response = self.handle_http(opts['status'], self.path)
        self.wfile.write(response)


if __name__ == '__main__':
    server_class = HTTPServer
    httpd = server_class((HOST_NAME, PORT_NUMBER), MyHandler)
    print(time.asctime(), 'Server Starts - %s:%s' % (HOST_NAME, PORT_NUMBER))
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    print(time.asctime(), 'Server Stops - %s:%s' % (HOST_NAME, PORT_NUMBER))


