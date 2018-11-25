#!/usr/bin/python
from http.server import BaseHTTPRequestHandler, HTTPServer
from tooth import *
import time
import json

HOST_NAME = 'localhost'
PORT_NUMBER = 9000

queue = []


class MyHandler(BaseHTTPRequestHandler):
    model = None
    dataset_val = None


    def do_HEAD(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

    def do_POST(self):

         # <--- Gets the size of data
        content_length = int(self.headers['Content-Length'])
        # <--- Gets the data itself
        post_data = self.rfile.read(content_length)
        post_body = json.loads(post_data.decode('utf-8'))
        print('Append for Parse', post_body['image'])
        print('QUEUe LEN', len(queue))
        queue.append(post_body['image'])
        my_num = bool(1) if len(queue) < 2 else bool(0)
        print (my_num)
        while my_num != bool(1):
            my_num = bool(1) if len(queue) < 2 else bool(0)
        try:
            print('FOOOOO POST', post_body['image'])
            for i in range(0,10):
                saveToFile(post_body['image'], self.model, self.dataset_val)
            print('FOOOOO POST DONE')
            # live queue
            queue.remove(post_body['image'])

            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
        except:
            print("Unexpected error:", sys.exc_info()[0])
            queue.remove(post_body['image'])
            pass

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


class http_server:

    def __init__(self, dataset_val, model):
        MyHandler.dataset_val = dataset_val
        MyHandler.model = model

        server = HTTPServer((HOST_NAME, PORT_NUMBER), MyHandler)
        server.serve_forever()


class main:
    def __init__(self):
        self.dataset_val, self.model, self.coco = prepareDatasetAndModel()
        self.server = http_server(self.dataset_val, self.model)


if __name__ == '__main__':
    # server_class = HTTPServer
    # httpd = server_class((HOST_NAME, PORT_NUMBER), MyHandler)
    # print(time.asctime(), 'Server Starts - %s:%s' % (HOST_NAME, PORT_NUMBER))
    # try:
    #     httpd.serve_forever()
    # except KeyboardInterrupt:
    #     pass
    # httpd.server_close()
    # print(time.asctime(), 'Server Stops - %s:%s' % (HOST_NAME, PORT_NUMBER))
    m = main()
