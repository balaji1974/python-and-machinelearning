# import request
import urllib.request
from urllib.request import urlopen


# urlblib is a package that collects several modules for working with URLs
# urlopen is a function that opens the URL
f=urllib.request.urlopen('http://www.example.com')

# read the data from the URL
print(f.read())

# http.client is a module that provides classes for HTTP and HTTPS connections
# HTTPConnection is a class that represents a connection to an HTTP server 
# HTTPConnection can be used to send requests and receive responses
import http.client
from http.client import HTTPConnection

# create an HTTP connection to a server
conn=HTTPConnection("www.example.com")
# send a GET request to the server
# the request is sent to the server and the response is returned
conn.request("GET", "/")
# get the response from the server
# the response is an HTTPResponse object that contains the status code, headers, and body
res=conn.getresponse()
# print the status code and reason
print(res.status, res.reason)
# check the headers
print(res.getheaders())



# http.cookies is a module that provides classes for parsing and creating HTTP cookies
import http.cookies
# SimpleCookie is a class that represents a cookie and its attributes
from http.cookies import SimpleCookie
# create a SimpleCookie object 
cookie = SimpleCookie()

# set a cookie with name, value, and attributes
cookie["name"] = "value"
# set additional attributes for the cookie
# the attributes can be used to specify the domain, path, expiration time, etc.
cookie["name"]["domain"] = "example.com"
cookie["name"]["path"] = "/"
cookie["name"]["expires"] = 3600
cookie["name"]["max-age"] = 3600
cookie["name"]["secure"] = True
cookie["name"]["httponly"] = True
# print the cookie as a string
print(cookie.output())

# urllib.parse is a module that provides functions for parsing and manipulating URLs
import urllib.parse
# urlparse is a function that parses a URL into its components
from urllib.parse import urlparse, urlunparse, urljoin

# parse a URL into its components
url = 'http://www.example.com/'
values = {'s': 'basic',
          'submit': 'Search'}
# urlunparse is a function that constructs a URL from its components
data = urllib.parse.urlencode(values)

data = data.encode('utf-8')  # data should be bytes
req = urllib.request.Request(url, data)
resp = urllib.request.urlopen(req)
respData = resp.read()
print(respData.decode('utf-8'))  # decode the bytes to string


