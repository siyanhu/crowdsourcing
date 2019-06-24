#upstream workspace/crowdapi_app_server {
# fail_timeout=0 means we always retry an upstream even if it failed
# to return a good HTTP response (in case the Unicorn master nukes a
# single worker for timing out).
#  server unix:/home/ubuntu/workspace/crowdapi/run/gunicorn.sock fail_timeout=0;
#}
#
# The default server
#
#server {
#    listen  80;
#    server_name crowd.compathnion.com;
#    return 301 https://crowd.compathnion.com$request_uri;
#}
server {
    listen  80;
    listen 443 default ssl;
    server_name crowd.compathnion.com
    keepalive_timeout   15;
    #ssl on;
    ssl_certificate /etc/nginx/ssl/crowd.compathnion.com.chained.crt;
    ssl_certificate_key /etc/nginx/ssl/crowd.compathnion.com.key;
    ssl_session_cache   shared:SSL:10m;
    ssl_session_timeout 10m;
    ssl_ciphers RC4:HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;

    client_max_body_size 4G;
 
    access_log /home/ubuntu/workspace/crowdapi/logs/nginx-access.log;
    error_log /home/ubuntu/workspace/crowdapi/logs/nginx-error.log;
 
    location /static/ {
        alias   /home/ubuntu/workspace/crowdapi/static/;
    }
    
    location /media/ {
        alias   /home/ubuntu/workspace/crowdapi/media/;
    }
 
    location / {
        # http://www.micahcarrick.com/using-ssl-behind-reverse-proxy-in-django.html
        proxy_set_header X-Forwarded-Proto $scheme;
        # an HTTP header important enough to have its own Wikipedia entry:
        #   http://en.wikipedia.org/wiki/X-Forwarded-For
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
 
        # enable this if and only if you use HTTPS, this helps Rack
        # set the proper protocol for doing redirects:
        # proxy_set_header X-Forwarded-Proto https;
 
        # pass the Host: header from the client right along so redirects
        # can be set properly within the Rack application
        proxy_set_header Host $http_host;
 
        # we don't want nginx trying to do something clever with
        # redirects, we set the Host: header above already.
        proxy_redirect off;
 
        # set "proxy_buffering off" *only* for Rainbows! when doing
        # Comet/long-poll stuff.  It's also safe to set if you're
        # using only serving fast clients with Unicorn + nginx.
        # Otherwise you _want_ nginx to buffer responses to slow
        # clients, really.
        # proxy_buffering off;
       
        # SET THE CONNECTOIN TIMEOUT
        proxy_connect_timeout 20;
        proxy_read_timeout 20;
 
        # Try to serve static files from nginx, no point in making an
        # *application* server like Unicorn/Rainbows! serve static files.
        if (!-f $request_filename) {
            proxy_pass http://127.0.0.1:9000;
            break;
        }
    }
}
