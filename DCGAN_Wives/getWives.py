import requests  # http lib
from bs4 import BeautifulSoup  # climb lib
import os # operation system
import traceback # trace deviance

def download(url,filename):
    if os.path.exists(filename):
        print('file exists!')
        return
    try:
        r = requests.get(url,stream=True,timeout=60)
        r.raise_for_status()
        with open(filename,'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk: # filter out keep-alove new chunks
                    f.write(chunk)
                    f.flush()
        return filename
    except KeyboardInterrupt:
        if os.path.exists(filename):
            os.remove(filename)
        return KeyboardInterrupt
    except Exception:
        traceback.print_exc()
        if os.path.exists(filename):
            os.remove(filename)

if os.path.exists('imgs') is False:
    os.makedirs('imgs')

start = 1
end = 8000
for i in range(start, end+1):
    url = 'http://konachan.net/post?page=%d&tags=' % i
    html = requests.get(url).text # gain the web's information
    soup =  BeautifulSoup(html,'html.parser') # doc's string and jie xi qi
    for img in soup.find_all('img',class_="preview"):# 遍历所有preview类，找到img标签
        target_url = img['src']
        filename = os.path.join('../dataset/wives/',target_url.split('/')[-1])
        download(target_url,filename)
    print('%d / %d' % (i,end))

