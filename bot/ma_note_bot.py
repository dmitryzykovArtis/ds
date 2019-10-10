from IPython.core.magic import cell_magic, Magics, magics_class
from IPython import get_ipython
import requests

#ADD
TOKEN = 'ADD_TOKEN'
URL = "https://api.telegram.org/bot"
ID = 0


@magics_class
class BotMagics(Magics):
    @cell_magic
    def bot_observe(self, line='', cell=None):
        send_telegram(cell)


def get_me():
    address = URL
    address += TOKEN
    method = address + "/getMe"
    r = requests.post(method)
    print(r.text)


def send_telegram(text: str):
    address = URL
	#ADD
    channel_id = "ADD_ID"
    address += TOKEN
    method = address + "/sendMessage"

    r = requests.post(method, data={
        "chat_id": channel_id,
        "text": text
    })

    if r.status_code != 200:
        print(r.status_code)
        raise Exception("post_text error")


def get_updates():
    address = URL
    address += TOKEN
    method = address + "/getUpdates"

    r = requests.post(method)

    if r.status_code != 200:
        raise Exception("post_text error")
    print(r.text)


def update_id():
    address = URL
    address += TOKEN
    method = address + "/getUpdates"

    r = requests.post(method)

    if r.status_code != 200:
        raise Exception("post_text error")
    answer = r.json()
    global ID
    ID = answer['result'][0]['message']['chat']['id']


ip = get_ipython()
ip.register_magics(BotMagics)

if __name__ == '__main__':
    send_telegram("hello world!")

