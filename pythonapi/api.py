import flask
import requests
from bs4 import BeautifulSoup
import pandas as pd
import datetime


app = flask.Flask(__name__)
app.config["DEBUG"] = True


@app.route('/', methods=['GET'])
def home():
    #return "<h1>Hello Python!</h1>"
    r = requests.get("https://www.twse.com.tw/exchangeReport/MI_INDEX?response=csv&date=20220421&type=0099P")
    soup = BeautifulSoup(r.text)
    current_dt = datetime.datetime.now().strftime("%Y-%m-%d %X")
    #print(type(soup))
    #print(r.text)
    data = r.text
    stock=[]
    for da in data.split("\n"):
        stock.append([ele.replace('"\r','').replace('"','').replace('=','') for ele in da.split('","')])

    df=pd.DataFrame(stock)
    df.drop(index=[1,2,141,142,143,144,145],axis=1,inplace=True) #remove index1
    df.drop(columns=[15],axis=1,inplace=True) #remove columns15
    #modify column name
    columns=['證券代號','證券名稱','成交股數','成交筆數','成交金額','開盤價','最高價','最低價','收盤價','漲跌','漲跌價差','最後顯示買價','最後顯示買量','最後顯示賣價','最後顯示賣量']
    df.columns=columns
    #insert current time
    df.iloc[140,0] = "Scraping time:"
    df.iloc[140,1] = current_dt
    #print dataframe
    df[1:140]
    #return json
    return df.to_json(orient='records',force_ascii=False)
if __name__=='__main__':
    app.run()